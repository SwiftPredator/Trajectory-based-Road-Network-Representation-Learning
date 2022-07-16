import math
import os
import random
from ast import walk
from operator import itemgetter
from turtle import forward

import networkx as nx
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from _walker import random_walks as _random_walks
from scipy import sparse
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.utils.data import DataLoader, Dataset
from torch_geometric.nn import GAE, GATConv, GCNConv, InnerProductDecoder, SGConv
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.norm import LayerNorm
from torch_geometric.utils import (
    add_self_loops,
    from_networkx,
    negative_sampling,
    remove_self_loops,
)
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_sparse import SparseTensor
from tqdm import tqdm

from .model import Model
from .utils import generate_trajid_to_nodeid


class GTNModel(Model):
    def __init__(
        self,
        data,
        device,
        network,
        traj_data,
        init_emb,
        batch_size=64,
        emb_dim=256,
        nlayers=2,
        nheads=4,
        hidden_dim=256,
        max_len=150,
    ):
        self.model = BertModel(
            emb_dim, nlayers, len(network.line_graph.nodes), nheads, hidden_dim, max_len
        )
        self.model.init_token_embed(init_emb)
        self.model = self.model.to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)
        self.device = device
        self.traj_data = traj_data["seg_seq"].tolist()
        self.network = network
        self.traj_to_node = generate_trajid_to_nodeid(network)
        self.train_loader = DataLoader(
            TrajectoryDataset(self.traj_data, self.network, self.traj_to_node),
            batch_size=batch_size,
            shuffle=True,
        )
        self.loss_func = nn.CrossEntropyLoss(reduction="none")

    def train(self, epochs: int = 1000):
        avg_loss = 0
        for e in range(epochs):
            self.model.train()
            total_loss = 0
            for i, data in enumerate(self.train_loader):
                data = {key: value.to(self.device) for key, value in data.items()}
                out = self.model.forward(
                    data["traj_input"], data["input_mask"], data["masked_pos"]
                )

                loss = self.loss_func(out.transpose(1, 2), data["masked_tokens"])
                loss = (loss * data["masked_weights"].float()).mean()

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                avg_loss += loss.item()

                if i % 1000 == 0:
                    print("Epoch: {}, iter {} loss: {}".format(e, i, loss.item()))

            # if e > 0 and e % 1 == 0:
            #     print(
            #         "Epoch: {}, avg_loss: {}".format(
            #             e, avg_loss / len(self.train_loader)
            #         )
            #     )
            #     avg_loss = 0

    def load_model():
        ...


def gelu(x):
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


class TrajectoryTransformer(nn.Module):
    def __init__(
        self, ntokens, emb_dim=128, dropout=0.5, nhead=2, hidden_dim=128, nlayers=2
    ):
        super().__init__()
        self.pos_encoder = PositionalEncoding(emb_dim * 2, dropout)
        encoder_layers = TransformerEncoderLayer(
            emb_dim * 2, nhead, hidden_dim, dropout, batch_first=True
        )
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.encoder = nn.Embedding(ntokens, emb_dim * 2)
        self.decoder = nn.Linear(emb_dim * 2, ntokens)
        self.linear = nn.Linear(emb_dim * 2, emb_dim * 2)
        self.activ2 = gelu
        self.norm = LayerNorm(hidden_dim)
        self.emb_dim = emb_dim

    def init_embed(self, embed) -> None:
        self.encoder.weight.data = embed

    def forward(self, src, src_mask, masked_pos):
        """
        Args:
            src: Tensor, shape [batch_size, seq_len]
            src_mask: Tensor, shape [seq_len, seq_len]

        Returns:
            output Tensor of shape [batch_size, seq_len, ntoken]
        """
        src = self.encoder(src) * math.sqrt(self.emb_dim * 2)
        src = self.pos_encoder(src)
        h = self.transformer_encoder(src, src_key_padding_mask=src_mask)
        masked_pos = masked_pos[:, :, None].expand(-1, -1, h.size(-1))  # B x S x D
        h_masked = torch.gather(h, 1, masked_pos)
        h_masked = self.norm(self.activ2(self.linear(h_masked)))

        output = self.decoder(h_masked)

        return output


class PositionalEncoding(nn.Module):
    def __init__(self, emb_dim: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, emb_dim, 2) * (-math.log(10000.0) / emb_dim)
        )
        pe = torch.zeros(1, max_len, emb_dim)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class TrajectoryDataset(Dataset):
    def __init__(self, trajs, network, traj_map, seq_len=150, mask_ratio=0.25):
        self.trajs = trajs
        self.network = network
        self.seq_len = seq_len
        self.mask_ratio = mask_ratio
        self.traj_map = traj_map

    def __len__(self):
        return len(self.trajs)

    def __getitem__(self, item):

        traj = list(itemgetter(*self.trajs[item])(self.traj_map))  # map to node ids
        if len(traj) > self.seq_len:
            traj = self.cut_traj(traj)

        (
            traj_input,
            traj_masked_tokens,
            traj_masked_pos,
            traj_masked_weights,
            traj_label,
        ) = self.random_word(traj)

        input_mask = [1] * len(traj_input)
        length = [len(traj_input)]

        masked_lenth = len(traj_masked_tokens)
        padding = [0 for _ in range(self.seq_len - len(traj_input))]
        traj_input.extend(padding)
        input_mask.extend(padding)
        traj_label.extend(padding)

        max_pred = int(self.seq_len * self.mask_ratio)
        if max_pred > masked_lenth:
            padding = [0] * (max_pred - masked_lenth)
            traj_masked_tokens.extend(padding)
            traj_masked_pos.extend(padding)
            traj_masked_weights.extend(padding)
        else:
            traj_masked_tokens = traj_masked_tokens[:max_pred]
            traj_masked_pos = traj_masked_pos[:max_pred]
            traj_masked_weights = traj_masked_weights[:max_pred]

        output = {
            "traj_input": traj_input,
            "traj_label": traj_label,
            "input_mask": input_mask,
            "length": length,
            "masked_pos": traj_masked_pos,
            "masked_tokens": traj_masked_tokens,
            "masked_weights": traj_masked_weights,
        }

        return {key: torch.tensor(value) for key, value in output.items()}

    def random_word(self, sentence):
        tokens = sentence
        output_label = []

        mask_len = int(len(tokens) * self.mask_ratio)
        start_loc = round(len(tokens) * random.random() * (1 - self.mask_ratio))

        masked_pos = list(range(start_loc, start_loc + mask_len))
        masked_tokens = tokens[start_loc : start_loc + mask_len]
        masked_weights = [1] * len(masked_tokens)

        for i, token in enumerate(tokens):
            if i >= start_loc and i < start_loc + mask_len:
                tokens[i] = 1
                output_label.append(token)
            else:
                output_label.append(0)

        assert len(tokens) == len(output_label)

        return tokens, masked_tokens, masked_pos, masked_weights, output_label

    def cut_traj(self, traj):
        start_idx = int((len(traj) - self.seq_len) * random.random())
        return traj[start_idx : start_idx + self.seq_len]


##### Toast Transformer implementation


class LayerNorm(nn.Module):
    def __init__(self, emb_dim, variance_epsilon=1e-12):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(emb_dim))
        self.beta = nn.Parameter(torch.zeros(emb_dim))
        self.variance_epsilon = variance_epsilon

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.gamma * x + self.beta


class Embeddings(nn.Module):
    def __init__(self, ntokens, emb_dim, max_len):
        super(Embeddings, self).__init__()
        self.tok_embed = nn.Embedding(ntokens, emb_dim)  # token embeddind
        self.pos_embed = nn.Embedding(max_len, emb_dim)  # position embedding
        self.norm = LayerNorm(emb_dim)
        self.drop = nn.Dropout(0.5)

    def forward(self, x):
        seq_len = x.size(1)
        pos = torch.arange(seq_len, dtype=torch.long, device=x.device)
        pos = pos.unsqueeze(0).expand_as(x)  # (S,) -> (B, S)

        e = self.tok_embed(x)  # + self.pos_embed(pos)
        res = self.drop(self.norm(e))
        return res


def split_last(x, shape):
    shape = list(shape)
    assert shape.count(-1) <= 1
    if -1 in shape:
        shape[shape.index(-1)] = int(x.size(-1) / -np.prod(shape))
    return x.view(*x.size()[:-1], *shape)


def merge_last(x, n_dims):
    s = x.size()
    assert n_dims > 1 and n_dims < len(s)
    return x.view(*s[:-n_dims], -1)


class MultiHeadedSelfAttention(nn.Module):
    def __init__(self, emb_dim, n_heads):
        super(MultiHeadedSelfAttention, self).__init__()
        self.proj_q = nn.Linear(emb_dim, emb_dim)
        self.proj_k = nn.Linear(emb_dim, emb_dim)
        self.proj_v = nn.Linear(emb_dim, emb_dim)
        self.drop = nn.Dropout(0.5)
        self.scores = None  # for visualization
        self.n_heads = n_heads

    def forward(self, x, mask):
        q, k, v = self.proj_q(x), self.proj_k(x), self.proj_v(x)
        q, k, v = (split_last(x, (self.n_heads, -1)).transpose(1, 2) for x in [q, k, v])

        scores = q @ k.transpose(-2, -1) / np.sqrt(k.size(-1))
        if mask is not None:
            mask = mask[:, None, None, :].float()
            scores -= 10000.0 * (1.0 - mask)
        scores = self.drop(F.softmax(scores, dim=-1))

        h = (scores @ v).transpose(1, 2).contiguous()
        # -merge-> (B, S, D)
        h = merge_last(h, 2)
        self.scores = scores
        return h


class PositionWiseFeedForward(nn.Module):
    def __init__(self, emb_dim, hidden_dim):
        super(PositionWiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(emb_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, emb_dim)

    def forward(self, x):
        return self.fc2(gelu(self.fc1(x)))


class Block(nn.Module):
    def __init__(self, emb_dim, hidden_dim, n_heads):
        super(Block, self).__init__()
        self.attn = MultiHeadedSelfAttention(emb_dim, n_heads)
        self.proj = nn.Linear(emb_dim, emb_dim)
        self.norm1 = LayerNorm(emb_dim)
        self.pwff = PositionWiseFeedForward(emb_dim, hidden_dim)
        self.norm2 = LayerNorm(emb_dim)
        self.drop = nn.Dropout(0.5)

    def forward(self, x, mask):
        h = self.attn(x, mask)
        h = self.norm1(x + self.drop(self.proj(h)))
        h = self.norm2(h + self.drop(self.pwff(h)))
        return h


class Transformer(nn.Module):
    def __init__(self, emb_dim, nlayers, ntokens, nheads, hidden_dim, max_len):
        super(Transformer, self).__init__()
        self.fc = nn.Linear(emb_dim, emb_dim)
        self.embed = Embeddings(ntokens, emb_dim, max_len)
        self.blocks = nn.ModuleList(
            [Block(emb_dim, hidden_dim, nheads) for _ in range(nlayers)]
        )

    def forward(self, x, mask):
        h = self.fc(self.embed(x))
        for block in self.blocks:
            h = block(h, mask)
        return h


class BertModel(nn.Module):
    def __init__(self, emb_dim, n_layers, ntokens, nheads, hidden_dim, max_len):
        super(BertModel, self).__init__()
        self.transformer = Transformer(
            emb_dim, n_layers, ntokens, nheads, hidden_dim, max_len
        )
        self.fc = nn.Linear(emb_dim, emb_dim)
        self.activ1 = nn.Tanh()
        self.linear = nn.Linear(emb_dim, emb_dim)
        self.activ2 = gelu
        self.norm = LayerNorm(emb_dim)
        self.classifier = nn.Linear(emb_dim, 2)

        embed_weight = self.transformer.embed.tok_embed.weight
        n_vocab, n_dim = embed_weight.size()
        self.decoder = nn.Linear(n_dim, n_vocab)

    def init_token_embed(self, embed):
        token_vocab = self.transformer.embed.tok_embed.weight.shape[0]

        if embed.shape[0] < token_vocab:
            self.transformer.embed.tok_embed.weight.data[
                token_vocab - embed.shape[0] :
            ] = embed
            print(self.transformer.embed.tok_embed.weight.shape)
        else:
            self.transformer.embed.tok_embed.weight.data = embed

    def forward(self, input_ids, input_mask, masked_pos):
        h = self.transformer(input_ids, input_mask)  # B x S x D
        # pooled_h = self.activ1(self.fc(h[:, 0]))

        masked_pos = masked_pos[:, :, None].expand(-1, -1, h.size(-1))  # B x S x D
        h_masked = torch.gather(h, 1, masked_pos)
        h_masked = self.norm(self.activ2(self.linear(h_masked)))
        # logits_lm = self.decoder(h_masked) + self.decoder_bias
        logits_lm = self.decoder(h_masked)

        return logits_lm
