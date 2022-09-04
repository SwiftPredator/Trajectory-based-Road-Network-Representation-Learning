from operator import itemgetter

import torch
import torch.nn as nn


class ContextualEmbeddingPlugin(nn.Module):
    def __init__(self, transformer: nn.Module, device):
        super().__init__()
        self.transformer = transformer
        self.device = device
        self.h_dim = self.transformer.embed.tok_embed.weight.shape[1]
        self.reduc = nn.Linear(self.h_dim * 2, self.h_dim)

        self.reduc.to(device)

    def forward(self, x):
        mask = self.mask.to(self.device)
        lengths = self.lengths.to(self.device)

        h = self.transformer(self.seq_batch.to(self.device), mask)
        traj_h = (
            torch.sum(h * mask.unsqueeze(-1).float(), dim=1)
            # / lengths.unsqueeze(1).float()
        )
        x = traj_h  # torch.concat([x, traj_h], dim=1)
        # x = self.reduc(x)

        return x

    def register_id_seq(self, X, mask, map, lengths):
        seq_batch = X.clone()
        for i, seq in enumerate(seq_batch):
            emb_ids = itemgetter(*seq[mask[i]].tolist())(map)
            seq_batch[i, mask[i]] = torch.tensor(emb_ids, dtype=int) + 2

        self.seq_batch = seq_batch
        self.mask = mask
        self.lengths = lengths
