from turtle import forward
import torch.nn as nn
import torch

from . import Model, GTCSubConv
from .utils import generate_torch_datasets
from torch.utils.data import DataLoader, Dataset
import torch_geometric.transforms as T
from torch_geometric.data import Data
from torch_geometric.nn import InnerProductDecoder
from tqdm import tqdm
from torch_geometric.utils import (
    add_self_loops,
    from_networkx,
    negative_sampling,
    remove_self_loops,
)

from .utils import recon_loss
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint_sequential
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from enum import Enum


class ModelVariant(Enum):
    TGCN = 1
    TGTC_BASE = 2
    TGTC_ATT = 3
    TGTC_FUSION = 4
    EXPERIMENTAL = 5


class TemporalGraphTrainer(Model):
    """
    Main idea behind this is to add another axis into the embedding, namely an axis for time.
    The goal is to model the change over time in the road network and therefore improve performance on several
    its applications like forecasting of traffic, trajectory task and so on.

    :Note: Add time features to the feature matrix like hour day of week and probably month to basically stamp the generated
    embeddings from the gtc model. Maybe it would be a good idea to emebed the time features seperately and concat them to the gtc_embed

    Args:
        Model (_type_): _description_
    """

    def __init__(
        self,
        data,
        device,
        adj,
        model_type: ModelVariant,
        edge_index=None,
        struc_emb=None,
        batch_size: int = 128,
        hidden_dim: int = 128,
        device_ids=[],
        log_name="model",
    ):
        super().__init__()

        if struc_emb is not None:
            struc_emb = torch.Tensor(struc_emb).to(device)

        self.batch_size = batch_size
        self.model = TemporalGraphModel(
            input_dim=data.shape[-1],
            adj=adj,
            device=device,
            hidden_dim=hidden_dim,
            num_nodes=data.shape[1],
            struc_emb=struc_emb,
            model_type=model_type,
            learning_rate=1e-3,
            batch_size=batch_size,
            edge_index=edge_index,
        )
        # if len(device_ids) > 1:
        #     self.model = nn.DataParallel(self.model, device_ids=device_ids)
        self.model.to(device)
        # self.model.gradient_checkpointing_enable()
        # self.optimizer = torch.optim.Adam(
        #     self.model.parameters(), lr=0.001, weight_decay=1e-5
        # )
        # self.loss_seq = nn.MSELoss(reduction="mean")
        self.device = device
        # data needs to be shape TxNxF T temporal, N nodes, F features
        train, validation = generate_torch_datasets(
            data=data, seq_len=12, pre_len=3, reconstruct=True, split_ratio=0.9
        )
        self.train_loader = DataLoader(
            train,
            batch_size=self.batch_size,
            shuffle=True,
        )
        self.eval_loader = DataLoader(
            validation,
            batch_size=self.batch_size,
        )
        logger = TensorBoardLogger("tb_logs", name=log_name, log_graph=True)
        early_stop_callback = EarlyStopping(
            monitor="val_loss", min_delta=0.005, patience=3, verbose=True, mode="min"
        )
        self.trainer = pl.Trainer(
            accelerator="gpu",
            devices=device_ids,
            auto_lr_find=True,
            logger=logger,
            max_epochs=100,
            callbacks=[early_stop_callback],
            enable_checkpointing=True,
        )
        self.data = data
        self._adj = adj
        self._hidden_dim = hidden_dim
        self._edge_index = edge_index
        # self._struc_emb = struc_emb

        # norm data
        for i in range(self.data.shape[-1]):
            max_val = self.data[:, :, i].max()
            self.data[:, :, i] = self.data[:, :, i] / max_val

    def train(self):
        self.trainer.fit(self.model, self.train_loader, self.eval_loader)

    def find_best_lr(self):
        self.trainer.tune(
            self.model,
            self.train_loader,
            self.eval_loader,
            lr_find_kwargs={"num_training": 100},
        )

        return self.model.learning_rate

    # def train(self, epochs):
    #     self.model.train()
    #     for e in range(epochs):
    #         total_loss = 0
    #         seq_loss = 0
    #         reconst_loss = 0
    #         for X, y in tqdm(self.train_loader, leave=False):
    #             X = X.to(self.device)
    #             z, seq_recon = self.model(X)
    #             z = (z.sum(axis=0) / self.batch_size).squeeze()

    #             y = y.to(self.device)

    #             loss_1 = self.loss_seq(seq_recon, y) * 100
    #             loss_2 = recon_loss(z, self._edge_index)
    #             loss = loss_1 + loss_2

    #             self.optimizer.zero_grad()
    #             loss.backward()
    #             self.optimizer.step()
    #             total_loss += loss.item()
    #             seq_loss += loss_1.item()
    #             reconst_loss += loss_2.item()

    #         print(
    #             f"Average training loss in episode {e} -- Total: {total_loss/len(self.train_loader)}, Seq: {seq_loss/len(self.train_loader)}, Recon: {reconst_loss/len(self.train_loader)}"
    #         )

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path, map_location=self.device))

    def load_emb(self):
        self.model.eval()
        z, _ = self.model(torch.Tensor(self.data[100:112]).unsqueeze(0).to(self.device))
        return z.squeeze().detach().cpu().numpy()


class TemporalGraphModel(pl.LightningModule):
    def __init__(
        self,
        adj,
        num_nodes,
        learning_rate,
        batch_size,
        edge_index,
        model_type: ModelVariant,
        input_dim: int,
        device=None,
        struc_emb=None,
        hidden_dim: int = 128,
    ):
        super(TemporalGraphModel, self).__init__()
        self.register_buffer("adj", torch.FloatTensor(adj))
        _model = None
        if model_type.value == ModelVariant.TGTC_BASE.value:
            _model = TemporalGraphEncoder
        elif model_type.value == ModelVariant.TGCN.value:
            _model = TemporalConvolutionEncoder
        elif model_type.value == ModelVariant.TGTC_FUSION.value:
            _model = TemporalFusionEncoder
        elif model_type.value == ModelVariant.TGTC_ATT.value:
            _model = TemporalAttentionEncoder
        elif model_type.value == ModelVariant.EXPERIMENTAL.value:
            _model = ExperimentalEncoder

        self.encoder = _model(
            input_dim=input_dim,
            adj=self.adj,
            device=device,
            hidden_dim=hidden_dim,
            struc_emb=struc_emb,
        )
        self.decoder = TemporalGraphDecoder(
            input_dim=hidden_dim, hidden_dim=hidden_dim, num_nodes=num_nodes
        )

        # self.device = device
        self.batch_size = batch_size
        self._edge_index = edge_index
        self.learning_rate = learning_rate
        self.loss_seq = nn.MSELoss(reduction="mean")

    def forward(self, X):
        z = self.encoder(X)
        seq_recon = self.decoder(z, X.shape[1])

        return z, seq_recon

    def training_step(self, batch, batch_idx):
        X, y = batch
        z, seq_recon = self.forward(X)
        z = (z.sum(axis=0) / self.batch_size).squeeze()

        loss_1 = self.loss_seq(seq_recon, y) * 100
        loss_2 = recon_loss(z, self._edge_index)
        loss = loss_1 + loss_2

        self.log("train_seq_loss", loss_1, logger=True)
        self.log("train_graph_loss", loss_2, logger=True)
        self.log("train_loss", loss, logger=True)

        return loss

    def validation_step(self, val_batch, val_batch_idx):
        X, y = val_batch
        z = self.encoder(X)
        seq_recon = self.decoder(z, X.shape[1])
        z = (z.sum(axis=0) / self.batch_size).squeeze()

        loss_1 = self.loss_seq(seq_recon, y) * 100
        loss_2 = recon_loss(z, self._edge_index)
        loss = loss_1 + loss_2

        self.log("val_seq_loss", loss_1, logger=True)
        self.log("val_graph_loss", loss_2, logger=True)
        self.log("val_loss", loss, logger=True)

    def custom_histogram_adder(self):
        for name, params in self.named_parameters():
            self.logger.experiment.add_histogram(name, params, self.current_epoch)

    def training_epoch_end(self, outputs) -> None:
        # if self.current_epoch == 1:
        #     sample = torch.rand(1, 12, 100, 4)
        #     self.logger.experiment.add_graph(self, sample)
        self.custom_histogram_adder()
        return super().training_epoch_end(outputs)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer


class TemporalGraphDecoder(nn.Module):
    def __init__(self, input_dim: int, num_nodes, hidden_dim: int = 128):
        super(TemporalGraphDecoder, self).__init__()
        self._tdecoder = nn.LSTM(input_dim, hidden_dim, num_layers=1, batch_first=True)
        self.dense = nn.Sequential(
            nn.Linear(hidden_dim, int(hidden_dim / 2)),
            nn.ReLU(),
            nn.Linear(int(hidden_dim / 2), 1),
        )
        self._hidden_dim = hidden_dim

    def forward(self, z, seq_len):
        # stack z to match seq leng
        # BxNxF
        batch_size, num_nodes, emb_dim = z.shape
        # hidden_state = torch.zeros(batch_size, num_nodes, self._hidden_dim).type_as(z)
        z = z.reshape(batch_size * num_nodes, 1, emb_dim).repeat(1, seq_len, 1)
        x, _ = self._tdecoder(z)
        reconstruct = self.dense(
            x.reshape(batch_size * num_nodes * seq_len, self._hidden_dim)
        )
        reconstruct = reconstruct.reshape(batch_size, seq_len, num_nodes)
        # # encode sequence with graph conv
        # reconstruct = torch.zeros(batch_size, seq_len, num_nodes, 1)
        # for i in range(seq_len):
        #     output, hidden_state = self._tdecoder(z, hidden_state)
        #     output = self.dense(output)
        #     output = output.reshape((batch_size, num_nodes, self._hidden_dim))
        #     reconstruct[:, i, :, :] = output

        return reconstruct


class TemporalGraphEncoder(nn.Module):
    def __init__(
        self,
        adj,
        input_dim: int,
        hidden_dim: int = 128,
        device=None,
        struc_emb=None,
    ):
        super(TemporalGraphEncoder, self).__init__()
        # self._gtc_conv = TGTCConvolution(adj, hidden_dim, hidden_dim, bias=1.0)
        # self._lstm_cell = nn.GRUCell(
        #     adj.shape[0] * hidden_dim,
        #     adj.shape[0] * hidden_dim,
        # )

        self._encoder = TGTCCellLSTM(
            adj,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            use_attention=False,
        )

        # if use_attention:
        #     # self._attention = ConcatAttention(hidden_dim=hidden_dim)
        #     # self._self_attention = nn.MultiheadAttention(
        #     #     hidden_dim, 4, batch_first=True
        #     # )
        #     # self.posff = nn.Sequential(
        #     #     nn.Linear(hidden_dim, 512), nn.ELU(), nn.Linear(512, 128)
        #     # )
        #     self._static_conv = TGTCConvolution(adj, 0, hidden_dim, 3)
        #     self.trans = nn.Sequential(
        #         nn.Linear(hidden_dim * 3, 512),
        #         nn.ELU(),
        #         nn.GroupNorm(8, 512),
        #         nn.Linear(512, 128),
        #     )

        #     self.gnorm = nn.GroupNorm(3, hidden_dim * 4)
        #     self.lnorm = nn.LayerNorm(hidden_dim * 4)
        #     # attention try 2
        #     # self.proj = nn.Linear(hidden_dim, hidden_dim)
        #     # self.lnorm_1 = nn.LayerNorm(hidden_dim)
        #     # self.lnorm_2 = nn.LayerNorm(hidden_dim)
        #     # self.drop = nn.Dropout(p=0.5)

        #     # self.lnorm_3 = nn.LayerNorm(hidden_dim)
        #     # self._lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers=1, batch_first=True)
        #     # self._transformer = nn.TransformerEncoder(
        #     #     encoder_layer=nn.TransformerEncoderLayer(d_model=128, nhead=4),
        #     #     num_layers=1,
        #     # )

        #     # self._self_attention = GraphSeqAttention(hidden_dim, num_nodes)
        #     # self.att_transform = nn.Linear(12, hidden_dim)  # seq_len -> hidden_dim

        #     # self.transform = nn.Linear(hidden_dim, hidden_dim)
        #     # self.dropout = nn.Dropout(p=0.3)
        #     # self.glu = nn.GLU()
        #     # self.norm = nn.LayerNorm(hidden_dim)
        #     # self.resgate_1 = GatedResidualNetwork(hidden_dim=hidden_dim)
        #     # self.resgate_2 = GatedResidualNetwork(hidden_dim=hidden_dim)
        #     # self.gate_linear_1 = nn.Linear(hidden_dim, hidden_dim)
        #     # self.gate_linear_2 = nn.Linear(hidden_dim, hidden_dim)

        self._hidden_dim = hidden_dim
        self._struc_emb = struc_emb
        self._device = device

    def apply_gate(self, x, skip=None, dropout=True):
        if dropout:
            x = self.dropout(x)
        x = self.glu(x.repeat(1, 2))
        if skip != None:
            x = x + skip
        x = self.norm(x)

        return x

    def forward(self, x):
        batch_size, seq_len, num_nodes, num_feat = x.shape
        # x_dynamic, x_static = x[:, :, :, 0].unsqueeze(-1), x[:, 0, :, 1:]
        struc_emb_unrolled = (
            self._struc_emb.unsqueeze(0)
            .repeat(batch_size, 1, 1)
            .reshape(batch_size, num_nodes * self._hidden_dim)
        )
        hidden_state = (
            struc_emb_unrolled.clone()
        )  # torch.zeros(batch_size, num_nodes * self._hidden_dim).type_as(x)
        cell_state = struc_emb_unrolled.clone()
        output = None
        # hidden_sequence = torch.zeros(
        #     size=(seq_len, batch_size, num_nodes, self._hidden_dim)
        # )
        # encode sequence with graph conv and generate lstm sequence embedding
        for i in range(seq_len):
            input = x[
                :, i, :, :
            ]  # if not self._use_attention else x_dynamic[:, i, :, :]
            hidden_state, cell_state, _ = self._encoder(input, hidden_state, cell_state)
            output = hidden_state.reshape((batch_size, num_nodes, self._hidden_dim))
            # conv = conv.reshape((batch_size, num_nodes, self._hidden_dim))
            # if self._use_attention:
            #     # gated = self.apply_gate(output.reshape(-1, self._hidden_dim)).reshape(
            #     #     (batch_size, num_nodes, self._hidden_dim)
            #     # )
            #     # out, skip = self.resgate_1(gated, self._struc_emb)
            #     # out = self.apply_gate(out, skip)
            #     # hidden_sequence[i, :, :, :] = out.reshape(
            #     #     ((batch_size, num_nodes, self._hidden_dim))
            #     # )
            #     hidden_sequence[i, :, :, :] = output

        # if self._use_attention:
        #     # propagte embedding into multihead attention to generate context vector
        #     hidden_sequence = hidden_sequence.reshape(
        #         seq_len, batch_size * num_nodes, self._hidden_dim
        #     )
        #     hidden_sequence = hidden_sequence.permute((1, 0, 2))
        #     hidden_sequence = hidden_sequence.to(self._device)
        #     # context = self._attention(
        #     #     output, self._struc_emb.unsqueeze(0).repeat(batch_size, 1, 1)
        #     # )
        #     # context, weights = self._self_attention(
        #     #     hidden_sequence,
        #     #     hidden_sequence,
        #     #     hidden_sequence,
        #     #     need_weights=True,
        #     #     average_attn_weights=True,
        #     # )
        #     conv_struc = self._static_conv(x_static).reshape(
        #         (batch_size * num_nodes, self._hidden_dim)
        #     )
        #     # context = self.lnorm_1(hidden_sequence + self.drop(self.proj(context)))
        #     # context = self.lnorm_2(context + self.drop(self.posff(context)))

        #     # out = context[:, -1, :].squeeze()
        #     # context = torch.concat((context, conv_struc), dim=-1)
        #     # out = checkpoint_sequential(self.posff, 2, context)
        #     out = torch.concat(
        #         [
        #             output.reshape(batch_size * num_nodes, self._hidden_dim),
        #             conv_struc.reshape(batch_size * num_nodes, self._hidden_dim),
        #             struc_emb_unrolled.reshape(
        #                 batch_size * num_nodes, self._hidden_dim
        #             ),
        #         ],
        #         dim=-1,
        #     )
        #     # out = self.lnorm(out)
        #     out = checkpoint_sequential(self.trans, 2, out)
        #     # out, _ = self._lstm(context)

        #     # context = context.mean(1).squeeze()

        #     # context = context.reshape(-1, seq_len)
        #     # context = self.att_transform(context)
        #     # context = context.reshape(batch_size, num_nodes, self._hidden_dim)

        #     # context = self._self_attention(
        #     #     output, self._struc_emb.unsqueeze(0).repeat(batch_size, 1, 1)
        #     # )

        #     # context = context[:, -1, :].squeeze()
        #     # context = self.norm(context)
        #     # context = self.transform(context)
        #     # context = self.apply_gate(context, output.reshape(-1, self._hidden_dim))  #
        #     # context = context.reshape((batch_size, num_nodes, self._hidden_dim))
        #     # context = context.reshape((batch_size, num_nodes, self._hidden_dim))
        #     # context, skip = self.resgate_2(context)
        #     # context = self.apply_gate(context, skip)
        #     # context = self.apply_gate(
        #     #     context,
        #     #     gated.reshape((batch_size * num_nodes, self._hidden_dim)),
        #     #     dropout=False,
        #     # )
        #     out = out.reshape((batch_size, num_nodes, self._hidden_dim))

        #     return out

        # print(context.shape)
        # Z -> BxNxE
        return output


class TemporalAttentionEncoder(nn.Module):
    def __init__(
        self, adj, input_dim: int, hidden_dim: int = 128, device=None, struc_emb=None
    ):
        super(TemporalAttentionEncoder, self).__init__()

        self._encoder = TGTCCellLSTM(
            adj,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
        )

        self._self_attention = nn.MultiheadAttention(hidden_dim, 4, batch_first=True)
        self._lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers=1, batch_first=True)

        self._hidden_dim = hidden_dim
        self._device = device

    def forward(self, x):
        batch_size, seq_len, num_nodes, num_feat = x.shape
        x_dynamic, x_static = x[:, :, :, 0].unsqueeze(-1), x[:, 0, :, 1:]
        hidden_state = torch.zeros(batch_size, num_nodes * self._hidden_dim).type_as(x)
        cell_state = torch.zeros(batch_size, num_nodes * self._hidden_dim).type_as(x)
        output = None

        hidden_sequence = torch.zeros(
            size=(seq_len, batch_size, num_nodes, self._hidden_dim)
        )

        # encode sequence with graph conv and generate gru sequence embedding
        for i in range(seq_len):
            input = x[:, i, :, :]
            hidden_state, cell_state = self._encoder(input, hidden_state, cell_state)
            output = hidden_state.reshape((batch_size, num_nodes, self._hidden_dim))
            hidden_sequence[i, :, :, :] = output

        hidden_sequence = hidden_sequence.reshape(
            seq_len, batch_size * num_nodes, self._hidden_dim
        )
        hidden_sequence = hidden_sequence.permute((1, 0, 2))
        hidden_sequence = hidden_sequence.to(self._device)

        context, weights = self._self_attention(
            hidden_sequence,
            hidden_sequence,
            hidden_sequence,
            need_weights=True,
            average_attn_weights=True,
        )
        # context = context.sum(1).squeeze()
        hidden_state_2 = torch.zeros(
            1, batch_size * num_nodes, self._hidden_dim
        ).type_as(x)
        cell_state_2 = torch.zeros(1, batch_size * num_nodes, self._hidden_dim).type_as(
            x
        )
        context, _ = self._lstm(context, (hidden_state_2, cell_state_2))
        context = context[:, -1, :]

        output = context.reshape((batch_size, num_nodes, self._hidden_dim))

        # Z -> BxNxE
        return output


class ExperimentalEncoder(nn.Module):
    def __init__(
        self, adj, input_dim: int, hidden_dim: int = 128, device=None, struc_emb=None
    ):
        super(ExperimentalEncoder, self).__init__()

        self._encoder = TGTCCellLSTM(
            adj,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
        )
        self._encoder2 = nn.LSTMCell(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
        )
        # TGTCCellLSTM(adj, input_dim=hidden_dim, hidden_dim=hidden_dim)

        # self._self_attention = nn.MultiheadAttention(hidden_dim, 4, batch_first=True)
        self._self_attention = DotProductAtt()

        self._hidden_dim = hidden_dim
        self._device = device

    def forward(self, x):
        batch_size, seq_len, num_nodes, num_feat = x.shape
        # x_dynamic, x_static = x[:, :, :, 0].unsqueeze(-1), x[:, 0, :, 1:]
        hidden_state = torch.zeros(batch_size, num_nodes * self._hidden_dim).type_as(x)
        cell_state = torch.zeros(batch_size, num_nodes * self._hidden_dim).type_as(x)
        hx = torch.zeros(num_nodes * batch_size, self._hidden_dim).type_as(x)
        cx = torch.zeros(num_nodes * batch_size, self._hidden_dim).type_as(x)
        embedding = torch.zeros(num_nodes).type_as(x)
        output = None

        hidden_sequence = torch.zeros(
            size=(seq_len, batch_size, num_nodes, self._hidden_dim)
        )

        # encode sequence with graph conv and generate gru sequence embedding
        for i in range(seq_len):
            input = x[:, i, :, :]
            hidden_state, cell_state = self._encoder(input, hidden_state, cell_state)
            output = hidden_state.reshape((batch_size, num_nodes, self._hidden_dim))
            hidden_sequence[i, :, :, :] = output

        hidden_sequence = hidden_sequence.reshape(
            seq_len, batch_size * num_nodes, self._hidden_dim
        )
        hidden_sequence = hidden_sequence.permute((1, 0, 2))
        hidden_sequence = hidden_sequence.to(self._device)
        for i in range(seq_len):
            # hx = hx.reshape(batch_size * num_nodes, self._hidden_dim)
            # hx = hx.unsqueeze(0).permute(1, 0, 2).repeat(1, seq_len, 1)
            # context, _ = self._self_attention(
            #     query=hidden_sequence, key=hx, value=hidden_sequence
            # )
            context = self._self_attention(hx, hidden_sequence)
            # hx = hx.squeeze()  # .reshape(batch_size, self._hidden_dim * num_nodes)
            # context = (
            #     context.squeeze()
            # )  # .reshape(batch_size, num_nodes, self._hidden_dim)
            hx, cx = self._encoder2(context.squeeze(), (hx, cx))
            output = hx.reshape((batch_size, num_nodes, self._hidden_dim))

        output = output.reshape((batch_size, num_nodes, self._hidden_dim))

        # Z -> BxNxE
        return output


class ExperimentalEncoder2(nn.Module):
    def __init__(
        self, adj, input_dim: int, hidden_dim: int = 128, device=None, struc_emb=None
    ):
        super(ExperimentalEncoder2, self).__init__()

        self._encoder = TGTCCellLSTM(
            adj,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
        )
        self._encoder2 = TGTCCellLSTM(adj, input_dim=hidden_dim, hidden_dim=hidden_dim)

        self._self_attention = nn.MultiheadAttention(hidden_dim, 4, batch_first=True)
        # self._self_attention = DotProductAtt()

        self._hidden_dim = hidden_dim
        self._device = device

    def forward(self, x):
        batch_size, seq_len, num_nodes, num_feat = x.shape
        # x_dynamic, x_static = x[:, :, :, 0].unsqueeze(-1), x[:, 0, :, 1:]
        hidden_state = torch.zeros(batch_size, num_nodes * self._hidden_dim).type_as(x)
        cell_state = torch.zeros(batch_size, num_nodes * self._hidden_dim).type_as(x)
        hx = torch.zeros(num_nodes * batch_size, self._hidden_dim).type_as(x)
        cx = torch.zeros(num_nodes * batch_size, self._hidden_dim).type_as(x)
        embedding = torch.zeros(num_nodes).type_as(x)
        output = None

        hidden_sequence = torch.zeros(
            size=(seq_len, batch_size, num_nodes, self._hidden_dim)
        )

        # encode sequence with graph conv and generate gru sequence embedding
        for i in range(seq_len):
            input = x[:, i, :, :]
            hidden_state, cell_state = self._encoder(input, hidden_state, cell_state)
            output = hidden_state.reshape((batch_size, num_nodes, self._hidden_dim))
            hidden_sequence[i, :, :, :] = output

        hidden_sequence = hidden_sequence.reshape(
            seq_len, batch_size * num_nodes, self._hidden_dim
        )
        hidden_sequence = hidden_sequence.permute((1, 0, 2))
        hidden_sequence = hidden_sequence.to(self._device)
        context = self._self_attention(
            hidden_sequence, hidden_sequence, hidden_sequence
        )
        for i in range(seq_len):
            input = context[:, i, :]
            hx, cx = self._encoder2(input, (hx, cx))
            output = hx.reshape((batch_size, num_nodes, self._hidden_dim))

        output = output.reshape((batch_size, num_nodes, self._hidden_dim))

        # Z -> BxNxE
        return output


class TemporalFusionEncoder(nn.Module):
    def __init__(
        self, adj, input_dim: int, hidden_dim: int = 128, device=None, struc_emb=None
    ):
        super(TemporalFusionEncoder, self).__init__()

        self._encoder = TGTCCellLSTM(
            adj, input_dim=1, hidden_dim=hidden_dim, use_attention=True
        )

        self._static_conv = TGTCConvolution(adj, 0, hidden_dim, 3)
        self.trans = nn.Sequential(
            nn.Linear(hidden_dim * 3, 512),
            nn.ELU(),
            nn.GroupNorm(8, 512),
            nn.Linear(512, 128),
        )

        self.gnorm = nn.GroupNorm(3, hidden_dim * 4)
        self.lnorm = nn.LayerNorm(hidden_dim * 4)

        self._hidden_dim = hidden_dim
        self._device = device
        self._struc_emb = struc_emb

    def forward(self, x):
        batch_size, seq_len, num_nodes, num_feat = x.shape
        struc_emb_unrolled = (
            self._struc_emb.unsqueeze(0)
            .repeat(batch_size, 1, 1)
            .reshape(batch_size, num_nodes * self._hidden_dim)
        )

        x_dynamic, x_static = x[:, :, :, 0].unsqueeze(-1), x[:, 0, :, 1:]
        hidden_state = torch.zeros(batch_size, num_nodes * self._hidden_dim).type_as(x)
        cell_state = torch.zeros(batch_size, num_nodes * self._hidden_dim).type_as(x)
        output = None

        # encode sequence with graph conv and generate gru sequence embedding
        for i in range(seq_len):
            input = x_dynamic[:, i, :, :]
            hidden_state, cell_state = self._encoder(input, hidden_state, cell_state)
            output = hidden_state.reshape((batch_size, num_nodes, self._hidden_dim))

        conv_struc = self._static_conv(x_static).reshape(
            (batch_size * num_nodes, self._hidden_dim)
        )
        out = torch.concat(
            [
                output.reshape(batch_size * num_nodes, self._hidden_dim),
                conv_struc.reshape(batch_size * num_nodes, self._hidden_dim),
                struc_emb_unrolled.reshape(batch_size * num_nodes, self._hidden_dim),
            ],
            dim=-1,
        )
        out = checkpoint_sequential(self.trans, 2, out)
        out = out.reshape((batch_size, num_nodes, self._hidden_dim))

        # Z -> BxNxE
        return out


class TemporalConvolutionEncoder(nn.Module):
    def __init__(
        self, adj, input_dim: int, hidden_dim: int = 128, device=None, struc_emb=None
    ):
        super(TemporalConvolutionEncoder, self).__init__()

        self._encoder = TGTCCellGRU(
            adj,
            input_dim=1,
            hidden_dim=hidden_dim,
        )

        self._hidden_dim = hidden_dim
        self._device = device

    def forward(self, x):
        batch_size, seq_len, num_nodes, num_feat = x.shape
        x_dynamic, x_static = x[:, :, :, 0].unsqueeze(-1), x[:, 0, :, 1:]
        hidden_state = torch.zeros(batch_size, num_nodes * self._hidden_dim).type_as(x)
        cell_state = torch.zeros(batch_size, num_nodes * self._hidden_dim).type_as(x)
        output = None

        # encode sequence with graph conv and generate gru sequence embedding
        for i in range(seq_len):
            input = x_dynamic[:, i, :, :]
            hidden_state, cell_state = self._encoder(input, hidden_state, cell_state)
            output = hidden_state.reshape((batch_size, num_nodes, self._hidden_dim))

        # Z -> BxNxE
        return output


class GatedResidualNetwork(nn.Module):
    def __init__(self, hidden_dim):
        super(GatedResidualNetwork, self).__init__()
        self.linear_1 = nn.Linear(hidden_dim, hidden_dim)
        self.linear_2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear_3 = nn.Linear(hidden_dim, hidden_dim)
        self.elu = nn.ELU()
        self.drop = nn.Dropout(p=0.3)

    def forward(self, x, static_context=None):
        batch_size, num_nodes, hidden_dim = x.shape
        x = x.reshape(batch_size * num_nodes, hidden_dim)
        skip = x
        x = self.linear_1(x)

        if static_context != None:
            static_context = (
                static_context.unsqueeze(0)
                .repeat(batch_size, 1, 1)
                .reshape(batch_size * num_nodes, hidden_dim)
            )
            x = x + self.linear_2(static_context)

        x = self.elu(x)
        x = self.linear_3(x)

        return x, skip


class GraphSeqAttention(nn.Module):
    def __init__(self, hidden_dim, node_num):
        super(GraphSeqAttention, self).__init__()
        self.linear_1 = nn.Linear(hidden_dim, 1)
        self.linear_2 = nn.Linear(node_num, 1)
        self.linear_3 = nn.Linear(hidden_dim, hidden_dim)
        self.soft = nn.Softmax(dim=-1)

    def forward(
        self,
        x,
    ):
        # x: (seq, batch, node_num, hdim)
        seq_len, batch_size, node_num, hdim = x.shape
        # remove hdim
        x = x.reshape(-1, hdim)
        x = self.linear_1(x)

        x = x.reshape(-1, node_num)
        f = self.linear_2(x)
        g = self.linear_2(x)

        f, g = f.reshape(-1, seq_len), g.reshape(-1, seq_len)
        s = f * g

        alpha = self.soft(s)
        context = alpha.unsqueeze(2) * x.reshape(-1, seq_len, node_num)
        context = context.permute((0, 2, 1))

        return context, alpha


class ConcatAttention(nn.Module):
    def __init__(self, hidden_dim):
        super(ConcatAttention, self).__init__()
        self.linear_1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.linear_2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear_3 = nn.Linear(hidden_dim, hidden_dim)
        self.soft = nn.Softmax(dim=-1)

    def forward(self, x, g):
        # x: (seq, batch, node_num, hdim)
        batch_size, node_num, hdim = x.shape
        a = self.linear_2(torch.tanh(self.linear_1(torch.concat([x, g], dim=-1))))
        print(a.shape)
        a = self.soft(a)
        print(a.shape)
        out = a * x
        # out = F.normalize(out, p=2, dim=-1)
        return out


class DotProductAtt(nn.Module):
    def __init__(self):
        super(DotProductAtt, self).__init__()
        self.soft = nn.Softmax(dim=-1)

    def forward(self, x, h):
        # x: bxd
        # h: bxsxd
        e = torch.bmm(h, x.unsqueeze(2))
        scores = self.soft(e)
        # bxsx1
        context = (h * scores).sum(1)

        return context


class TGTCConvolution(nn.Module):
    def __init__(
        self, adj, num_gru_units: int, output_dim: int, add_dims: int, bias: float = 0.0
    ):
        super(TGTCConvolution, self).__init__()
        self._num_gru_units = num_gru_units
        self._output_dim = output_dim
        self._bias_init_value = bias
        self.register_buffer("laplacian", torch.FloatTensor(adj))
        self.weights = nn.Parameter(
            torch.FloatTensor(self._num_gru_units + add_dims, self._output_dim)
        )
        self.biases = nn.Parameter(torch.FloatTensor(self._output_dim))
        # self.register_buffer("struc_emb", torch.FloatTensor(struc_emb))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weights)
        nn.init.constant_(self.biases, self._bias_init_value)

    def forward(self, inputs, hidden_state=None):
        if hidden_state == None and self._num_gru_units > 0:
            raise ValueError("hidden state must be given if gru units > 0.")

        batch_size, num_nodes, num_feats = inputs.shape
        # inputs (batch_size, num_nodes) -> (batch_size, num_nodes, 1)
        # inputs = inputs.reshape((batch_size, num_nodes, 1))
        # hidden_state (batch_size, num_nodes, num_gru_units)
        # [x, h] (batch_size, num_nodes, num_gru_units + 1)
        concatenation = inputs
        if hidden_state is not None:
            hidden_state = hidden_state.reshape(
                (batch_size, num_nodes, self._num_gru_units)
            )
            concatenation = torch.cat((inputs, hidden_state), dim=2)
        # [x, h] (num_nodes, num_gru_units + num_feats, batch_size)
        concatenation = concatenation.transpose(0, 1).transpose(1, 2)
        # [x, h] (num_nodes, (num_gru_units + num_feats) * batch_size)
        concatenation = concatenation.reshape(
            (num_nodes, (self._num_gru_units + num_feats) * batch_size)
        )
        # A[x, h] (num_nodes, (num_gru_units + num_feats) * batch_size)
        a_times_concat = self.laplacian @ concatenation
        # A[x, h] (num_nodes, num_gru_units + num_feats, batch_size)
        a_times_concat = a_times_concat.reshape(
            (num_nodes, self._num_gru_units + num_feats, batch_size)
        )
        # A[x, h] (batch_size, num_nodes, num_gru_units + num_feats)
        a_times_concat = a_times_concat.transpose(0, 2).transpose(1, 2)

        # A[x, h] (batch_size * num_nodes, num_gru_units + num_feats)
        a_times_concat = a_times_concat.reshape(
            (batch_size * num_nodes, self._num_gru_units + num_feats)
        )
        # A[x, h]W + b (batch_size * num_nodes, output_dim)
        outputs = a_times_concat @ self.weights + self.biases
        # A[x, h]W + b (batch_size, num_nodes, output_dim)
        outputs = outputs.reshape((batch_size, num_nodes, self._output_dim))
        # A[x, h]W + b (batch_size, num_nodes * output_dim)
        outputs = outputs.reshape((batch_size, num_nodes * self._output_dim))
        return outputs

    @property
    def hyperparameters(self):
        return {
            "num_gru_units": self._num_gru_units,
            "output_dim": self._output_dim,
            "bias_init_value": self._bias_init_value,
        }


class TGTCCellLSTM(nn.Module):
    def __init__(self, adj, input_dim: int, hidden_dim: int, use_attention=False):
        super(TGTCCellLSTM, self).__init__()
        self._input_dim = input_dim
        self._hidden_dim = hidden_dim
        add_dims = 1 if use_attention else 4
        self.graph_conv1 = TGTCConvolution(
            adj,
            self._hidden_dim,
            self._hidden_dim * 3,
            add_dims=self._input_dim,
            bias=1.0,
        )
        self.graph_conv2 = TGTCConvolution(
            adj, self._hidden_dim, self._hidden_dim, add_dims=add_dims
        )

    def forward(self, inputs, hidden_state, cell_state):
        concatenation = torch.sigmoid(self.graph_conv1(inputs, hidden_state))
        f, i, o = torch.chunk(concatenation, chunks=3, dim=1)

        cs = torch.tanh(self.graph_conv2(inputs, hidden_state))
        c = f * cell_state + i * cs

        new_hidden_state = o * torch.tanh(c)

        return new_hidden_state, cell_state

    @property
    def hyperparameters(self):
        return {"input_dim": self._input_dim, "hidden_dim": self._hidden_dim}


class CellAttLSTM(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super(CellAttLSTM, self).__init__()
        self._input_dim = input_dim
        self._hidden_dim = hidden_dim
        self.score_weight = nn.Parameter(
            torch.FloatTensor(self._hidden_dim, self._hidden_dim)
        )
        self.score_bias = nn.Parameter(torch.FloatTensor(self._hidden_dim))
        self.hidden_weight = nn.Parameter(
            torch.FloatTensor(self._hidden_dim, self._hidden_dim)
        )
        self.input_weight = nn.Parameter(
            torch.FloatTensor(self._hidden_dim, self._hidden_dim)
        )

    def forward(self, inputs, hidden_state, cell_state, embedding):
        print(inputs.shape, hidden_state.shape, cell_state.shape, embedding.shape)
        scores = self.score_weight * torch.tanh(
            self.hidden_weight * hidden_state
            + self.input_weight * inputs
            + self.score_bias
        )
        print(scores.shape)
        weights = torch.softmax(dim=-1)(scores)
        print(weights.shape)

        # concatenation = torch.sigmoid(self.graph_conv1(inputs, hidden_state))
        # f, i, o = torch.chunk(concatenation, chunks=3, dim=1)

        # conv = self.graph_conv2(inputs, hidden_state)
        # cs = torch.tanh(conv)
        # c = f * cell_state + i * cs

        # new_hidden_state = o * torch.tanh(c)

        # return new_hidden_state, cell_state, conv

    @property
    def hyperparameters(self):
        return {"input_dim": self._input_dim, "hidden_dim": self._hidden_dim}


class TGTCCellGRU(nn.Module):
    def __init__(self, adj, input_dim: int, hidden_dim: int):
        super(TGTCCellGRU, self).__init__()
        self._input_dim = input_dim
        self._hidden_dim = hidden_dim
        self.graph_conv1 = TGTCConvolution(
            adj, self._hidden_dim, self._hidden_dim * 2, bias=1.0, add_dims=1
        )
        self.graph_conv2 = TGTCConvolution(
            adj, self._hidden_dim, self._hidden_dim, add_dims=1
        )

    def forward(self, inputs, hidden_state, cell_state):
        # [r, u] = sigmoid(A[x, h]W + b)
        # [r, u] (batch_size, num_nodes * (2 * num_gru_units))
        concatenation = torch.sigmoid(self.graph_conv1(inputs, hidden_state))
        # r (batch_size, num_nodes, num_gru_units)
        # u (batch_size, num_nodes, num_gru_units)
        r, u = torch.chunk(concatenation, chunks=2, dim=1)
        # c = tanh(A[x, (r * h)W + b])
        # c (batch_size, num_nodes * num_gru_units)
        c = torch.tanh(self.graph_conv2(inputs, r * hidden_state))
        # h := u * h + (1 - u) * c
        # h (batch_size, num_nodes * num_gru_units)
        new_hidden_state = u * hidden_state + (1.0 - u) * c

        return new_hidden_state, cell_state

    @property
    def hyperparameters(self):
        return {"input_dim": self._input_dim, "hidden_dim": self._hidden_dim}


class TCell(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super(TCell, self).__init__()
        self._input_dim = input_dim
        self._hidden_dim = hidden_dim

    def forward(self, inputs, hidden_state):
        # [r, u] = sigmoid(A[x, h]W + b)
        # [r, u] (batch_size, num_nodes * (2 * num_gru_units))
        concatenation = torch.sigmoid(torch.concat((inputs, hidden_state), dim=2))
        # r (batch_size, num_nodes, num_gru_units)
        # u (batch_size, num_nodes, num_gru_units)
        r, u = torch.chunk(concatenation, chunks=2, dim=1)
        # c = tanh(A[x, (r * h)W + b])
        # c (batch_size, num_nodes * num_gru_units)
        c = torch.tanh(torch.concat((inputs, r * hidden_state), dim=2))
        # h := u * h + (1 - u) * c
        # h (batch_size, num_nodes * num_gru_units)
        new_hidden_state = u * hidden_state + (1.0 - u) * c
        return new_hidden_state, new_hidden_state

    @property
    def hyperparameters(self):
        return {"input_dim": self._input_dim, "hidden_dim": self._hidden_dim}
