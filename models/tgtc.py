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
        edge_index=None,
        struc_emb=None,
        batch_size: int = 128,
        hidden_dim: int = 128,
        cell_type="lstm",
        use_attention=False,
        device_ids=[],
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
            cell_type=cell_type,
            use_attention=use_attention,
        )
        if len(device_ids) > 0:
            self.model = nn.DataParallel(self.model, device_ids=device_ids)
        self.model.to(device)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=0.001, weight_decay=1e-5
        )
        self.loss_seq = nn.MSELoss(reduction="mean")
        self.device = device
        # data needs to be shape TxNxF T temporal, N nodes, F features
        train, _ = generate_torch_datasets(
            data=data, seq_len=12, pre_len=3, reconstruct=True, split_ratio=1
        )
        self.train_loader = DataLoader(
            train,
            batch_size=self.batch_size,
            shuffle=True,
        )
        # self.eval_loader = DataLoader(
        #     test,
        #     batch_size=self.batch_size,
        # )

        self.data = data
        self._adj = adj
        self._hidden_dim = hidden_dim
        self._edge_index = edge_index
        # self._struc_emb = struc_emb

        # norm data
        for i in range(self.data.shape[-1]):
            max_val = self.data[:, :, i].max()
            self.data[:, :, i] = self.data[:, :, i] / max_val

    def train(self, epochs):
        self.model.train()
        for e in range(epochs):
            total_loss = 0
            seq_loss = 0
            reconst_loss = 0
            for X, y in tqdm(self.train_loader, leave=False):
                X = X.to(self.device)
                z, seq_recon = self.model(X)
                z = (z.sum(axis=0) / self.batch_size).squeeze()

                y = y.to(self.device)

                loss_1 = self.loss_seq(seq_recon, y) * 100
                loss_2 = recon_loss(z, self._edge_index)
                loss = loss_1 + loss_2

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
                seq_loss += loss_1.item()
                reconst_loss += loss_2.item()

            print(
                f"Average training loss in episode {e} -- Total: {total_loss/len(self.train_loader)}, Seq: {seq_loss/len(self.train_loader)}, Recon: {reconst_loss/len(self.train_loader)}"
            )

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path, map_location=self.device))

    def load_emb(self):
        self.model.eval()
        z, _ = self.model(torch.Tensor(self.data[100:112]).unsqueeze(0).to(self.device))
        return z.squeeze().detach().cpu().numpy()


class TemporalGraphModel(nn.Module):
    def __init__(
        self,
        adj,
        num_nodes,
        input_dim: int,
        device=None,
        struc_emb=None,
        hidden_dim: int = 128,
        cell_type="lstm",
        use_attention=False,
    ):
        super(TemporalGraphModel, self).__init__()
        self.register_buffer("adj", torch.FloatTensor(adj))
        self.encoder = TemporalGraphEncoder(
            input_dim=input_dim,
            adj=self.adj,
            device=device,
            hidden_dim=hidden_dim,
            num_nodes=num_nodes,
            struc_emb=struc_emb,
            cell_type=cell_type,
            use_attention=use_attention,
        )
        self.decoder = TemporalGraphDecoder(
            input_dim=hidden_dim, hidden_dim=hidden_dim, num_nodes=num_nodes
        )

    def forward(self, X):
        z = self.encoder(X)
        recon = self.decoder(z, X.shape[1])

        return z, recon


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
        num_nodes: int,
        hidden_dim: int = 128,
        lstm_layers=2,
        device=None,
        struc_emb=None,
        cell_type="lstm",
        use_attention=False,
    ):
        super(TemporalGraphEncoder, self).__init__()
        # self._gtc_conv = TGTCConvolution(adj, hidden_dim, hidden_dim, bias=1.0)
        # self._lstm_cell = nn.GRUCell(
        #     adj.shape[0] * hidden_dim,
        #     adj.shape[0] * hidden_dim,
        # )
        cell = TGTCCellLSTM if cell_type == "lstm" else TGTCCellGRU

        self._encoder = cell(
            adj, input_dim=input_dim, hidden_dim=hidden_dim, struc_emb=struc_emb
        )

        if use_attention:
            # self._attention = ConcatAttention(hidden_dim=hidden_dim)
            self._self_attention = nn.MultiheadAttention(
                hidden_dim * 2, 4, batch_first=True
            )
            self.posff = nn.Sequential(
                nn.Linear(hidden_dim * 2, 512),
                nn.ReLU(),
                nn.Linear(512, hidden_dim),
            )
            # self._lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers=1, batch_first=True)
            # self._transformer = nn.TransformerEncoder(
            #     encoder_layer=nn.TransformerEncoderLayer(d_model=128, nhead=4),
            #     num_layers=1,
            # )

            # self._self_attention = GraphSeqAttention(hidden_dim, num_nodes)
            # self.att_transform = nn.Linear(12, hidden_dim)  # seq_len -> hidden_dim

            # self.transform = nn.Linear(hidden_dim, hidden_dim)
            # self.dropout = nn.Dropout(p=0.3)
            # self.glu = nn.GLU()
            # self.norm = nn.LayerNorm(hidden_dim)
            # self.resgate_1 = GatedResidualNetwork(hidden_dim=hidden_dim)
            # self.resgate_2 = GatedResidualNetwork(hidden_dim=hidden_dim)
            # self.gate_linear_1 = nn.Linear(hidden_dim, hidden_dim)
            # self.gate_linear_2 = nn.Linear(hidden_dim, hidden_dim)

        self._hidden_dim = hidden_dim
        self._cell_type = cell_type
        self._struc_emb = struc_emb
        self._use_attention = use_attention
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
        hidden_state = (
            self._struc_emb.unsqueeze(0)
            .repeat(batch_size, 1, 1)
            .reshape(batch_size, num_nodes * self._hidden_dim)
        )  # torch.zeros(batch_size, num_nodes * self._hidden_dim).type_as(x)
        cell_state = (
            self._struc_emb.unsqueeze(0)
            .repeat(batch_size, 1, 1)
            .reshape(batch_size, num_nodes * self._hidden_dim)
        )
        output = None
        hidden_sequence = torch.zeros(
            size=(seq_len, batch_size, num_nodes, self._hidden_dim * 2)
        )
        # encode sequence with graph conv and generate lstm sequence embedding
        for i in range(seq_len):
            hidden_state, cell_state, conv = self._encoder(
                x[:, i, :, :], hidden_state, cell_state
            )
            output = hidden_state.reshape((batch_size, num_nodes, self._hidden_dim))
            conv = conv.reshape((batch_size, num_nodes, self._hidden_dim))
            if self._use_attention:
                # gated = self.apply_gate(output.reshape(-1, self._hidden_dim)).reshape(
                #     (batch_size, num_nodes, self._hidden_dim)
                # )
                # out, skip = self.resgate_1(gated, self._struc_emb)
                # out = self.apply_gate(out, skip)
                # hidden_sequence[i, :, :, :] = out.reshape(
                #     ((batch_size, num_nodes, self._hidden_dim))
                # )
                hidden_sequence[i, :, :, :] = torch.concat(
                    [
                        output,
                        hidden_state.reshape(batch_size, num_nodes, self._hidden_dim),
                    ],
                    dim=-1,
                )

        if self._use_attention:
            # propagte embedding into multihead attention to generate context vector
            hidden_sequence = hidden_sequence.reshape(
                seq_len, batch_size * num_nodes, self._hidden_dim * 2
            )
            hidden_sequence = hidden_sequence.permute((1, 0, 2))
            hidden_sequence = hidden_sequence.to(self._device)
            # context = self._attention(
            #     output, self._struc_emb.unsqueeze(0).repeat(batch_size, 1, 1)
            # )
            context, weights = self._self_attention(
                hidden_sequence,
                hidden_sequence,
                hidden_sequence,
                need_weights=True,
                average_attn_weights=True,
            )
            context = context.sum(1).squeeze()
            out = self.posff(context)
            # out, _ = self._lstm(context)

            # context = context.mean(1).squeeze()

            # context = context.reshape(-1, seq_len)
            # context = self.att_transform(context)
            # context = context.reshape(batch_size, num_nodes, self._hidden_dim)

            # context = self._self_attention(
            #     output, self._struc_emb.unsqueeze(0).repeat(batch_size, 1, 1)
            # )

            # context = context[:, -1, :].squeeze()
            # context = self.norm(context)
            # context = self.transform(context)
            # context = self.apply_gate(context, output.reshape(-1, self._hidden_dim))  #
            # context = context.reshape((batch_size, num_nodes, self._hidden_dim))
            # context = context.reshape((batch_size, num_nodes, self._hidden_dim))
            # context, skip = self.resgate_2(context)
            # context = self.apply_gate(context, skip)
            # context = self.apply_gate(
            #     context,
            #     gated.reshape((batch_size * num_nodes, self._hidden_dim)),
            #     dropout=False,
            # )
            out = out.reshape((batch_size, num_nodes, self._hidden_dim))

            return out

        # print(context.shape)
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


class TGTCConvolution(nn.Module):
    def __init__(self, adj, num_gru_units: int, output_dim: int, bias: float = 0.0):
        super(TGTCConvolution, self).__init__()
        self._num_gru_units = num_gru_units
        self._output_dim = output_dim
        self._bias_init_value = bias
        self.register_buffer("laplacian", torch.FloatTensor(adj))
        self.weights = nn.Parameter(
            torch.FloatTensor(self._num_gru_units + 4, self._output_dim)
        )
        self.biases = nn.Parameter(torch.FloatTensor(self._output_dim))
        # self.register_buffer("struc_emb", torch.FloatTensor(struc_emb))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weights)
        nn.init.constant_(self.biases, self._bias_init_value)

    def forward(self, inputs, hidden_state):
        batch_size, num_nodes, num_feats = inputs.shape
        # inputs (batch_size, num_nodes) -> (batch_size, num_nodes, 1)
        # inputs = inputs.reshape((batch_size, num_nodes, 1))
        # hidden_state (batch_size, num_nodes, num_gru_units)
        hidden_state = hidden_state.reshape(
            (batch_size, num_nodes, self._num_gru_units)
        )
        # [x, h] (batch_size, num_nodes, num_gru_units + 1)
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
    def __init__(self, adj, input_dim: int, hidden_dim: int, struc_emb=None):
        super(TGTCCellLSTM, self).__init__()
        self._input_dim = input_dim
        self._hidden_dim = hidden_dim
        self.graph_conv1 = TGTCConvolution(
            adj, self._hidden_dim, self._hidden_dim * 3, bias=1.0
        )
        self.graph_conv2 = TGTCConvolution(adj, self._hidden_dim, self._hidden_dim)

    def forward(self, inputs, hidden_state, cell_state):
        # [r, u] = sigmoid(A[x, h]W + b)
        # [r, u] (batch_size, num_nodes * (2 * num_gru_units))
        concatenation = torch.sigmoid(self.graph_conv1(inputs, hidden_state))
        # r (batch_size, num_nodes, num_gru_units)
        # u (batch_size, num_nodes, num_gru_units)
        f, i, o = torch.chunk(concatenation, chunks=3, dim=1)
        # c = tanh(A[x, (r * h)W + b])
        # c (batch_size, num_nodes * num_gru_units)
        conv = self.graph_conv2(inputs, hidden_state)
        cs = torch.tanh(conv)
        c = f * cell_state + i * cs
        # h := u * h + (1 - u) * c
        # h (batch_size, num_nodes * num_gru_units)
        new_hidden_state = o * torch.tanh(c)

        return new_hidden_state, cell_state, conv

    @property
    def hyperparameters(self):
        return {"input_dim": self._input_dim, "hidden_dim": self._hidden_dim}


class TGTCCellGRU(nn.Module):
    def __init__(self, adj, input_dim: int, hidden_dim: int, struc_emb=None):
        super(TGTCCellGRU, self).__init__()
        self._input_dim = input_dim
        self._hidden_dim = hidden_dim
        self.graph_conv1 = TGTCConvolution(
            adj, self._hidden_dim, self._hidden_dim * 2, bias=1.0
        )
        self.graph_conv2 = TGTCConvolution(adj, self._hidden_dim, self._hidden_dim)

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
