import torch.nn as nn
import torch

from . import Model, GTCSubConv
from .utils import generate_torch_datasets, transform_data
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
        edge_index,
        struc_emb=None,
        batch_size: int = 128,
        hidden_dim: int = 128,
    ):
        super().__init__()
        self.batch_size = batch_size
        if struc_emb is not None:
            ...
        self.model = TemporalGraphModel(
            input_dim=data.shape[-1],
            adj=adj,
            hidden_dim=hidden_dim,
            num_nodes=data.shape[1],
        )
        self.model = nn.DataParallel(self.model, device_ids=[0, 1, 2])
        self.model.to(device)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=0.001, weight_decay=1e-5
        )
        self.loss_seq = nn.MSELoss()
        self.device = device
        # data needs to be shape TxNxF T temporal, N nodes, F features
        train, test = generate_torch_datasets(
            data=data, seq_len=12, pre_len=3, reconstruct=True
        )
        self.train_loader = DataLoader(
            train,
            batch_size=self.batch_size,
            shuffle=True,
        )
        self.eval_loader = DataLoader(
            test,
            batch_size=self.batch_size,
        )

        self.data = data
        self._adj = adj
        self._hidden_dim = hidden_dim
        self._edge_index = edge_index
        self._struc_emb = struc_emb

    def train(self, epochs):
        self.model.train()
        for e in range(epochs):
            total_loss = 0
            seq_loss = 0
            reconst_loss = 0
            for X, y in tqdm(self.train_loader, leave=False):
                X = X.to(self.device)
                if self._struc_emb is not None:
                    struc = torch.Tensor(self._struc_emb)
                    X = torch.concat((X, struc.unsqueeze(0)), axis=-1)
                z, seq_recon = self.model(X)
                z = (z.sum(axis=0) / self.batch_size).squeeze()

                y = y.to(self.device)

                loss_1 = self.loss_seq(seq_recon, y)
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
        ...


class TemporalGraphModel(nn.Module):
    def __init__(self, adj, num_nodes, input_dim: int, hidden_dim: int = 128):
        super(TemporalGraphModel, self).__init__()
        self.register_buffer("adj", torch.FloatTensor(adj))
        self.encoder = TemporalGraphEncoder(
            input_dim=input_dim, adj=self.adj, hidden_dim=hidden_dim
        )
        self.decoder = TemporalGraphDecoder(
            input_dim=hidden_dim, hidden_dim=hidden_dim, num_nodes=num_nodes
        )

    def forward(self, X):
        self.train()
        z = self.encoder(X)
        recon = self.decoder(z, X.shape[1])

        return z, recon


class TemporalGraphDecoder(nn.Module):
    def __init__(self, input_dim: int, num_nodes, hidden_dim: int = 128):
        super(TemporalGraphDecoder, self).__init__()
        self._tdecoder = nn.GRU(input_dim, hidden_dim, num_layers=1, batch_first=True)
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
    def __init__(self, adj, input_dim: int, hidden_dim: int = 128, lstm_layers=2):
        super(TemporalGraphEncoder, self).__init__()
        # enc_conv = GTCSubConv(in_dim=input_dim, out_dim=hidden_dim, norm=False)
        # enc_lstm = nn.LSTM(
        #     hidden_dim,
        #     hidden_dim,
        #     num_layers=lstm_layers,
        #     batch_first=True,
        #     dropout=0.5,
        # )
        self._encoder = TGTCCell(adj, input_dim=input_dim, hidden_dim=hidden_dim)

        self._hidden_dim = hidden_dim

    def forward(self, x):
        batch_size, seq_len, num_nodes, num_feat = x.shape
        hidden_state = torch.zeros(batch_size, num_nodes * self._hidden_dim).type_as(x)
        output = None
        # encode sequence with graph conv
        for i in range(seq_len):
            output, hidden_state = self._encoder(x[:, i, :, :], hidden_state)
            output = output.reshape((batch_size, num_nodes, self._hidden_dim))

        # Z -> BxNxE
        return output


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


class TGTCCell(nn.Module):
    def __init__(self, adj, input_dim: int, hidden_dim: int):
        super(TGTCCell, self).__init__()
        self._input_dim = input_dim
        self._hidden_dim = hidden_dim
        self.graph_conv1 = TGTCConvolution(
            adj, self._hidden_dim, self._hidden_dim * 2, bias=1.0
        )
        self.graph_conv2 = TGTCConvolution(adj, self._hidden_dim, self._hidden_dim)

    def forward(self, inputs, hidden_state):
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
        return new_hidden_state, new_hidden_state

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
        print(inputs.shape, hidden_state.shape)
        concatenation = torch.sigmoid(torch.concat((inputs, hidden_state), dim=2))
        print(concatenation.shape)
        # r (batch_size, num_nodes, num_gru_units)
        # u (batch_size, num_nodes, num_gru_units)
        r, u = torch.chunk(concatenation, chunks=2, dim=1)
        # c = tanh(A[x, (r * h)W + b])
        # c (batch_size, num_nodes * num_gru_units)
        print(r.shape, inputs.shape, hidden_state.shape)
        c = torch.tanh(torch.concat((inputs, r * hidden_state), dim=2))
        print(c.shape)
        # h := u * h + (1 - u) * c
        # h (batch_size, num_nodes * num_gru_units)
        new_hidden_state = u * hidden_state + (1.0 - u) * c
        return new_hidden_state, new_hidden_state

    @property
    def hyperparameters(self):
        return {"input_dim": self._input_dim, "hidden_dim": self._hidden_dim}
