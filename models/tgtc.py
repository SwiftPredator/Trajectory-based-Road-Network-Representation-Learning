import torch.nn as nn
import torch

from . import Model, GTCSubConv
from .utils import generate_torch_datasets, transform_data
from torch.utils.data import DataLoader, Dataset
import torch_geometric.transforms as T
from torch_geometric.data import Data
from torch_geometric.nn import InnerProductDecoder


class TemporalGraphModel(Model):
    """
    Main idea behind this is to add another axis into the embedding, namely an axis for time.
    The goal is to model the change over time in the road network and therefore improve performance on several
    its applications like forecasting of traffic, trajectory task and so on.

    :Note: Add time features to the feature matrix like hour day of week and probably month to basically stamp the generated
    embeddings from the gtc model. Maybe it would be a good idea to emebed the time features seperately and concat them to the gtc_embed

    Args:
        Model (_type_): _description_
    """

    def __init__(self, data, device, batch_size: int = 128):
        super().__init__()
        self.batch_size = batch_size
        self.model = TemporalGraphEncoder()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.model = self.model.to(device)
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

        def train(self, epochs):
            self.train()
            for e in range(epochs):
                total_loss = 0
                for X, y in tqdm(loader, leave=False):
                    emb_batch = emb_batch.to(self.device)
                    if self.plugin is not None:
                        self.plugin.register_id_seq(X, mask, map, lengths)

                    y = y.to(self.device)
                    yh = self.forward(emb_batch, lengths, neigh_masks)

                    loss = self.loss(yh.squeeze(), y)
                    self.opt.zero_grad()
                    loss.backward()
                    self.opt.step()
                    total_loss += loss.item()

                print(f"Average training loss in episode {e}: {total_loss/len(loader)}")

        def recon_loss(self, z, pos_edge_index, neg_edge_index=None):
            decoder = InnerProductDecoder()
            EPS = 1e-15

            pos_loss = -torch.log(decoder(z, pos_edge_index, sigmoid=True) + EPS).mean()

            # Do not include self-loops in negative samples
            pos_edge_index, _ = remove_self_loops(pos_edge_index)
            pos_edge_index, _ = add_self_loops(pos_edge_index)
            if neg_edge_index is None:
                neg_edge_index = negative_sampling(pos_edge_index, z.size(0))
            neg_loss = -torch.log(
                1 - decoder(z, neg_edge_index, sigmoid=True) + EPS
            ).mean()

            return pos_loss + neg_loss


class TemporalGraphDecoder(nn.Module):
    def __init__(self, adj, input_dim: int, hidden_dim: int = 128, lstm_layers=2):
        super().__init__()
        enc_conv = GTCSubConv(in_dim=input_dim, out_dim=hidden_dim, norm=False)
        enc_lstm = nn.LSTM(
            hidden_dim,
            hidden_dim,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=0.5,
        )
        self._tdecoder = ...

        self._hidden_dim = hidden_dim

    def forward(self, z, seq_len):
        # stack z to match seq leng
        # BxNxF
        ...


class TemporalGraphEncoder(nn.Module):
    def __init__(self, adj, input_dim: int, hidden_dim: int = 128, lstm_layers=2):
        super().__init__()
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
        self.register_buffer(
            "laplacian", calculate_laplacian_with_self_loop(torch.FloatTensor(adj))
        )
        self.weights = nn.Parameter(
            torch.FloatTensor(self._num_gru_units + 1, self._output_dim)
        )
        self.biases = nn.Parameter(torch.FloatTensor(self._output_dim))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weights)
        nn.init.constant_(self.biases, self._bias_init_value)

    def forward(self, inputs, hidden_state):
        batch_size, num_nodes = inputs.shape
        # inputs (batch_size, num_nodes) -> (batch_size, num_nodes, 1)
        inputs = inputs.reshape((batch_size, num_nodes, 1))
        # hidden_state (batch_size, num_nodes, num_gru_units)
        hidden_state = hidden_state.reshape(
            (batch_size, num_nodes, self._num_gru_units)
        )
        # [x, h] (batch_size, num_nodes, num_gru_units + 1)
        concatenation = torch.cat((inputs, hidden_state), dim=2)
        # [x, h] (num_nodes, num_gru_units + 1, batch_size)
        concatenation = concatenation.transpose(0, 1).transpose(1, 2)
        # [x, h] (num_nodes, (num_gru_units + 1) * batch_size)
        concatenation = concatenation.reshape(
            (num_nodes, (self._num_gru_units + 1) * batch_size)
        )
        # A[x, h] (num_nodes, (num_gru_units + 1) * batch_size)
        a_times_concat = self.laplacian @ concatenation
        # A[x, h] (num_nodes, num_gru_units + 1, batch_size)
        a_times_concat = a_times_concat.reshape(
            (num_nodes, self._num_gru_units + 1, batch_size)
        )
        # A[x, h] (batch_size, num_nodes, num_gru_units + 1)
        a_times_concat = a_times_concat.transpose(0, 2).transpose(1, 2)
        # A[x, h] (batch_size * num_nodes, num_gru_units + 1)
        a_times_concat = a_times_concat.reshape(
            (batch_size * num_nodes, self._num_gru_units + 1)
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
        self.register_buffer("adj", torch.FloatTensor(adj))
        self.graph_conv1 = TGTCConvolution(
            self.adj, self._hidden_dim, self._hidden_dim * 2, bias=1.0
        )
        self.graph_conv2 = TGTCConvolution(self.adj, self._hidden_dim, self._hidden_dim)

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
