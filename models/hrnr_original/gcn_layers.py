# coding=utf-8

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter


class SpecialSpmmFunction(torch.autograd.Function):
    """Special function for only sparse region backpropataion layer."""

    @staticmethod
    def forward(ctx, indices, values, shape, b):
        assert indices.requires_grad == False
        a = torch.sparse_coo_tensor(indices, values, shape)
        ctx.save_for_backward(a, b)
        ctx.N = shape[0]
        return torch.matmul(a, b)

    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        grad_values = grad_b = None
        if ctx.needs_input_grad[1]:
            grad_a_dense = grad_output.matmul(b.t())
            edge_idx = a._indices()[0, :] * ctx.N + a._indices()[1, :]
            grad_values = grad_a_dense.view(-1)[edge_idx]
        if ctx.needs_input_grad[3]:
            grad_b = a.t().matmul(grad_output)
        return None, grad_values, None, grad_b


class SpecialSpmm(nn.Module):
    def forward(
        self, indices, values, shape, b
    ):  # indices, value and shape define a sparse tensor, it will do mm() operation with b
        return SpecialSpmmFunction.apply(indices, values, shape, b)


class SPGCN(Module):
    def __init__(self, in_features, out_features, device, concat=True):
        super(SPGCN, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.concat = concat
        self.device = device

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_normal_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.zeros(size=(1, 2 * out_features)))
        nn.init.xavier_normal_(self.a.data, gain=1.414)
        #    self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.special_spmm = SpecialSpmm()
        pass

    def forward(self, inputs, adj):
        inputs = inputs.squeeze()
        #   adj = adj.squeeze()
        dv = self.device
        #    self_loop = torch.eye(adj.shape[0]).to(self.device)
        #    adj = adj + self_loop
        N = inputs.size()[0]
        #    edge = adj.nonzero().t()
        edge = adj._indices()
        #    edge = edge[1:, :]
        h = torch.mm(inputs, self.W)
        # h: N x out
        assert not torch.isnan(h).any()
        #    edge_h = torch.cat((h[edge[0, :], :], h[edge[1, :], :]), dim=1).t()
        # edge: 2*D x E
        #    edge_e = torch.exp(-self.leakyrelu(self.a.mm(edge_h).squeeze()))
        edge_e = torch.ones(edge.shape[1], dtype=torch.float).to(self.device)
        assert not torch.isnan(edge_e).any()
        # edge_e: E
        e_rowsum = self.special_spmm(
            edge, edge_e, torch.Size([N, N]), torch.ones(size=(N, 1), device=dv)
        )
        # e_rowsum: N x 1
        #    edge_e = self.dropout(edge_e)
        # edge_e: E

        h_prime = self.special_spmm(edge, edge_e, torch.Size([N, N]), h)
        assert not torch.isnan(h_prime).any()
        # h_prime: N x out
        h_prime = h_prime.div(e_rowsum)
        # h_prime: N x out
        assert not torch.isnan(h_prime).any()
        if self.concat:  # if this layer is not last layer,
            return F.elu(h_prime)
        else:  # if this layer is last layer
            return h_prime


class SPGAT(Module):
    def __init__(self, in_features, out_features, dropout, alpha, device, concat=True):
        super(SPGAT, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat
        self.device = device

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_normal_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.zeros(size=(1, 2 * out_features)))
        nn.init.xavier_normal_(self.a.data, gain=1.414)
        self.dropout = nn.Dropout(dropout)
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.special_spmm = SpecialSpmm()
        pass

    def forward(self, inputs, adj):
        inputs = inputs.squeeze()
        #   adj = adj.squeeze()
        dv = self.device
        #    self_loop = torch.eye(adj.shape[0]).to(self.device)
        #    adj = adj + self_loop
        N = inputs.size()[0]
        #    edge = adj.nonzero().t()
        edge = adj._indices()
        #    print(edge.shape)
        #    edge = edge[1:, :]
        # print(inputs.isnan().sum())
        h = torch.mm(inputs, self.W)
        # h: N x out
        assert not torch.isnan(h).any()
        edge_h = torch.cat((h[edge[0, :], :], h[edge[1, :], :]), dim=1).t()
        # edge: 2*D x E
        edge_e = torch.exp(-self.leakyrelu(self.a.mm(edge_h).squeeze()))
        assert not torch.isnan(edge_e).any()
        # edge_e: E
        e_rowsum = self.special_spmm(
            edge, edge_e, torch.Size([N, N]), torch.ones(size=(N, 1), device=dv)
        )
        # e_rowsum: N x 1
        edge_e = self.dropout(edge_e)
        # edge_e: E
        h_prime = self.special_spmm(edge, edge_e, torch.Size([N, N]), h)
        assert not torch.isnan(h_prime).any()
        # h_prime: N x out
        h_prime = h_prime.div(e_rowsum)
        # h_prime: N x out
        assert not torch.isnan(h_prime).any()
        if self.concat:  # if this layer is not last layer,
            return F.elu(h_prime)
        else:  # if this layer is last layer
            return h_prime


class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, device, bias=True):
        super(GraphConvolution, self).__init__()
        self.device = device
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 0.5
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def norm(self, adj):
        node_num = adj.shape[-1]
        # add remaining self-loops
        self_loop = torch.eye(node_num).to(self.device)
        self_loop = self_loop.reshape((1, node_num, node_num))
        self_loop = self_loop.repeat(adj.shape[0], 1, 1)
        adj_post = adj + self_loop
        # signed adjacent matrix
        deg_abs = torch.sum(torch.abs(adj_post), dim=-1)
        deg_abs_sqrt = deg_abs.pow(-0.5)
        diag_deg = torch.diag_embed(deg_abs_sqrt, dim1=-2, dim2=-1)

        norm_adj = torch.matmul(torch.matmul(diag_deg, adj_post), diag_deg)
        return norm_adj

    def forward(self, inputs, adj):
        support = torch.matmul(inputs, self.weight)
        adj_norm = self.norm(adj)
        output = torch.matmul(support.transpose(1, 2), adj_norm.transpose(1, 2))
        output = output.transpose(1, 2)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return (
            self.__class__.__name__
            + " ("
            + str(self.in_features)
            + " -> "
            + str(self.out_features)
            + ")"
        )
