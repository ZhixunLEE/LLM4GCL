import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import degree


class SGConv(MessagePassing):
    def __init__(self, k=1, cached=False, norm=None, allow_zero_in_degree=False):
        super(SGConv, self).__init__(aggr='add')
        self._cached = cached
        self._cached_h = None
        self._k = k
        self.norm = norm
        self._allow_zero_in_degree = allow_zero_in_degree

    def set_allow_zero_in_degree(self, set_value):
        self._allow_zero_in_degree = set_value

    def forward(self, x, edge_index):
        if self._cached_h is not None:
            return self._cached_h

        # edge_index, _ = self.add_self_loops(edge_index, num_nodes=x.size(0))

        row, col = edge_index
        deg = degree(row, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        h = x
        for _ in range(self._k):
            h = self.propagate(edge_index, x=h, norm=norm)

        if self.norm is not None:
            h = self.norm(h)

        if self._cached:
            self._cached_h = h

        return h

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def __repr__(self):
        return '{}(k={}, cached={}, norm={})'.format(
            self.__class__.__name__, self._k, self._cached, self.norm)


class SGCNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, dropout, k=1, cached=False):
        super(SGCNet, self).__init__()
        self.sgc_agg = SGConv(k=k)
        self.convs = torch.nn.ModuleList()
        self.convs.append(torch.nn.Linear(input_dim, hidden_dim))
        self.bns = torch.nn.ModuleList()
        self.bns.append(torch.nn.BatchNorm1d(hidden_dim))
        for _ in range(num_layers - 2):
            self.convs.append(torch.nn.Linear(hidden_dim, hidden_dim))
            self.bns.append(torch.nn.BatchNorm1d(hidden_dim))
        self.convs.append(torch.nn.Linear(hidden_dim, output_dim))
        self.dropout = dropout

    def forward(self, x, edge_index):        
        for i, conv in enumerate(self.convs[:-1]):
            x = self.sgc_agg(x, edge_index)
            x = conv(x)
            x = self.bns[i](x)
            # x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x)

        return x

    def reset_params(self):
        for layer in self.convs:
            layer.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()