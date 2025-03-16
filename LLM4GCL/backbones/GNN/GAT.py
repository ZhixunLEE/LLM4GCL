import torch
import torch.nn.functional as F

from torch_geometric.nn import GATConv


class GATNet(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout, num_heads=4):
        super(GATNet, self).__init__()
        self.convs = torch.nn.ModuleList()
        self.convs.append(GATConv(in_channels, hidden_channels, heads=num_heads, concat=False))
        self.bns = torch.nn.ModuleList()
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(GATConv(hidden_channels, hidden_channels, heads=num_heads, concat=False))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        self.convs.append(GATConv(hidden_channels, out_channels, heads=num_heads, concat=False))
        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index=edge_index)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index=edge_index)
        return x