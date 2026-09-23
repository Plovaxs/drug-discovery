"""Heterogeneous Interaction Layer, ported near-verbatim from GIGN
(github.com/guaguabujianle/GIGN, GIGN/HIL.py). Per
guidance/reference_code/IMPLEMENTATION_NOTES.md: both edge types
(covalent/intra, non-covalent/inter) go through the SAME MessagePassing
instance with aggr='add' -- the cov/ncov differentiation happens via
separate MLPs applied after aggregation, not via a different aggregation
function. This is a straight port; only the RBF distance range was kept
identical to the source (0-6A, 9 bins).
"""
import torch
from torch import Tensor
from torch_geometric.nn.conv import MessagePassing
import torch.nn as nn


class HIL(MessagePassing):
    def __init__(self, in_channels: int, out_channels: int, **kwargs):
        kwargs.setdefault('aggr', 'add')
        super().__init__(**kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.mlp_node_cov = nn.Sequential(
            nn.Linear(self.in_channels, self.out_channels),
            nn.Dropout(0.1),
            nn.LeakyReLU(),
            nn.BatchNorm1d(self.out_channels))
        self.mlp_node_ncov = nn.Sequential(
            nn.Linear(self.in_channels, self.out_channels),
            nn.Dropout(0.1),
            nn.LeakyReLU(),
            nn.BatchNorm1d(self.out_channels))

        self.mlp_coord_cov = nn.Sequential(nn.Linear(9, self.in_channels), nn.SiLU())
        self.mlp_coord_ncov = nn.Sequential(nn.Linear(9, self.in_channels), nn.SiLU())

    def forward(self, x, edge_index_intra, edge_index_inter, pos=None, size=None):
        row_cov, col_cov = edge_index_intra
        coord_diff_cov = pos[row_cov] - pos[col_cov]
        radial_cov = self.mlp_coord_cov(
            _rbf(torch.norm(coord_diff_cov, dim=-1), D_min=0., D_max=6., D_count=9, device=x.device))
        out_node_intra = self.propagate(edge_index=edge_index_intra, x=x, radial=radial_cov, size=size)

        row_ncov, col_ncov = edge_index_inter
        coord_diff_ncov = pos[row_ncov] - pos[col_ncov]
        radial_ncov = self.mlp_coord_ncov(
            _rbf(torch.norm(coord_diff_ncov, dim=-1), D_min=0., D_max=6., D_count=9, device=x.device))
        out_node_inter = self.propagate(edge_index=edge_index_inter, x=x, radial=radial_ncov, size=size)

        out_node = self.mlp_node_cov(x + out_node_intra) + self.mlp_node_ncov(x + out_node_inter)
        return out_node

    def message(self, x_j: Tensor, x_i: Tensor, radial, index: Tensor):
        return x_j * radial


def _rbf(D, D_min=0., D_max=20., D_count=16, device='cpu'):
    D_mu = torch.linspace(D_min, D_max, D_count).to(device)
    D_mu = D_mu.view([1, -1])
    D_sigma = (D_max - D_min) / D_count
    D_expand = torch.unsqueeze(D, -1)
    RBF = torch.exp(-((D_expand - D_mu) / D_sigma) ** 2)
    return RBF
