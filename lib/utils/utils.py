import torch

from tsl.ops.connectivity import adj_to_edge_index


def adjs_to_edge_index(adjs):
    return adj_to_edge_index(adjs)


def adjs_to_fc_edge_index(adjs):
    num_nodes = adjs.shape[-1]
    adjs = adjs.transpose(-2, -1)
    edge_weight = adjs.flatten()
    idx = torch.arange(num_nodes, device=adjs.device)
    edge_index = torch.cartesian_prod(idx, idx).T
    if adjs.dim() == 3:
        edge_index = [edge_index + num_nodes * i for i in range(adjs.size(0))]
        edge_index = torch.cat(edge_index, dim=-1)
    return edge_index, edge_weight


def soft_clip(x, a):
    return a * torch.tanh(x / a)
