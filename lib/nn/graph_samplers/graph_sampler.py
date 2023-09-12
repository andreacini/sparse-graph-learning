import torch
from torch import nn

from lib.nn.graph_samplers.bes import ConcreteBinarySampler, \
                                      StraightThroughBinarySampler, \
                                      BinaryEdgeSampler
from lib.nn.graph_samplers.sns import SubsetNeighborhoodSampler, StraightThroughSubsetSampler
from lib.utils.utils import adjs_to_edge_index, adjs_to_fc_edge_index


class GraphSampler(nn.Module):
    def __init__(self, mode, sampler_type, k=None, tau=1., dummy_nodes=0):
        super(GraphSampler, self).__init__()
        if mode == 'pd':
            if sampler_type == 'bes':
                self.k = None
                self.sampler = ConcreteBinarySampler()
            else:
                raise NotImplementedError('Sampler {} not implemented for mode {}'.format(sampler_type, mode))
        elif mode == 'st':
            if sampler_type == 'bes':
                self.k = None
                self.sampler = StraightThroughBinarySampler()
            elif sampler_type == 'sns':
                self.k = k
                self.sampler = StraightThroughSubsetSampler(k=k)
            else:
                raise NotImplementedError('Sampler {} not implemented for mode {}'.format(sampler_type, mode))
        elif mode == 'sf':
            if sampler_type == 'sns':
                self.k = k
                self.sampler = SubsetNeighborhoodSampler(k=k)
            elif sampler_type == 'bes':
                self.k = None
                self.sampler = BinaryEdgeSampler()
            else:
                raise NotImplementedError('Sampler {} not implemented for mode {}'.format(sampler_type, mode))
        tau = torch.tensor(tau)
        self.register_buffer('tau', tau)
        self.dummy_nodes = dummy_nodes
        self.sampling_mode = mode

    def forward(self, scores):
        adj, ll = self.sampler(scores, tau=self.tau)
        adj, edge_index = self.to_connectivity(adj)
        if self.dummy_nodes > 0 and ll is not None:
            assert ll.shape[-1] == adj.shape[-1] + self.dummy_nodes
            ll = ll[..., :-self.dummy_nodes]
        return adj, edge_index, ll

    def to_connectivity(self, adjs):
        # remove dummy nodes
        if self.dummy_nodes > 0:
            adjs = adjs[..., :-self.dummy_nodes, :-self.dummy_nodes]

        if self.sampling_mode == 'sf':
            edge_index, edge_weight = adjs_to_edge_index(adjs)
        else:
            edge_index, edge_weight = adjs_to_fc_edge_index(adjs)

        return adjs, (edge_index, edge_weight)

    def mode(self, scores):
        adj = self.sampler.mode(scores, tau=self.tau)
        return self.to_connectivity(adj)
