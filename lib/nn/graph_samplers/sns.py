import torch
from torch import nn

from torch.distributions import Gumbel

from einops import rearrange

import lib.utils.distributions as sutils


class SubsetNeighborhoodSampler(nn.Module):
    """"""
    def __init__(self, k, estimator='unord'):
        super(SubsetNeighborhoodSampler, self).__init__()
        self.k = k
        self.estimator = estimator

    def _sample(self, phi):
        # phi = torch.log_softmax(scores, 1)
        g_phi = Gumbel(phi, torch.ones_like(phi))
        sample = g_phi.sample()
        # top k
        khot = torch.zeros_like(sample)
        _, ind = torch.topk(sample, self.k, dim=-1)
        khot.scatter_(-1, ind, 1)
        return ind, khot

    def _compute_log_p(self, scores, topk):
        log_p = scores - torch.logsumexp(scores, dim=-1, keepdim=True)
        # take log_p of the sampled nodes
        log_p_k = torch.gather(log_p, -1, topk)
        if self.estimator == 'ord':
            # compute likelihood of the ordered sample
            # normalize probabilities
            ll = sutils.log_pl_rec(log_p_k, -1)
        elif self.estimator == 'unord':
            # take log_p of the sampled nodes
            # approximate log likelihood with numerical integration
            b, *_ = log_p_k.size()
            log_p_k = rearrange(log_p_k, 'b n k -> (b n) k')
            ll = sutils.compute_log_p_subset(log_p_k, num_points=100, a=5)
            ll = rearrange(ll, '(b n) -> b n', b=b)
        else:
            raise NotImplementedError
        return ll

    def forward(self, scores, tau, **kwargs):
        topk, khot = self._sample(scores / tau)

        if not self.training:
            ll = None
        else:
            ll = self._compute_log_p(scores / tau, topk)

        return khot, ll

    def mode(self, scores, tau, **kwargs):
        # top k
        khot = torch.zeros_like(scores.detach())
        _, ind = torch.topk(scores, self.k, dim=-1)
        khot.scatter_(-1, ind, 1)
        return khot


class StraightThroughSubsetSampler(nn.Module):
    """"""
    def __init__(self, k):
        super(StraightThroughSubsetSampler, self).__init__()
        self.k = k

    def forward(self, scores, tau, **kwargs):
        phi = scores / tau
        g_phi = Gumbel(phi, torch.ones_like(phi))
        gumbels = g_phi.sample()
        # top k
        khot = torch.zeros_like(gumbels)
        _, ind = torch.topk(gumbels, self.k, dim=-1)
        khot.scatter_(-1, ind, 1.)

        soft_out = torch.softmax(phi, -1)
        return khot + soft_out - soft_out.detach(), None

    def mode(self, scores, tau, **kwargs):
        # top k
        khot = torch.zeros_like(scores.detach())
        _, ind = torch.topk(scores, self.k, dim=-1)
        khot.scatter_(-1, ind, 1.)
        return khot

