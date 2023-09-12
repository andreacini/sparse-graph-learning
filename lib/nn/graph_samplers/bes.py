import torch
import tsl
from torch import nn
from torch.distributions import Bernoulli


class ConcreteBinarySampler(nn.Module):
    """
    Adapted from https://github.com/yaringal/ConcreteDropout
    """
    def forward(self, scores, tau):

        # Gumbel trick for a
        p = torch.sigmoid(scores)
        unif_noise = torch.rand_like(scores)
        eps = tsl.epsilon

        logit = torch.log(p + eps) - torch.log(1 - p + eps) + \
                torch.log(unif_noise + eps) - torch.log(1 - unif_noise + eps)

        soft_out = torch.sigmoid(logit / tau)

        return soft_out, None

    def mode(self, score, tau):
        return torch.sigmoid(score / tau)


class StraightThroughBinarySampler(nn.Module):
    """
    """
    def forward(self, scores, tau):
        sample = Bernoulli(logits=scores / tau).sample()
        soft_out = torch.sigmoid(scores / tau)
        return sample + soft_out - soft_out.detach(), None

    def mode(self, scores, tau):
        return torch.where(torch.sigmoid(scores / tau) > .5, 1., 0.)


class BinaryEdgeSampler(nn.Module):
    def forward(self, scores, tau, **kwargs):
        # scores (b) n n
        dist = Bernoulli(logits=scores / tau)
        sample = dist.sample()
        if self.training:
            ll = dist.log_prob(sample).sum(-1)
        else:
            ll = None
        return sample, ll

    def mode(self, scores, tau):
        return torch.where(torch.sigmoid(scores / tau) > .5, 1., 0.)

