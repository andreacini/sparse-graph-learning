"""

Adapted from https://github.com/wouterkool/estimating-gradients-without-replacement

"""

import torch
from itertools import permutations
import math


def log1mexp(x):
    # Computes log(1-exp(-|x|))
    # See https://cran.r-project.org/web/packages/Rmpfr/vignettes/log1mexp-note.pdf
    x = -x.abs()
    return torch.where(x > -0.693, torch.log(-torch.expm1(x)), torch.log1p(-torch.exp(x)))


def gumbel_log_survival(x):
    """Computes log P(g > x) = log(1 - P(g < x)) = log(1 - exp(-exp(-x))) for a standard Gumbel"""
    y = torch.exp(-x)
    return torch.where(
        x >= 10,  # means that y < 1e-4 so O(y^6) <= 1e-24 so we can use series expansion
        -x - y / 2 + y ** 2 / 24 - y ** 4 / 2880,  # + O(y^6), https://www.wolframalpha.com/input/?i=log(1+-+exp(-y))
        log1mexp(y)  # Hope for the best
    )


def all_perms(S, device=None):
    return torch.tensor(list(permutations(S)), device=device)


def log_pl_rec(log_p, dim=-1):
    """Recursive function of Plackett Luce log probability"""
    assert dim == -1
    if log_p.size(-1) == 1:
        return log_p[..., 0]
    return log_p[..., 0] + log_pl_rec(log_p[..., 1:] - log1mexp(log_p[..., 0:1]), dim=dim)


def log_pl(log_p, dim=-1):
    # compute the log likelihood of the Plackett-Luce distribution
    # P = p_1 / 1 * p_2 / (1 - p_1) * p_3 / (1 - p_1 - p_2) ...
    # log P = log p_1 - log(1) + log p_2 - log(1 - p_1) + ...
    #       = sum_i log p_i - sum_i log(1 - sum_j<i p_j)
    # Note that the first term is log_likelihood,
    # and note that sum_j<i p_j = (sum_j<=i p_j) - p_i = cumsum(p_i) - p_i
    #
    # For the stability of cumsum we can use logsumexp trick: https://gregorygundersen.com/blog/2020/02/09/log-sum-exp/

    a, _ = log_p.max(dim, keepdim=True)
    p = (log_p - a).exp()
    return log_p.sum(dim) - log1mexp(a + (p.cumsum(dim) - p).log()).sum(dim)


def compute_log_p_subset(log_p, num_points=100, a=5.):
    # Computes the (log) likelihood of an unordered sample without replacement

    # Constant for numeric stability
    a = log_p.new_tensor(a)

    # Create range of integration points, (1 ... N-1)/N (bounds are 0 to 1)
    v = torch.arange(1, num_points, out=log_p.new()) / num_points
    log_v = v.log()

    # First dim, numerical integration (N - 1)
    # Second dim, batch dimension (B)
    # Last dim, i in S (|S|)

    # compute first term of the integrand
    # first_term = log(v^(exp(a + log(1 - exp(logsumexp(log_p))) - 1)) =
    #            = log(v^(exp(a + phi_D_S) - 1)) =
    #            = (exp(a + phi_D_S) - 1) * log_v
    phi_S = torch.logsumexp(log_p, -1)
    phi_D_S = log1mexp(phi_S)

    first_term = torch.expm1(a + phi_D_S)[None] * log_v[:, None]

    # second term

    # compute g = log(-log(v^exp(log_p + a))) =
    #           = log(- exp(log_p + a) * log v) =
    #           = log(exp(log_p + a) * (- log v)) =
    #           = log_p + a + log(-log_v)
    g = (log_p + a)[None] + torch.log(-log_v[:, None, None])

    # second_term = sum log(1 - exp(-exp(g)))
    second_term = gumbel_log_survival(-g).sum(-1)

    # Compute the integrands (N - 1 x B)
    integrands = first_term + second_term

    # sum all trapezoids using the logsumexp trick
    sum_S = torch.logsumexp(integrands, 0)
    # multiply for normalization term in log space
    log_P_S = sum_S + a + phi_D_S
    # divide for number of trapezoids
    log_P_S = log_P_S - math.log(num_points)
    return log_P_S
