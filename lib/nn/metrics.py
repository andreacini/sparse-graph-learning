from torchmetrics.utilities.checks import _check_same_shape
from functools import partial
import warnings

from torchmetrics import Metric

import torch

from einops import rearrange

class MaskedScoreFuctionLoss(Metric):
    r"""
    Base class to implement the metrics used in `tsl`.

    In particular a `MaskedMetric` accounts for missing values in the input sequences by accepting a boolean mask as
    additional input.

    Args:
        cost_fn: Base function to compute the metric point wise.
        mask_nans (bool, optional): Whether to automatically mask nan values.
        mask_inf (bool, optional): Whether to automatically mask infinite values.
        compute_on_step (bool, optional): Whether to compute the metric right-away or to accumulate the results.
                         This should be `True` when using the metric to compute a loss function, `False` if the metric
                         is used for logging the aggregate value across different mini-batches.
        at (int, optional): Whether to compute the metric only w.r.t. a certain time step.
    """
    is_differentiable = True
    full_state_update = False
    def __init__(self,
                 cost_fn,
                 variance_reduced=True,
                 cost_momentum=0.1,
                 cost_kwargs=None,
                 lam=None,
                 **kwargs):
        super(MaskedScoreFuctionLoss, self).__init__(**kwargs)

        if cost_kwargs is None:
            cost_kwargs = dict()
        self.cost_fn = partial(cost_fn, **cost_kwargs)
        self.ma_momentum = cost_momentum
        self.variance_reduced = variance_reduced
        self.lam = lam
        self.add_state('value', dist_reduce_fx='sum', default=torch.tensor(0., dtype=torch.float))
        self.add_state('numel', dist_reduce_fx='sum', default=torch.tensor(0., dtype=torch.float))

    def _check_mask(self, mask, val):
        if mask is None:
            mask = torch.ones_like(val, dtype=torch.bool)
        else:
            mask = mask.bool()
            _check_same_shape(mask, val)
        return mask

    def compute_cost(self, y_hat, y, mask):
        _check_same_shape(y_hat, y)
        cost = self.cost_fn(y_hat, y)
        mask = self._check_mask(mask, cost)
        cost = torch.where(mask, cost, torch.zeros_like(cost))
        return cost, mask

    def update(self, score, y_hat, y, y_b=None, mask=None):
        # make sure the score has a proper shape
        if score.dim() == 1:
            score = score[None]
        score = rearrange(score, 'b n -> b 1 n 1')
        # account for
        score_mask = torch.isfinite(score)
        if not score_mask.all():
            warnings.warn("Nan values in scores.")
            score = torch.where(score_mask, score, torch.zeros_like(score))

        # make sure the mask has a proper shape
        mask = self._check_mask(mask, y)
        mask = mask.float() * score_mask.float()

        cost, mask = self.compute_cost(y_hat, y, mask)

        # compute baseline
        if y_b is not None:
            b, _ = self.compute_cost(y_b, y, mask)
            cost = cost - b

        # make sure the cost is considered as a constant
        cost = cost.detach()

        # cost shape : b n
        if self.lam is None:
            lam = 1 / score.size(2)  # lam = 1 / num_nodes
        else:
            lam = self.lam
        if self.variance_reduced:
            # compute surrogate loss
            val = cost * score + lam * cost * score.sum(-2, keepdims=True)
        else:
            # compute standard loss
            val = cost * score.sum(-2, keepdims=True)
        self.value += val.sum()
        self.numel += mask.sum()

    def compute(self):
        if self.numel > 0:
            return self.value / self.numel
        return self.value
