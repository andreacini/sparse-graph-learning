import torch

from lib.nn.metrics import MaskedScoreFuctionLoss
from tsl.engines import Predictor


class LatentGraphPredictor(Predictor):
    def __init__(self,
                 graph_module_class,
                 graph_module_kwargs,
                 model_class,
                 model_kwargs,
                 optim_class,
                 optim_kwargs,
                 loss_fn,
                 scale_target=False,
                 metrics=None,
                 scheduler_class=None,
                 scheduler_kwargs=None,
                 mc_samples=1,
                 eval_mode='mode'):
        super().__init__(model_class=model_class,
                         model_kwargs=model_kwargs,
                         optim_class=optim_class,
                         optim_kwargs=optim_kwargs,
                         loss_fn=loss_fn,
                         scale_target=scale_target,
                         metrics=metrics,
                         scheduler_class=scheduler_class,
                         scheduler_kwargs=scheduler_kwargs)

        self.graph_module_class = graph_module_class
        self.graph_module_kwargs = graph_module_kwargs
        self.graph_module = self.graph_module_class(**self.graph_module_kwargs)
        self.mc_samples = mc_samples
        self.eval_mode = eval_mode

    def forward(self, *args, mode='forward', **kwargs):
        if mode == 'pred_only':
            connectivity = kwargs.pop('connectivity', None)
            if connectivity is not None:
                kwargs.update(**connectivity)
            return self.model(*args, **kwargs)
        if mode == 'graph_only':
            return self.graph_module(*args, **kwargs)
        if self.training or mode == 'sampling':
            connectivity = self.graph_module(*args, **kwargs)
            kwargs.update(**connectivity)
            return self.model(*args, **kwargs)

        if self.eval_mode == 'sampling':
            # if in inference mode, take the average of M MC samples
            outs = []
            for _ in range(self.mc_samples):
                out = self.forward(*args, **kwargs, mode='sampling')
                outs.append(out)
            return torch.stack(outs).mean(0)
        if self.eval_mode == 'mode':
            connectivity = self.graph_module(*args, **kwargs)['mean_graph']
            kwargs.update(**connectivity, disjoint=True, adj=None)
            return self.model(*args, **kwargs)


class SFGraphPredictor(LatentGraphPredictor):
    def __init__(self,
                 graph_module_class,
                 graph_module_kwargs,
                 model_class,
                 model_kwargs,
                 optim_class,
                 optim_kwargs,
                 loss_fn,
                 scale_target=False,
                 metrics=None,
                 scheduler_class=None,
                 scheduler_kwargs=None,
                 use_baseline=False,
                 sf_weight=1.,
                 mc_samples=1,
                 eval_mode='mode',
                 variance_reduced=True,
                 surrogate_lam=None):
        super().__init__(graph_module_class=graph_module_class,
                         graph_module_kwargs=graph_module_kwargs,
                         model_class=model_class,
                         model_kwargs=model_kwargs,
                         optim_class=optim_class,
                         optim_kwargs=optim_kwargs,
                         loss_fn=loss_fn,
                         scale_target=scale_target,
                         metrics=metrics,
                         scheduler_class=scheduler_class,
                         scheduler_kwargs=scheduler_kwargs,
                         mc_samples=mc_samples,
                         eval_mode=eval_mode,)

        self.sf_loss = MaskedScoreFuctionLoss(
            cost_fn=self.loss_fn.metric_fn,
            variance_reduced=variance_reduced,
            lam=surrogate_lam
        )

        self.use_baseline = use_baseline
        self.sf_weight = sf_weight

    def training_step(self, batch, batch_idx):

        y = batch.y
        mask = batch.mask

        connectivity = self.predict_batch(batch,
                                          preprocess=False,
                                          postprocess=False,
                                          mode='graph_only')

        mean_graph = connectivity.pop('mean_graph')

        y_hat_scaled = self.predict_batch(batch,
                                          preprocess=False,
                                          postprocess=False,
                                          mode='pred_only',
                                          connectivity=connectivity)

        y_hat = batch.transform['y'].inverse_transform(y_hat_scaled)
        y_scaled = batch.transform['y'].transform(y)

        if self.use_baseline:
            with torch.no_grad():

                edge_index = mean_graph['edge_index']
                edge_weight = mean_graph['edge_weight']

                conn = dict(
                    edge_index=edge_index,
                    edge_weight=edge_weight,
                    adj=None,
                    disjoint=True
                )
                y_b = self.predict_batch(batch,
                                         preprocess=False,
                                         postprocess=not self.scale_target,
                                         forward_kwargs=dict(
                                             mode='pred_only',
                                             connectivity=conn
                                         ))
        else:
            y_b = None

        # Scale target and output, eventually
        if self.scale_target:
            y_hat_loss = y_hat_scaled
            y_loss = y_scaled
        else:
            y_hat_loss = y_hat
            y_loss = y

        # Compute loss
        # Prediction loss
        pred_loss = self.loss_fn(y_hat_loss, y_loss, mask)
        b_loss = self.loss_fn(y_b, y_loss, mask)
        # Graph loss
        score = connectivity['ll']
        graph_loss = self.sf_loss(score=score,
                                  y_hat=y_hat_loss.detach(),
                                  y=y_loss,
                                  y_b=y_b,
                                  mask=mask)

        # Logging
        self.train_metrics.update(y_hat.detach(), y, mask)
        self.log_metrics(self.train_metrics, batch_size=batch.batch_size)
        self.log_loss('train', pred_loss, batch_size=batch.batch_size)
        self.log_loss('train_baseline', b_loss, batch_size=batch.batch_size)
        self.log_loss('graph', graph_loss, batch_size=batch.batch_size)
        return pred_loss + self.sf_weight * graph_loss
