import os

from tsl.experiment import Experiment

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor

from tsl.data import SpatioTemporalDataset, SpatioTemporalDataModule
from tsl.data.preprocessing import StandardScaler

import tsl

from tsl.metrics.torch import MaskedMAE, MaskedMAPE, MaskedMSE

import torch


import lib

from lib.predictors.latent_graph_predictor import LatentGraphPredictor, SFGraphPredictor
from lib.nn.graph_module import GraphModule
from lib.datasets.graph_polynomial_var import GraphPolyVARDataset


def get_dataset(dataset_name):
    if dataset_name == 'gpolyvar':
        T = 30000
        communities = 5
        connectivity = "line"
        data_path = os.path.join(lib.config['data_dir'], f"gpvar-T{T}_{connectivity}-c{communities}")

        dataset = GraphPolyVARDataset(coefs=torch.tensor([[5, 2], [-4, 6], [-1, 0]], dtype=torch.float32),
                                      sigma_noise=.4,
                                      communities=communities, connectivity=connectivity)
        dataset.generate_data(T=T)
        dataset.dump_dataset(path=data_path)
    else:
        raise ValueError(f"Dataset {dataset_name} not available in this setting.")
    return dataset


def run_experiment(cfg):
    dataset = get_dataset(cfg.dataset.name)

    gm_class = GraphModule

    ########################################
    # data module                          #
    ########################################

    adj = None

    torch_dataset = SpatioTemporalDataset(*dataset.numpy(return_idx=True),
                                          connectivity=adj,
                                          mask=dataset.mask.bool(),
                                          horizon=cfg.horizon,
                                          window=cfg.window,
                                          stride=cfg.stride)

    dm = SpatioTemporalDataModule(
        dataset=torch_dataset,
        scalers={'target': StandardScaler(axis=(0, 1))},
        splitter=dataset.get_splitter(**cfg.dataset.splits),
        batch_size=cfg.batch_size,
        workers=cfg.workers
    )

    dm.setup()
    ########################################
    # predictor                            #
    ########################################
    gm_kwargs = dict(n_nodes=torch_dataset.n_nodes,
                     mode=cfg.graph_mode)

    gm_kwargs.update(cfg.graph_module.hparams)

    loss_fn = MaskedMAE()

    metrics = {'mae': MaskedMAE(),
               'mse': MaskedMSE(),
               'mape': MaskedMAPE()}

    if cfg.graph_mode == 'pd' or cfg.graph_mode == 'st':
        predictor_class = LatentGraphPredictor
        pred_kwargs = dict(graph_module_class=gm_class,
                           graph_module_kwargs=gm_kwargs,
                           mc_samples=cfg.mc_samples,
                           )

    elif cfg.graph_mode == 'sf':
        predictor_class = SFGraphPredictor
        pred_kwargs = dict(sf_weight=cfg.sf_weight,
                           graph_module_class=gm_class,
                           graph_module_kwargs=gm_kwargs,
                           use_baseline=cfg.use_baseline,
                           mc_samples=cfg.mc_samples,
                           variance_reduced=cfg.variance_reduced,
                           surrogate_lam=cfg.lam)
    else:
        raise NotImplementedError(f"Graph learning mode {cfg.graph_mode} not available.")

    model_cls = dataset.model_class
    if cfg.experiment_name == 'graph_id':
        model_kwargs = dataset.model_kwargs
    elif cfg.experiment_name == 'joint':
        model_kwargs = dict(
            spatial_order=3,
            temporal_order=4
        )
    elif cfg.experiment_name == 'joint_hard':
        model_kwargs = dict(
            spatial_order=4,
            temporal_order=6
        )
    else:
        raise NotImplementedError(f"Experiment {cfg.experiment_name} not avaiable.")

    scheduler_class = None
    scheduler_kwargs = None

    predictor = predictor_class(
        model_class=model_cls,
        model_kwargs=model_kwargs,
        optim_class=torch.optim.Adam,
        optim_kwargs=dict(cfg.optimizer.hparams),
        loss_fn=loss_fn,
        metrics=metrics,
        scheduler_class=scheduler_class,
        scheduler_kwargs=scheduler_kwargs,
        scale_target=False,
        **pred_kwargs
    )

    ########################################
    # training                             #
    ########################################

    early_stop_callback = EarlyStopping(
        monitor='val_mae',
        patience=cfg.patience,
        mode='min'
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=cfg.run.dir,
        save_top_k=1,
        monitor='val_mae',
        mode='min',
    )

    lr_monitor = LearningRateMonitor(
        logging_interval='epoch'
    )

    batches_epoch = 1.0 if cfg.batches_epoch < 0 else cfg.batches_epoch
    trainer = pl.Trainer(max_epochs=cfg.epochs,
                         limit_train_batches=batches_epoch,
                         default_root_dir=cfg.run.dir,
                         accelerator='gpu' if torch.cuda.is_available() else 'cpu',
                         callbacks=[early_stop_callback,
                                    checkpoint_callback,
                                    lr_monitor],
                         gradient_clip_algorithm='value',
                         gradient_clip_val=.5 if cfg.clip_grad else None,
                         )

    tsl.logger.info(f"Optimal MAE (analytical) {dataset.mae_optimal}")

    trainer.fit(predictor,
                train_dataloaders=dm.train_dataloader(),
                val_dataloaders=dm.val_dataloader())

    ########################################
    # testing                              #
    ########################################

    predictor.load_state_dict(
        torch.load(checkpoint_callback.best_model_path,
                   lambda storage, loc: storage)['state_dict'])

    predictor.freeze()
    trainer.test(predictor, dataloaders=dm.test_dataloader())


if __name__ == '__main__':
    exp = Experiment(run_fn=run_experiment, config_path='config/synthetic')
    exp.run()
