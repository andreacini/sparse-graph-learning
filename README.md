# Sparse Graph Learning from Spatiotemporal Time Series (JMLR 2023)

[![JMLR](https://img.shields.io/badge/JMLR-2023-blue.svg?style=flat-square)](https://www.jmlr.org/papers/v24/22-1154.html)
[![PDF](https://img.shields.io/badge/%E2%87%A9-PDF-orange.svg?style=flat-square)](https://www.jmlr.org/papers/volume24/22-1154/22-1154.pdf)
[![arXiv](https://img.shields.io/badge/arXiv-2205.13492-b31b1b.svg?style=flat-square)](https://arxiv.org/abs/2205.13492)

This repository contains the code for the reproducibility of the controlled experiments presented in the paper "Sparse Graph Learning from Spatiotemporal Time Series" (JMLR 2023).

**Authors**: [Andrea Cini](mailto:andrea.cini@usi.ch), [Daniele Zambon](mailto:daniele.zambon@usi.ch), Cesare Alippi

---

## In a nutshell

We propose novel, principled - yet practical - probabilistic score-based methods that learn the relational dependencies as distributions over graphs while maximizing end-to-end the performance at task. The proposed graph learning framework is based on consolidated variance reduction techniques for Monte Carlo score-based gradient estimation, is theoretically grounded, and, as we show, effective in practice. In this paper, we focus on the time series forecasting problem and show that, by tailoring the gradient estimators to the graph learning problem, we are able to achieve state-of-the-art performance while controlling the sparsity of the learned graph and the computational scalability.

---

## Directory structure

The directory is structured as follows:

```
.
├── config/synthetic/
│   └── defaults.yaml
├── lib/
├── conda_env.yaml
├── default_config.yaml
└── experiments/
    └── run_synthetic.py

```

## Configuration files

The `config` directory stores all the configuration file used to run the experiments.

## Requirements

We run all the experiments in `python 3.10`. To solve all dependencies, we recommend using Anaconda and the provided environment configuration by running the command:

```bash
conda env create -f conda_env.yaml
conda activate sgl
```

## Library

The support code, including the models and the datasets readers, are provided within in a python library named `lib` which heavily relies on [tsl](https://torch-spatiotemporal.readthedocs.io/en/latest/).

## Experiments

The script used for the experiments is in the `experiments` folder.

The script `run_synthetic.py` is used to run experiments on the GPVAR dataset. An example of usage is

```
CUDA_VISIBLE_DEVICES=0 python -m experiments.run_synthetic config=defaults
```

which runs the experiment with the default configuration (SNS with variance reduction and dummy nodes), on the GPU with ID 0. To run experiments in different settings, you can edit the configuration file in the `config/synthetic` folder. 

## Bibtex reference

If you find this code useful please consider to cite our paper:

```
@article{cini2023sparse,
  title={Sparse Graph Learning from Spatiotemporal Time Series},
  author={Cini, Andrea and Zambon, Daniele and Alippi, Cesare},
  journal={Journal of Machine Learning Research},
  volume={24},
  number={242},
  pages={1--36},
  year={2023}
}
```