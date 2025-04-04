# Spectral Graph Pruning Against Over-squashing and Over-smoothing (NeurIPS 2024)

[![arXiv](https://img.shields.io/badge/arXiv-2404.04612-b31b1b.svg)](https://arxiv.org/abs/2404.04612)
[![NeurIPS](https://img.shields.io/badge/NeurIPS-2024-purple.svg)](https://openreview.net/forum?id=EMkrwJY2de)

>Adarsh Jamadandi\*, Celia Rubio-Madrigal\*, Rebekka Burkholz.
>CISPA Helmholtz Center for Information Security. \* Equal Contribution.

This repository contains the code for the paper "Spectral Graph Pruning Against Over-squashing and Over-smoothing", which has been accepted at NeurIPS 2024!

#### Abstract

>Message Passing Graph Neural Networks are known to suffer from two problems that are sometimes believed to be diametrically opposed: over-squashing and over-smoothing. The former results from topological bottlenecks that hamper the information flow from distant nodes and are mitigated by spectral gap maximization, primarily, by means of edge additions. However, such additions often promote over-smoothing that renders nodes of different classes less distinguishable. Inspired by the Braess phenomenon, we argue that deleting edges can address over-squashing and over-smoothing simultaneously. This insight explains how edge deletions can improve generalization, thus connecting spectral gap optimization to a seemingly disconnected objective of reducing computational resources by pruning graphs for lottery tickets. To this end, we propose a computationally effective spectral gap optimization framework to add or delete edges and demonstrate its effectiveness on the long range graph benchmark and on larger heterophilous datasets.

![Braess Ring](https://github.com/AdarshMJ/SpectralPruningBraess/blob/main/BraessRing.jpg)

## Requirements

The following libraries are required to run the code and reproduce the results:

```Python
Pytorch-Geometric == 2.5.3
Pytorch ==2.2.1
DGL == 2.1.0+cu118
```

Alternatively, the exact environment can be recreated using the provided `environment.yml` file.

## Usage

To run the code, use the provided script:

```bash
bash gen_spect.sh
```

## Structure

The [NodeClassification/rewiring](NodeClassification/rewiring) folder contains the following methods for graph rewiring:

1. Proxyaddmax - add edges by maximizing proxy spectral gap.
2. Proxyaddmin - add edges by minmizing proxy spectral gap.
3. Proxydelmax - delete edges by maximizing proxy spectral gap.
4. Proxydelmin - delete edges by maximizing proxy spectral gap.
5. FoSR - add edges by maximizing first order approximation of spectral gap proposed in [1].

## Datasets
   
* Cora, Citeseer, and Pubmed are automatically downloaded from [Pytorch-Geometric](https://pytorch-geometric.readthedocs.io/en/latest/modules/datasets.html#torch_geometric.datasets.Planetoid).

* For Cornell, Texas, Wisconsin, Chameleon, Squirrel and Actor we use the .npz files given in the official repository of [2]: [https://github.com/yandex-research/heterophilous-graphs](https://github.com/yandex-research/heterophilous-graphs)

* For generating results for large heterophilic graphs:

1. Download the code base from [2]: [https://github.com/yandex-research/heterophilous-graphs](https://github.com/yandex-research/heterophilous-graphs)
2. Install DGL
3. Change `datasets.py` to include the rewiring schemes. We also provide the [`datasets.py`](datasets.py) file that needs to be replaced and can be used to perform graph rewiring on the large heterophilic graphs.
4. Adjust the learning rate (LR = 3e-4 or LR = 3e-3) for improved results on these datasets.

[1] Kedar Karhadkar, Pradeep Kr. Banerjee, Guido Montufar. FoSR: First-order spectral rewiring for addressing oversquashing in GNNs. The Eleventh International Conference on Learning Representations, 2023.

[2] Oleg Platonov, Denis Kuznedelev, Michael Diskin, Artem Babenko, Liudmila Prokhorenkova. A critical look at evaluation of GNNs under heterophily: Are we really making progress? The Eleventh International Conference on Learning Representations, 2023.

## Citation

If you found this work helpful, please consider citing our paper:

```bibtex
@inproceedings{jamadandi2024spectral,
title={Spectral Graph Pruning Against Over-Squashing and Over-Smoothing},
author={Adarsh Jamadandi and Celia Rubio-Madrigal and Rebekka Burkholz},
booktitle={The Thirty-eighth Annual Conference on Neural Information Processing Systems},
year={2024}
}
```