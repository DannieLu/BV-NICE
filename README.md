# BV-NICE: Balancing Variational Neural Inference of Causal Effects

Code for individualized treatment effect (ITE) estimation for imbalanced data.

Clone the repository, e.g:

```
git clone https://github.com/DannieLu/BV-NICE.git
```

## Prerequisites

The algorithm is built with:

* Python (version 3.7 or higher)
* Tensorflow (version 1.14.0)

## Running the example dataset

Here we present an example of BVNICE application using IHDP data

```
python bvnice_example.py
```

## Cite [the paper](https://proceedings.neurips.cc/paper/2020/file/f5cfbc876972bd0d031c8abc37344c28-Paper.pdf):

```
@article{lu2020reconsidering,
  title={Reconsidering Generative Objectives For Counterfactual Reasoning},
  author={Lu, Danni and Tao, Chenyang and Chen, Junya and Li, Fan and Guo, Feng and Carin, Lawrence},
  journal={Advances in Neural Information Processing Systems},
  volume={33},
  year={2020}
}
```

