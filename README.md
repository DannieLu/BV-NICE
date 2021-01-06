# BV-NICE: Balancing Variational Neural Inference of Causal Effects

This is the official code repository for the NeurIPS 2020 paper [link](https://proceedings.neurips.cc/paper/2020/file/f5cfbc876972bd0d031c8abc37344c28-Paper.pdf)

```Reconsidering Generative Objectives For Counterfactual Reasoning
```

Code for individualized treatment effect (ITE) estimation for imbalanced data.

You can clone this repository by running: 

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

## Citation

If you reference or use our method, code or results in your work, please consider citing the [BV-NICE paper](https://proceedings.neurips.cc/paper/2020/file/f5cfbc876972bd0d031c8abc37344c28-Paper.pdf):

```
@article{lu2020reconsidering,
  title={Reconsidering Generative Objectives For Counterfactual Reasoning},
  author={Lu, Danni and Tao, Chenyang and Chen, Junya and Li, Fan and Guo, Feng and Carin, Lawrence},
  journal={Advances in Neural Information Processing Systems},
  volume={33},
  year={2020}
}
```

