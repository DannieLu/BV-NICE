# BV-NICE: Balancing Variational Neural Inference of Causal Effects

This is the official code repository for the NeurIPS 2020 paper [Reconsidering Generative Objectives For Counterfactual Reasoning](https://proceedings.neurips.cc/paper/2020/file/f5cfbc876972bd0d031c8abc37344c28-Paper.pdf).

BV-NICE is a novel generative Bayesian estimation framework that integrates representation learning, adversarial matching and causal estimation. It is designed for individualized treatment effect (ITE) estimation (a.k.a, conditional average treatment effect (CATE), heterogeneous treatment effect (HTE)) for observational causal inference. Existing solutions often fail to address issues that are unique to causal inference, such as covariate balancing and counterfactual validation. By appealing to the Robinson decomposition, BV-NICE exploits a reformulated variational bound that explicitly targets the causal effect estimation rather than specific predictive goals. Our procedure acknowledges the uncertainties in representation and solves a Fenchel mini-max game to resolve the representation imbalance for better counterfactual generalization, justified by new theory.The latent variable formulation enables robustness to unobservable latent confounders, extending the scope of its applicability.

You can clone this repository by running: 

```
git clone https://github.com/DannieLu/BV-NICE.git
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

## Contents

This repository contains the following contents. 

#### - Jupyter notebooks
Jupter notebook examples of our BV-NICE model and various baselines (CFR, BART, R-learner, EB-learner, Causal Forest, GANITE, etc.). 

#### - Experiment codes
Python codes used for our experiments. For example, to run BV-NICE with IHDP dataset ```0```, use the ```BVNICE.py``` file in folder: ```Experiments\IHDP\```:
```
python BVNICE.py 0
```


#### - Results and visualization
Python codes used for the visualization of our results. 

## Prerequisites

The algorithm is built with:

* Python (version 3.7 or higher)
* Tensorflow (version 1.14.0)


## Installing third-party packages
Some of our baseline models (e.g., BART, Causal Forest, GANITE, CFR) are based on third party implementations. We try to provide all native python implementations of competing models rather than calling R libraries as in the perfect_match package. Note some of these implementations are from unstable development versions of the packages, and we did find compatibility issues when we run the experiments. Please use the following commands to install the versions we have installed and used in our experiments. If they do not run successfully on your computer, just try it on another machine (with a different OS or python environment). 

We have used the BART python implementation from [bartpy](https://github.com/JakeColtman/bartpy)
```
pip3 install git+https://github.com/JakeColtman/bartpy.git@ReadOneTrees --upgrade
```

We have used the Causal Forest model (propensity forest & double-sample forest) from a dev version of [scikit-learn](https://github.com/kjung/scikit-learn) 0.18.
```
pip3 install git+https://github.com/kjung/scikit-learn.git --upgrade
```

The GANITE implementation was extracted from the [perfect_match](https://github.com/d909b/perfect_match) package . So you do not really need to install the perfect_match package to run our experiments, but we strongly recommend so. 
```
pip3 install git+https://github.com/d909b/perfect_match.git --upgrade
```

We use the CFR implementation from [https://github.com/clinicalml/cfrnet].

## Datasets
* ACIC : https://jenniferhill7.wixsite.com/acic-2016/competition 
* IHDP1000: Originally from http://mit.edu/~fredrikj/files/IHDP-1000.tar.gz 
  * The original link no longer works, so we have provided a copy in our repo for completeness. Please contact us if there is any redistribution concern. 
* JOBS: https://github.com/d909b/perfect_match/tree/master/perfect_match/data_access/jobs 
* NDS SHRP2: https://insight.shrp2nds.us (Need to apply for access (e.g., submitting research proposal))
