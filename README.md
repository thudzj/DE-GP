# Deep Ensemble as a Gaussian Process Posterior [[arXiv]](#)
```BibTex
None
```

## How to train **DE-GP**

### UCI regression:
```bash
    python uci.py
```

### FashionMNIST classification:
For DE
```bash
    python fmnist.py --method free
```
For rDE
```bash
    python fmnist.py --method reg
```
For RMS
```bash
    python fmnist.py --method anc
```
For DE-GP
```bash
    python fmnist.py --method our
```

### CIFAR-10 classification:
For DE
```bash
    python -u cifar.py --method free
    python -u cifar.py --method free --arch resnet56
    python -u cifar.py --method free --n_ensemble 5 --arch resnet110
```
For rDE
```bash
    python -u cifar.py --method reg
    python -u cifar.py --method reg --arch resnet56
    python -u cifar.py --method reg --n_ensemble 5 --arch resnet110
```
For RMS
```bash
    python -u cifar.py --method anc --w_alpha 0.01
    python -u cifar.py --method anc --arch resnet56 --w_alpha 0.01
    python -u cifar.py --method anc --n_ensemble 5 --arch resnet110 --w_alpha 0.01
```
For DE-GP
```bash
    python -u cifar.py --method our --f_alpha 0.05 --remove_residual --with_w_reg
    python -u cifar.py --method our --arch resnet56 --f_alpha 0.01 --remove_residual --with_w_reg
    python -u cifar.py --method our --n_ensemble 5 --arch resnet110 --f_alpha 0.02 --remove_residual --with_w_reg
```


### CIFAR-10 classification under weight sharing:
For DE
```bash
    python -u cifar_ws.py --method free
```
For rDE
```bash
    python -u cifar_ws.py --method reg
```
For DE-GP
```bash
    python -u cifar_ws.py --method our --remove_residual --with_w_reg
```

## Requirement
* Python 3.6+
* Pytorch==1.7.1
* theano==1.0.3, pymc3==3.5 (for VI and HMC baselines)
* jax==0.2.12, jaxlib==0.1.65, neural-tangents==0.3.6 (for NN-GP baseline)
