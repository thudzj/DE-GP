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
For MC dropout
```bash
    python fmnist.py --method free --n_ensemble 1 --dropout 0.3 --n_dropout_inf 10
```
For snapshot ensemble
```bash
    python fmnist.py --method free --n_ensemble 1 --n_snapshots 10 --epochs 4
```
For logits-space deep ensemble
```bash
    python fmnist.py --method free --logits_mean
```

## Requirement
* Python 3.6+
* Pytorch>=1.4.0
* theano==1.0.3, pymc3==3.5 (for VI and HMC baselines)
* jax==0.2.12, jaxlib==0.1.65, neural-tangents==0.3.6 (for NN-GP baseline)
* genrl
