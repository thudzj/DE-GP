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

### CIFAR-100 classification (ssh -p 10022 dengzhijie@106.38.203.236):
For DE
```bash
    CUDA_VISIBLE_DEVICES=2 python -u cifar.py --dataset cifar100 --data-root /data1/dengzhijie/cifar --save-dir /data1/dengzhijie/snapshots_degp/ --method free
    CUDA_VISIBLE_DEVICES=2 python -u cifar.py --dataset cifar100 --data-root /data1/dengzhijie/cifar --save-dir /data1/dengzhijie/snapshots_degp/ --method free --arch resnet56
```
For rDE
```bash
    CUDA_VISIBLE_DEVICES=3 python -u cifar.py --dataset cifar100 --data-root /data1/dengzhijie/cifar --save-dir /data1/dengzhijie/snapshots_degp/ --method reg
    CUDA_VISIBLE_DEVICES=3 python -u cifar.py --dataset cifar100 --data-root /data1/dengzhijie/cifar --save-dir /data1/dengzhijie/snapshots_degp/ --method reg --arch resnet56
```
For RMS
```bash
    CUDA_VISIBLE_DEVICES=4 python -u cifar.py --dataset cifar100 --data-root /data1/dengzhijie/cifar --save-dir /data1/dengzhijie/snapshots_degp/ --method anc --w_alpha 0.01
    CUDA_VISIBLE_DEVICES=4 python -u cifar.py --dataset cifar100 --data-root /data1/dengzhijie/cifar --save-dir /data1/dengzhijie/snapshots_degp/ --method anc --w_alpha 0.01 --arch resnet56
```
For DE-GP
```bash
    CUDA_VISIBLE_DEVICES=7 python -u cifar.py --dataset cifar100 --data-root /data1/dengzhijie/cifar --save-dir /data1/dengzhijie/snapshots_degp/ --method our --f_alpha 0.01 --remove_residual --with_w_reg --test-batch-size 100
    (N_ensemble: 10, Test set: Average loss: 0.8599, Accuracy: 0.7659, ECE: 0.0585)
    CUDA_VISIBLE_DEVICES=4 python -u cifar.py --dataset cifar100 --data-root /data1/dengzhijie/cifar --save-dir /data1/dengzhijie/snapshots_degp/ --method our --arch resnet56 --f_alpha 0.01 --remove_residual --with_w_reg -b 100 --test-batch-size 100
    (N_ensemble: 10, Test set: Average loss: 0.7754, Accuracy: 0.7951, ECE: 0.0303)
```

### Contextual bandit:
```bash
    python contextual_bandit.py --run-experiment --download --bandit covertype
    python contextual_bandit.py --run-experiment --download --bandit mushroom
```

## Requirement
* Python 3.6+
* Pytorch>=1.4.0
* theano==1.0.3, pymc3==3.5 (for VI and HMC baselines)
* jax==0.2.12, jaxlib==0.1.65, neural-tangents==0.3.6 (for NN-GP baseline)
* genrl
