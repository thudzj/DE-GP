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
        seed 6: N_ensemble: 10, Test set: Average loss: 0.2133, Accuracy: 0.9330, ECE: 0.0106
        seed 7: N_ensemble: 10, Test set: Average loss: 0.2151, Accuracy: 0.9343, ECE: 0.0106
    python -u cifar.py --method free --arch resnet56
        seed 6: N_ensemble: 10, Test set: Average loss: 0.1950, Accuracy: 0.9402, ECE: 0.0090
        seed 7: N_ensemble: 10, Test set: Average loss: 0.1994, Accuracy: 0.9413, ECE: 0.0090
    python -u cifar.py --method free --n_ensemble 5 --arch resnet110
```
For rDE
```bash
    python -u cifar.py --method reg
        seed 6: N_ensemble: 10, Test set: Average loss: 0.1669, Accuracy: 0.9472, ECE: 0.0103
        seed 7: N_ensemble: 10, Test set: Average loss: 0.1654, Accuracy: 0.9459, ECE: 0.0110
    python -u cifar.py --method reg --arch resnet56
        seed 6: N_ensemble: 10, Test set: Average loss: 0.1469, Accuracy: 0.9561, ECE: 0.0067
        seed 7: N_ensemble: 10, Test set: Average loss: 0.1460, Accuracy: 0.9538, ECE: 0.0092
    python -u cifar.py --method reg --n_ensemble 5 --arch resnet110
```
For RMS
```bash
    python -u cifar.py --method anc --w_alpha 0.01
        seed 6: N_ensemble: 10, Test set: Average loss: 0.2007, Accuracy: 0.9372, ECE: 0.0113
        seed 7: N_ensemble: 10, Test set: Average loss: 0.2017, Accuracy: 0.9375, ECE: 0.0104
    python -u cifar.py --method anc --arch resnet56 --w_alpha 0.01
        seed 6: N_ensemble: 10, Test set: Average loss: 0.1793, Accuracy: 0.9439, ECE: 0.0069
        seed 7: N_ensemble: 10, Test set: Average loss: 0.1781, Accuracy: 0.9455, ECE: 0.0046
    python -u cifar.py --method anc --n_ensemble 5 --arch resnet110 --w_alpha 0.01
```
For DE-GP
```bash
    python -u cifar.py --method our --f_alpha 0.01 --remove_residual --with_w_reg (use this finally)
        seed 1: N_ensemble: 10, Test set: Average loss: 0.1623, Accuracy: 0.9472, ECE: 0.0093
        seed 6: N_ensemble: 10, Test set: Average loss: 0.1647, Accuracy: 0.9461, ECE: 0.0094
        seed 7: N_ensemble: 10, Test set: Average loss: 0.1670, Accuracy: 0.9456, ECE: 0.0094
        seed 8: N_ensemble: 10, Test set: Average loss: 0.1665, Accuracy: 0.9473, ECE: 0.0110
        seed 9: N_ensemble: 10, Test set: Average loss: 0.1605, Accuracy: 0.9475, ECE: 0.0123
                                    --f_alpha 0.005
        seed 1: N_ensemble: 10, Test set: Average loss: 0.1618, Accuracy: 0.9504, ECE: 0.0092
        seed 6: N_ensemble: 10, Test set: Average loss: 0.1631, Accuracy: 0.9469, ECE: 0.0088
        seed 7: N_ensemble: 10, Test set: Average loss: 0.1646, Accuracy: 0.9488, ECE: 0.0105
        seed 8: N_ensemble: 10, Test set: Average loss: 0.1599, Accuracy: 0.9485, ECE: 0.0092
        seed 9: N_ensemble: 10, Test set: Average loss: 0.1632, Accuracy: 0.9470, ECE: 0.0089
                                    --f_alpha 0.05 (can reuse original results)
        seed 1: N_ensemble: 10, Test set: Average loss: 0.1697, Accuracy: 0.9461, ECE: 0.0120
        seed 6: N_ensemble: 10, Test set: Average loss: 0.1724, Accuracy: 0.9479, ECE: 0.0131
        seed 7:  N_ensemble: 10, Test set: Average loss: 0.1668, Accuracy: 0.9475, ECE: 0.0123
                                    --f_alpha 0.1
        seed 6: N_ensemble: 10, Test set: Average loss: 0.1696, Accuracy: 0.9473, ECE: 0.0167
        seed 7: N_ensemble: 10, Test set: Average loss: 0.1712, Accuracy: 0.9488, ECE: 0.0167
        seed 8: N_ensemble: 10, Test set: Average loss: 0.1759, Accuracy: 0.9455, ECE: 0.0185
        seed 9: N_ensemble: 10, Test set: Average loss: 0.1699, Accuracy: 0.9460, ECE: 0.0132
        seed 10: N_ensemble: 10, Test set: Average loss: 0.1697, Accuracy: 0.9461, ECE: 0.0131
    python -u cifar.py --method our --arch resnet56 --f_alpha 0.01 --remove_residual --with_w_reg
        seed 6: N_ensemble: 10, Test set: Average loss: 0.1447, Accuracy: 0.9569, ECE: 0.0098
        seed 7: N_ensemble: 10, Test set: Average loss: 0.1504, Accuracy: 0.9555, ECE: 0.0097
        seed 8: N_ensemble: 10, Test set: Average loss: 0.1439, Accuracy: 0.9550, ECE: 0.0100
        seed 9: N_ensemble: 10, Test set: Average loss: 0.1549, Accuracy: 0.9560, ECE: 0.0115
    python -u cifar.py --method our --n_ensemble 5 --arch resnet110 --f_alpha 0.02 --remove_residual --with_w_reg
```
For DE-GP (beta=0)
```bash
    python -u cifar.py --method our --f_alpha 0.01 --remove_residual (use this finally)
        seed 6: N_ensemble: 10, Test set: Average loss: 0.1966, Accuracy: 0.9358, ECE: 0.0106
        seed 7: N_ensemble: 10, Test set: Average loss: 0.1962, Accuracy: 0.9383, ECE: 0.0112
        seed 8: N_ensemble: 10, Test set: Average loss: 0.1950, Accuracy: 0.9376, ECE: 0.0107
        seed 9: N_ensemble: 10, Test set: Average loss: 0.1955, Accuracy: 0.9367, ECE: 0.0098
                                    --f_alpha 0.05 (can reuse original results)
        seed 6: N_ensemble: 10, Test set: Average loss: 0.1977, Accuracy: 0.9350, ECE: 0.0122
        seed 7: N_ensemble: 10, Test set: Average loss: 0.2001, Accuracy: 0.9368, ECE: 0.0127
    python -u cifar.py --method our --arch resnet56 --f_alpha 0.01 --remove_residual
        seed 6: N_ensemble: 10, Test set: Average loss: 0.1961, Accuracy: 0.9405, ECE: 0.0126
        seed 7: N_ensemble: 10, Test set: Average loss: 0.2055, Accuracy: 0.9428, ECE: 0.0114
        seed 8: N_ensemble: 10, Test set: Average loss: 0.1877, Accuracy: 0.9424, ECE: 0.0099  
        seed 9: N_ensemble: 10, Test set: Average loss: 0.1891, Accuracy: 0.9415, ECE: 0.0105
    python -u cifar.py --method our --n_ensemble 5 --arch resnet110 --f_alpha 0.02 --remove_residual
```


### CIFAR-10 classification under weight sharing:
For DE
```bash
    python -u cifar_ws.py --method free
        seed 1: N_ensemble: 10, Test set: Average loss: 0.5541, Accuracy: 0.9069, ECE: 0.0705
        seed 2: N_ensemble: 10, Test set: Average loss: 0.5870, Accuracy: 0.9030, ECE: 0.0734
        seed 3: N_ensemble: 10, Test set: Average loss: 0.5776, Accuracy: 0.9043, ECE: 0.0735
        seed 4: N_ensemble: 10, Test set: Average loss: 0.5483, Accuracy: 0.9076, ECE: 0.0686
        seed 5: N_ensemble: 10, Test set: Average loss: 0.5797, Accuracy: 0.9023, ECE: 0.0741
```
For rDE
```bash
    python -u cifar_ws.py --method reg
        seed 1: N_ensemble: 10, Test set: Average loss: 0.3801, Accuracy: 0.9228, ECE: 0.0499
        seed 2: N_ensemble: 10, Test set: Average loss: 0.3755, Accuracy: 0.9187, ECE: 0.0548
        seed 3: N_ensemble: 10, Test set: Average loss: 0.3580, Accuracy: 0.9219, ECE: 0.0508
        seed 4: N_ensemble: 10, Test set: Average loss: 0.3586, Accuracy: 0.9199, ECE: 0.0522
        seed 5: N_ensemble: 10, Test set: Average loss: 0.3743, Accuracy: 0.9208, ECE: 0.0528
```
For DE-GP
```bash
    python -u cifar_ws.py --method our --remove_residual --with_w_reg
        ** we choose the best...
        seed 1: N_ensemble: 10, Test set: Average loss: 0.3534, Accuracy: 0.9250, ECE: 0.0496
        seed 2: N_ensemble: 10, Test set: Average loss: 0.3549, Accuracy: 0.9243, ECE: 0.0508
        seed 3: N_ensemble: 10, Test set: Average loss: 0.3814, Accuracy: 0.9210, ECE: 0.0547
        seed 4: N_ensemble: 10, Test set: Average loss: 0.3689, Accuracy: 0.9238, ECE: 0.0527
        seed 5: N_ensemble: 10, Test set: Average loss: 0.3510, Accuracy: 0.9266, ECE: 0.0486
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
