# Awesome-L2O

This repository is to offer a TF implemented toolbox v1 for learning to optimize.

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)



## Supported Model-base Learnable Optimizers





## Supported Model-free Learnable Optimizers

1. L2O-DM from *Learning to learn by gradient descent by gradient descent* [[Paper]()] [[Code]()]
2. L2O-RNNProp *Learning Gradient Descent: Better Generalization and Longer Horizons* from [[Paper]()] [[Code]()]
3. L2O-Scale from *Learned Optimizers that Scale and Generalize* [[Paper]()] [[Code]()]
4. L2O-enhanced from *Training Stronger Baselines for Learning to Optimize* [[Paper](https://arxiv.org/pdf/2010.09089.pdf)] [[Code]()]
5. L2O-Swarm from *Learning to Optimize in Swarms* [[Paper](https://papers.nips.cc/paper/2019/file/ec04e8ebba7e132043e5b4832e54f070-Paper.pdf)] [[Code]()]
6. L2O-Jacobian from *HALO: Hardware-Aware Learning to Optimize* [[Paper](http://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123540477.pdf)] [[Code]()]
7. L2O-Minmax from *Learning A Minimax Optimizer: A Pilot Study* [[Paper](https://openreview.net/forum?id=nkIDwI6oO4_)] [[Code]()]



## Supported Optimizees

Conv Functions:

- [x] Quadratic
- [x]  Lasso

Non-convex Functions:

- [x] Rastrigin

Minmax Functions:

- [x] Saddle
- [x] Rotated Saddle
- [x] Seesaw
- [x] Matrix Game

Neural Networks:

- [x] MLPs on MNIST
- [x] ConvNets on MNIST and CIFAR-10
- [x] LeNet
- [x] NAS searched archtectures



## Other Resources

- This is a Pytorch implementation of L2O-DM. [[Code](https://github.com/chenwydj/learning-to-learn-by-gradient-descent-by-gradient-descent)]
- This is the original L2O-Swarm repository. [[Code](https://github.com/Shen-Lab/LOIS)]
- This is the original L2O-Jacobian repository. [[Code](https://github.com/RICE-EIC/HALO)]



## Future Works

- [ ] TF2.0 Implementated toolbox v2 with a unified framework and lib dependency.



## Cite

```
TBD
```

