## L2O-Swarm

### Overview

Learning to optimize has emerged as a powerful framework for various optimization and machine learning tasks. Current such "meta-optimizers" often learn in the space of continuous optimization algorithms that are point-based and uncertainty-unaware.  To overcome the limitations, we propose a meta-optimizer that learns in the algorithmic space of both point-based and population-based optimization algorithms. The meta-optimizer targets at a meta-loss function consisting of both cumulative regret and entropy. Specifically, we learn and interpret the update formula through a population of LSTMs embedded with sample- and feature-level attentions. Meanwhile, we estimate the posterior directly over the global optimum and use an uncertainty measure to help guide the learning process.  Empirical results over non-convex test functions and a protein docking application demonstrate that this new meta-optimizer outperforms existing competitors. 

http://papers.nips.cc/paper/9641-learning-to-optimize-in-swarms



### Experiments on L2O-Scale

#### Environment

* TensorFlow >=1.13
* [cNMA](https://github.com/Shen-Lab/cNMA) (For protein docking application: sampling)
* [CHARMM](https://www.charmm.org/charmm/) (For protein docking application: initial structure minimization)

#### Important Source Files

*  meta.py:   The architecture of the model.
*  train.py:  The training program.
*  problems.py:  Store the optimization problems.
*  util.py:   Store the utility subroutines.
*  evaluate.py: The evaluation programs.

#### Train L2O-Swarm

Example: quadratic functions (see more problem options in the code)

```shell
python train.py --problem=quadratic --save_path=./quadratic
```

#### Evaluate with L2O-Swarm

Example: quadratic functions (see more problem options in the code)

```shell
python evaluate.py --problem=quadratic --optimizer=L2L --path=./quadratic
```



### Application to Protein Docking

#### File Explanation

* get_12basis.py: Get the 12 basis vectors through [cNMA](https://github.com/Shen-Lab/cNMA) 
* force_field.py: The python implementation of CHARMM19 force field (including atom charge and rdii). 
* prepocess_prot.py: Generate input data.
* dataloader.py: Load input data into the model.
* charmm_setup.prl: Set up prerequsites before runing CHARMM.

#### Datasets

* You can download data from https://drive.google.com/file/d/1x-Jye87YTWk_8WiooPJ23QJ_0zxiDT9V/view?usp=sharing.
* You can also generate your own data using above programs.
* ZDOCK data are from (https://zlab.umassmed.edu/zdock/) without CHARMM minimization.

#### Data Generation Steps

* 1.Prepare your own pdbs in a folder.
* 2.Modify the path in prepocess_prot.py and get_12basis.py to be the folder of your own pdbs.
* 3.Download CHARMM27 and minimize the pdb complex through CHARMM.
* 3.Download cNMA, and change the path of cNMA in get_12basis.py
* 4.run get_12basis.py, force_filed.py and then prepocess_prot.py.



### Citation

```
@incollection{NIPS2019_9641,
title = {Learning to Optimize in Swarms},
author = {Cao, Yue and Chen, Tianlong and Wang, Zhangyang and Shen, Yang},
booktitle = {Advances in Neural Information Processing Systems 32},
editor = {H. Wallach and H. Larochelle and A. Beygelzimer and F. d\textquotesingle Alch\'{e}-Buc and E. Fox and R. Garnett},
pages = {15018--15028},
year = {2019},
publisher = {Curran Associates, Inc.},
url = {http://papers.nips.cc/paper/9641-learning-to-optimize-in-swarms.pdf}
}
```








