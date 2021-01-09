## L2O-Minimax

### Overview

Solving continuous minimax optimization is of extensive practical interest, yet notoriously unstable and difficult. We introduce the *learning to optimize* (L2O) methodology to the minimax problems for the first time, and addresses its accompanying unique challenges. We present *Twin L2O*, the first dedicated minimax L2O framework consisting of two LSTMs for updating min and max variables, respectively.  We then discuss a crucial concern of Twin-L2O, i.e., its inevitably limited generalizability to unseen optimizees, and present two complementary strategies. Our first solution, *Enhanced Twin-L2O*, is empirically applicable for general minimax problems, by improving L2O training via leveraging curriculum learning. Our second alternative, called *Safeguarded Twin L2O*, is a preliminary theoretical exploration stating that under some strong assumptions, it is possible to theoretically establish the convergence of Twin-L2O. 

### Configuration

The configuation files are stored in `/config` folder, depicting the experimental setting.

### Toy Examples Using Twin-L2O

Four examples are demonstrated, where $a \sim U[0.9,1]$, $b \sim U[0.9,1]$.

- Saddle $\min _{x} \max _{y} ax^2-by^2$
- Rotated Saddle  $\min _{x} \max _{y} ax^2-by^2+2xy$
- Seesaw  $\min _{x} \max _{y} -bvsin(a \pi u)$
- Matrix Game $\min _{\mathbf{x}} \max _{\mathbf{y}} \mathbf{x}^{T} \mathbf{A} \mathbf{y}, \mathbf{A} \in \mathbb{R}^{5 \times 5}, \mathbf{A}_{i, j} \sim \operatorname{Bernoulli}(0.5) \cdot U[-1,1]$

#### Dataset

`/dataset/range1_train.txt`, `/dataset/range1_eval.txt` and `/dataset/range1_test.txt`  contain the data of $a$  and $b$  to construct training, evaluating, testing problems respectively , which are randomly generated using `np.random`. 

#### Training

In order to perform the training process, run the following command:

```shell
python Twin-L2O.py --config saddle_train  #saddle
python Twin-L2O.py --config rotatedsaddle_train  #rotated saddle
python Twin-L2O.py --config seesaw_train  #seesaw
```

#### Tesing

In order to perform the testing process, run the following command:

```shell
python Twin-L2O.py --config saddle_test  #saddle
python Twin-L2O.py --config rotatedsaddle_test  #rotated saddle
python Twin-L2O.py --config seesaw_test  #seesaw
```

### Seesaw Problem Using Twin-L2O+Curriculum Learning

#### Dataset

The distribution of $a$ and $b$ are stretched to the following setting, where the indices in the following table correspond with the dataset in the   `/dataset` folder.

| Uniform Distribution | 1            | 2            | 3            | 4            | 5            | 6          | 7          |
| -------------------- | ------------ | ------------ | ------------ | ------------ | ------------ | ---------- | ---------- |
| **a**                | **[0.9,1 ]** | **[0, 1]**   | **[0, 3.5]** | **[0, 5]**   | **[0.9, 1]** | **[0, 5]** | **[0, 5]** |
| **b**                | **[0.9, 1]** | **[0.9, 1]** | **[0.9, 1]** | **[0.9, 1]** | **[0, 1]**   | **[0, 1]** | **[0, 2]** |

#### Training

To perform training for Twin-L2O without curriculum Learning (Non-CL):

```shell
python Twin-L2O.py --config seesaw_range2_train 
# the same applies for other paramter ranges
```

To perform training for Twin-L2O with Curriculum Learning (CL): 

```shell
python Twin-L2O.py --config seesaw_range2_CL_train 
# the same applies for other paramter ranges
```

#### Testing

To perform testing for Twin-L2O without curriculum Learning (Non-CL):

```shell
python Twin-L2O.py --config seesaw_range2_test 
# the same applies for other paramter ranges
```

To perform testing for Twin-L2O with Curriculum Learning (CL):

```shell
python Twin-L2O.py --config seesaw_range2_CL_test 
# the same applies for other paramter ranges
```


