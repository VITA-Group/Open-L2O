## L2O-DM, L2O-RNNProp and their Enhanced Versions

### Experiments on L2O-DM

#### Environment

- Python 3.6 

- TensorFlow 1.14.0 (pip install tensorflow==1.14.0 or pip install tensorflow-gpu==1.14.0)

- sonnet 1.11 (pip install dm-sonnet=1.11)

- tensorflow-probability==0.7.0

#### Train L2O-DM

```shell
python train_dm.py --save_path=${OUTPUT_DIR} --problem=mnist --if_cl=False --if_mt=False --num_epochs=10000 --num_steps=100
```

#### Train L2O-DM (enhanced)

```shell
python train_dm.py --save_path=${OUTPUT_DIR} --problem=mnist --if_cl=True --if_mt=True
```

Remark. The experiment with both curriculum learning and imitation learning techniques. 

**Detailed Instructions**

- --save_path: the directory to save models

- --problem: the optimizee problem for training

- --if_cl: whether use curriculum learning or not

- --if_mt: whether use imitation technique or not

- --num_epochs: the number of epochs for training

- --num_steps: the number of optimization steps in each epoch for training the learned optimizer

More details are referred to [README](https://github.com/VITA-Group/L2O-Training-Techniques/blob/master/L2O-DM%20%26%20RNNProp/README.md).


#### Evaluate with L2O-DM
```shell
python evaluate_dm.py --path=${MODEL_FILE} --num_steps=10000 --problem=mnist_relu --output_path=${OUTPUT_DIR} --seed=2
```

**Detailed Instructions**

- --path: the file path of the trained model

- --num_steps: the optimization steps for testing the learned optimizer

- --problem: the optimizee problems for testing. Options as described in our paper: mnist, mnist_relu, mnist_deeper, mnist_conv, cifar_conv, nas, lenet.

- --output_path: the directory to output the evaluation results.

- --seed: random seed.



### Experiments on L2O-RNNProp

#### Train L2O-RNNProp

```shell
python train_rnnprop.py --save_path=${OUTPUT_DIR} --problem=mnist --if_cl=False --if_mt=False --num_epochs=100 --num_steps=100
```

#### Train L2O-RNNProp (enhanced)

```shell
python train_rnnprop.py --save_path=${OUTPUT_DIR} --problem=mnist --if_cl=True --if_mt=True
```

Remark. The experiment with both curriculum learning and imitation learning techniques. 

#### Evaluate with L2O-RNNProp
```shell
python evaluate_rnnprop.py --path=${MODEL_FILE} --num_epochs=1 --num_steps=10000 --problem=mnist_relu --output_path=${OUTPUT_DIR} --seed=2
```



### Trained Optimizer:

L2O-DM: trained_models/dm/cw.l2l-0

L2O-DM (enhanced): trained_models_cl_il/dm/cw.l2l-0

L2O-RNNProp: trained_models_cl_il/rnnprop/rp.l2l-0

L2O-RNNProp (enhanced): trained_models_cl_il/rnnprop_cl_il/rp.l2l-0
