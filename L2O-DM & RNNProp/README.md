## Training Stronger Baselines for Learning to Optimize

**Detailed instructions for experiments on L2O-DM and RNNProp**

### Experiments on L2O-DM

#### Environment

- Python 3.7

- TensorFlow 1.14.0

- sonnet 1.11 (pip install dm-sonnet=1.11)

#### Train L2O-DM

```shell
python train_dm.py --save_path=${OUTPUT_DIR} --problem=mnist --if_cl=False --if_mt=False —num_epochs=10000 --num_steps=100
```

--save_path: the directory to save models

--problem: the optimizee problem for training

--if_cl: whether use curriculum learning or not

--if_mt: whether use multi-task technique or not

--num_epochs: the number of epochs for training

--num_steps: the number of optimization steps in each epoch for training the learned optimizer

#### Run L2O-DM-AUG

```shell
python train_dm.py --save_path=${OUTPUT_DIR} --problem=mnist --if_cl=False --if_mt=False --num_epochs=5000 --num_steps=500
```

The experiment with more optimization steps, e.g., num_steps=500.


#### Run L2O-DM-CL

```shell
python train_dm.py --save_path=${OUTPUT_DIR} --problem=mnist --if_cl=True --if_mt=False
```

The experiment with curriculum learning technique alone. 


#### Run L2O-DM-CL-IL

```shell
python train_dm.py --save_path=${OUTPUT_DIR} --problem=mnist --if_cl=True --if_mt=True
```

The experiment with both curriculum learning and multi-task learning techniques. 


#### Evaluate with L2O-DM:
```shell
python evaluate_dm.py --path=${MODEL_FILE} --num_steps=10000 --problem=mnist_relu --output_path=${OUTPUT_DIR} --seed=2
```

--path: the file path of the trained model

--num_steps: the optimization steps for testing the learned optimizer

--problem: the optimizee problems for testing. Options as described in our paper: mnist, mnist_relu, mnist_deeper, mnist_conv, cifar_conv, nas, lenet.

--output_path: the directory to output the evaluation results.

--seed: random seed.




### Experiments on RNNProp

```shell
python train_rnnprop.py --save_path=${OUTPUT_DIR} --problem=mnist --if_cl=False --if_mt=False --num_epochs=100 --num_steps=100
```

The experiment with the RNNProp alone, smaller number of training epochs is needed here.



### Experiments on RNNProp-CL-IL

```shell
python train_rnnprop.py --save_path=${OUTPUT_DIR} --problem=mnist --if_cl=True --if_mt=True
```

The experiment with RNNProp using both curriculum learning and multi-task learning techniques.

#### Evaluate with RNNProp:
```shell
python evaluate_rnnprop.py --path=${MODEL_FILE} --num_epochs=1 --num_steps=10000 --problem=mnist_relu --output_path=${OUTPUT_DIR} --seed=2
```

The same as evaluation with L2O-DM.

#### Trained models:

L2O-DM: trained_models/dm/cw.l2l-0

L2O-DM-CL: trained_models/dm_cl/cw.l2l-0

L2O-DM-IL: trained_models/dm_il/cw.l2l-0

L2O-DM-CL-IL: trained_models_cl_il/dm/cw.l2l-0

RNNProp: trained_models_cl_il/rnnprop/rp.l2l-0

RNNProp-CL-IL: trained_models_cl_il/rnnprop_cl_il/rp.l2l-0
