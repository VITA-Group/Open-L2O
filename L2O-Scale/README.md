## Training Stronger Baselines for Learning to Optimize

**Detailed instructions for experiments on L2O-DM and RNNProp**

Code for [Learned Optimizers that Scale and Generalize](https://arxiv.org/abs/1703.04813).

### Experiments on L2O-Scale

#### Environment

* Bazel ([install](https://bazel.build/versions/master/docs/install.html))
* TensorFlow >= v1.3

### Code Overview

In the top-level directory, ```metaopt.py``` contains the code to train and test a learned optimizer. ```metarun.py``` packages the actual training procedure into a single file, defining and exposing many flags to tune the procedure, from selecting the optimizer type and problem set to more fine-grained hyperparameter settings.

There is no testing binary; testing can be done ad-hoc via ```metaopt.test_optimizer``` by passing an optimizer object and a directory with a checkpoint.

The ```optimizer``` directory contains a base ```trainable_optimizer.py``` class and a number of extensions, including the ```hierarchical_rnn``` optimizer used in the paper, a ```coordinatewise_rnn``` optimizer that more closely matches previous work, and a number of simpler optimizers to demonstrate the basic mechanics of
a learnable optimizer.

The ```problems``` directory contains the code to build the problems that were used in the meta-training set.

The ```metarun.py``` is used to meta-train a learnable optimizer.

### Command-Line Flags

The flags most relevant to meta-training are defined in ```metarun.py```. The default values will meta-train a HierarchicalRNN optimizer with the hyperparameter settings used in the paper.

### Using a Learned Optimizer as a Black Box

The ```trainable_optimizer``` inherits from ```tf.train.Optimizer```, so a properly instantiated version can be used to train any model in any APIs that accept this class. There are just 2 caveats:

1. If using the Hierarchical RNN optimizer, the apply_gradients return type must be changed (see comments inline for what exactly must be removed)

2. Care must be taken to restore the variables from the optimizer without overriding them. Optimizer variables should be loaded manually using a pretrained checkpoint
   and a ```tf.train.Saver``` with only the optimizer variables. Then, when constructing the session, ensure that any automatic variable initialization does not
   re-initialize the loaded optimizer variables.

### Experiments on L2O-Scale

#### Training

```shell
cd l2o-scale-regularize-train

python metarun.py --train_dir=hess_cl_mt --regularize_time=none --alpha=1e-4 --reg_optimizer=True --reg_option=hessian-esd --include_mnist_mlp_problems --num_problems=1 --num_meta_iterations=100 --fix_unroll=True --fix_unroll_length=20 --evaluation_period=1 --evaluation_epochs=5 --use_second_derivatives=False --if_cl=True --if_mt=True --mt_ratio=0.1 --mt_k=1
```

#### Evaluation

```shell
cd learned_optimizer

python metatest.py --train_dir=../l2o-scale-regularize-train/hess_cl_mt --save_dir=hess_cl_mt_eval --include_mnist_mlp_relu_problems --model_name=mnist-relu --restore_model_name=model.ckpt-0 --num_testing_itrs=10000
```

- You can modify --model_name and --include_mnist_conv_problem --include_cifar10_conv_problems --include_mnist_mlp_deeper_problems --include_mnist_mlp_problems for different problems

- You can change metatest.py line 473 for different random seeds
