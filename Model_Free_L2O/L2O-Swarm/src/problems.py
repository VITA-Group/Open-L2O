
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tarfile
import sys
import numpy as np
from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin
import sonnet as snt
import tensorflow as tf
from dataloader import data_loader
from tensorflow.contrib.learn.python.learn.datasets import mnist as mnist_dataset
import pdb

_nn_initializers = {
    "w": tf.random_normal_initializer(mean=0, stddev=0.01),
    "b": tf.random_normal_initializer(mean=0, stddev=0.01),
}


def simple():
  """Simple problem: f(x) = x^2."""

  def build():
    """Builds loss graph."""
    x = tf.get_variable(
        "x",
        shape=[],
        dtype=tf.float32,
        initializer=tf.ones_initializer())
    return tf.square(x, name="x_squared")

  return build


def simple_multi_optimizer(num_dims=2):
  """Multidimensional simple problem."""

  def get_coordinate(i):
    return tf.get_variable("x_{}".format(i),
                           shape=[],
                           dtype=tf.float32,
                           initializer=tf.ones_initializer())

  def build():
    coordinates = [get_coordinate(i) for i in xrange(num_dims)]
    x = tf.concat([tf.expand_dims(c, 0) for c in coordinates], 0)
    return tf.reduce_sum(tf.square(x, name="x_squared"))

  return build


def quadratic(batch_size=128, num_dims=10, stddev=0.01, dtype=tf.float32):
  """Quadratic problem: f(x) = ||Wx - y||."""

  def build():
    """Builds loss graph."""
#    pdb.set_trace()
    
    # Trainable variable.
    x = tf.get_variable(
        "x",
        shape=[batch_size, num_dims],
        dtype=dtype,
        initializer=tf.random_normal_initializer(stddev=stddev))
#    print(batch_size)
#    print(x.get_shape())
    w = tf.get_variable("w",
                        shape=[batch_size, num_dims, num_dims],
                        dtype=dtype,
                        initializer=tf.random_uniform_initializer(),
                        trainable=False)
    print(w.get_shape())
    y = tf.get_variable("y",
                        shape=[batch_size, num_dims],
                        dtype=dtype,
                        initializer=tf.random_uniform_initializer(),
                        trainable=False)
    print(y.get_shape())
    product = tf.squeeze(tf.matmul(w, tf.expand_dims(x, -1)))
    
        
    return (tf.reduce_sum((product - y) ** 2, 1))

  return build


def square_cos(batch_size=128, num_dims=10,  stddev=0.01, dtype=tf.float32):
  def build():
    """Builds loss graph."""

    # Trainable variable.
    x = tf.get_variable(
        "x",
        shape=[batch_size, num_dims],
        dtype=dtype,
        initializer=tf.random_normal_initializer(stddev=stddev))

    # Non-trainable variables.
    w = tf.get_variable("w",
                        shape=[batch_size, num_dims, num_dims],
                        dtype=dtype,
                        initializer=tf.random_uniform_initializer(),
                        trainable=False)
    y = tf.get_variable("y",
                        shape=[batch_size, num_dims],
                        dtype=dtype,
                        initializer=tf.random_uniform_initializer(),
                        trainable=False)

    wcos = tf.get_variable("wcos",
                        shape=[batch_size, num_dims, num_dims],
                        dtype=dtype,
                        initializer=tf.random_uniform_initializer(),
                        trainable=False)

    product = tf.squeeze(tf.matmul(w, tf.expand_dims(x, -1)))
    product2 = tf.squeeze(tf.matmul(wcos, tf.expand_dims(10*tf.math.cos(2*3.1415926*x), -1)))
    product3 = tf.reduce_sum((product - y) ** 2, 1) - tf.reduce_sum(product2, 1) + 10*num_dims
   
    return product3
    return tf.reduce_mean(tf.reduce_sum((product - y) ** 2, 1)) - tf.reduce_mean(tf.reduce_sum(product2, 1)) + 10*num_dims

  return build



def protein_dock(batch_size=128, num_dims=12, stddev=0.5, dtype=tf.float32):
  scoor_init, sq, se, sr, sbasis, seval = data_loader()  
  batch_size=125
  num_dims=12
  natoms=100
  def build():
    """Builds loss graph."""
  #    pdb.set_trace()
   # scoor_init, sq, se, sr, sbasis, seval = data_loader()
    # Trainable variable.
    x = tf.get_variable(
        "x",
        shape=[batch_size, num_dims],
        dtype=dtype,
        initializer=tf.random_normal_initializer(stddev=stddev))

    coor_init = tf.get_variable("coor_init",
                        shape=[*scoor_init.shape],
                        dtype=dtype,
                        initializer=tf.constant_initializer(scoor_init, dtype = tf.float32),
                        trainable=False)

    q = tf.get_variable("q",
                        shape=[*sq.shape],
                        dtype=dtype,
                        initializer=tf.constant_initializer(sq, dtype = tf.float32),
                        trainable=False)
    e = tf.get_variable("e",
                        shape=[*se.shape],
                        dtype=dtype,
                        initializer=tf.constant_initializer(se, dtype = tf.float32),
                        trainable=False)

    r = tf.get_variable("r",
                        shape=[*sr.shape],
                        dtype=dtype,
                        initializer=tf.constant_initializer(sr, dtype = tf.float32),
                        trainable=False)

    basis = tf.get_variable("basis",
                        shape=[*sbasis.shape],
                        dtype=dtype,
                        initializer=tf.constant_initializer(sbasis, dtype = tf.float32),
                        trainable=False)

    eigval = tf.get_variable("eval",
                        shape=[*seval.shape],
                        dtype=dtype,
                        initializer=tf.constant_initializer(seval, dtype = tf.float32),
                        trainable=False)
    # print ('x',x.get_shape())
    # print ('ini',coor_init.get_shape())
    # print ('q',q.get_shape())
    # print ('e',e.get_shape())
    # print ('r',r.get_shape())
    # print ('basis',basis.get_shape())
    # print ('eig', eigval.get_shape())

    init = tf.reshape(coor_init, [batch_size, basis.get_shape()[2]])
    eigval = 1.0/tf.sqrt(eigval)

    #tf.print(eigval, output_channels)

    product = tf.squeeze(tf.matmul(tf.expand_dims(x*eigval, 1), basis))

    new_coor = tf.reshape(product, coor_init.get_shape()) + coor_init

    print ('product', product.get_shape())
    print ('new_coor', new_coor.get_shape())

    p2 = tf.reduce_sum(new_coor*new_coor, 2)
    p3 = tf.matmul(new_coor, tf.transpose(new_coor, perm=[0,2,1]))
    p2 = tf.expand_dims(p2, -1)
    pair_dis = tf.sqrt(p2 - 2*p3 + tf.transpose(p2, perm=[0, 2, 1]) + 0.01)


    print ('p2', p2.get_shape())
    print ('p3', p3.get_shape())
    print ('pair_dis', pair_dis.get_shape())

    
    c7_small = tf.math.less(pair_dis, 7)
    c7 = tf.math.greater(pair_dis, 7)
    c0 = tf.math.greater(pair_dis, 0.1)
    c9 = tf.math.less(pair_dis, 9)

    c7=tf.cast(c7, dtype)
    c0=tf.cast(c0, dtype)
    c9=tf.cast(c9, dtype)
    c7_small=tf.cast(c7_small, dtype)

    c79=c7*c9*c0
    c7_small=c7_small*c0

    pair_dis += tf.eye(natoms, num_columns=natoms, batch_shape=[batch_size])

    coeff = q/(4.*pair_dis) + tf.sqrt(e) * ( (r/pair_dis)**12 - (r/pair_dis)**6 )

    energy = tf.reduce_mean(tf.reduce_sum( c7_small* coeff*10 + 10*c79 * coeff * ( (9-pair_dis)**2 * (-12 + 2*pair_dis) / 8 ), 1), -1)- 7000

    return energy
    #tf.reduce_mean (tf.reduce_sum(pair_dis, 1))

  return build





def ensemble(problems, weights=None):
  """Ensemble of problems.

  Args:
    problems: List of problems. Each problem is specified by a dict containing
        the keys 'name' and 'options'.
    weights: Optional list of weights for each problem.

  Returns:
    Sum of (weighted) losses.

  Raises:
    ValueError: If weights has an incorrect length.
  """
  if weights and len(weights) != len(problems):
    raise ValueError("len(weights) != len(problems)")

  build_fns = [getattr(sys.modules[__name__], p["name"])(**p["options"])
               for p in problems]

  def build():
    loss = 0
    for i, build_fn in enumerate(build_fns):
      with tf.variable_scope("problem_{}".format(i)):
        loss_p = build_fn()
        if weights:
          loss_p *= weights[i]
        loss += loss_p
    return loss

  return build


def _xent_loss(output, labels):
  loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=output,
                                                        labels=labels)
  return tf.reduce_mean(loss)


def mnist(layers,  # pylint: disable=invalid-name
          activation="sigmoid",
          batch_size=128,
          mode="train"):
  """Mnist classification with a multi-layer perceptron."""

  if activation == "sigmoid":
    activation_op = tf.sigmoid
  elif activation == "relu":
    activation_op = tf.nn.relu
  else:
    raise ValueError("{} activation not supported".format(activation))

  # Data.
  data = mnist_dataset.load_mnist()
  data = getattr(data, mode)
  images = tf.constant(data.images, dtype=tf.float32, name="MNIST_images")
  images = tf.reshape(images, [-1, 28, 28, 1])
  labels = tf.constant(data.labels, dtype=tf.int64, name="MNIST_labels")

  # Network.
  mlp = snt.nets.MLP(list(layers) + [10],
                     activation=activation_op,
                     initializers=_nn_initializers)
  network = snt.Sequential([snt.BatchFlatten(), mlp])

  def build():
    indices = tf.random_uniform([batch_size], 0, data.num_examples, tf.int64)
    batch_images = tf.gather(images, indices)
    batch_labels = tf.gather(labels, indices)
    output = network(batch_images)
    return _xent_loss(output, batch_labels)

  return build


CIFAR10_URL = "http://www.cs.toronto.edu/~kriz"
CIFAR10_FILE = "cifar-10-binary.tar.gz"
CIFAR10_FOLDER = "cifar-10-batches-bin"


def _maybe_download_cifar10(path):
  """Download and extract the tarball from Alex's website."""
  if not os.path.exists(path):
    os.makedirs(path)
  filepath = os.path.join(path, CIFAR10_FILE)
  if not os.path.exists(filepath):
    print("Downloading CIFAR10 dataset to {}".format(filepath))
    url = os.path.join(CIFAR10_URL, CIFAR10_FILE)
    filepath, _ = urllib.request.urlretrieve(url, filepath)
    statinfo = os.stat(filepath)
    print("Successfully downloaded {} bytes".format(statinfo.st_size))
    tarfile.open(filepath, "r:gz").extractall(path)


def cifar10(path,  # pylint: disable=invalid-name
            conv_channels=None,
            linear_layers=None,
            batch_norm=True,
            batch_size=128,
            num_threads=4,
            min_queue_examples=1000,
            mode="train"):
  """Cifar10 classification with a convolutional network."""

  # Data.
  _maybe_download_cifar10(path)

  # Read images and labels from disk.
  if mode == "train":
    filenames = [os.path.join(path,
                              CIFAR10_FOLDER,
                              "data_batch_{}.bin".format(i))
                 for i in xrange(1, 6)]
  elif mode == "test":
    filenames = [os.path.join(path, "test_batch.bin")]
  else:
    raise ValueError("Mode {} not recognised".format(mode))

  depth = 3
  height = 32
  width = 32  
  label_bytes = 1
  image_bytes = depth * height * width
  record_bytes = label_bytes + image_bytes
  reader = tf.FixedLengthRecordReader(record_bytes=record_bytes)
  _, record = reader.read(tf.train.string_input_producer(filenames))
  record_bytes = tf.decode_raw(record, tf.uint8)

  label = tf.cast(tf.slice(record_bytes, [0], [label_bytes]), tf.int32)
  raw_image = tf.slice(record_bytes, [label_bytes], [image_bytes])
  image = tf.cast(tf.reshape(raw_image, [depth, height, width]), tf.float32)
  # height x width x depth.
  image = tf.transpose(image, [1, 2, 0])
  image = tf.div(image, 255)

  queue = tf.RandomShuffleQueue(capacity=min_queue_examples + 3 * batch_size,
                                min_after_dequeue=min_queue_examples,
                                dtypes=[tf.float32, tf.int32],
                                shapes=[image.get_shape(), label.get_shape()])
  enqueue_ops = [queue.enqueue([image, label]) for _ in xrange(num_threads)]
  tf.train.add_queue_runner(tf.train.QueueRunner(queue, enqueue_ops))

  # Network.
  def _conv_activation(x):  # pylint: disable=invalid-name
    return tf.nn.max_pool(tf.nn.relu(x),
                          ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1],
                          padding="SAME")

  conv = snt.nets.ConvNet2D(output_channels=conv_channels,
                            kernel_shapes=[5],
                            strides=[1],
                            paddings=[snt.SAME],
                            activation=_conv_activation,
                            activate_final=True,
                            initializers=_nn_initializers,
                            use_batch_norm=batch_norm)

  if batch_norm:
    linear_activation = lambda x: tf.nn.relu(snt.BatchNorm()(x))
  else:
    linear_activation = tf.nn.relu

  mlp = snt.nets.MLP(list(linear_layers) + [10],
                     activation=linear_activation,
                     initializers=_nn_initializers)
  network = snt.Sequential([conv, snt.BatchFlatten(), mlp])

  def build():
    image_batch, label_batch = queue.dequeue_many(batch_size)
    label_batch = tf.reshape(label_batch, [batch_size])

    output = network(image_batch)
    return _xent_loss(output, label_batch)

  return build
