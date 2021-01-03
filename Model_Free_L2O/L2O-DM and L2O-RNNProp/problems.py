# Copyright 2016 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License
# ==============================================================================
"""Learning 2 Learn problems."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tarfile
import sys

from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin
import sonnet as snt
import tensorflow as tf
import pdb
from tensorflow.contrib.learn.python.learn.datasets import mnist as mnist_dataset
from vgg16 import VGG16
import tensorflow_probability as tfp
tfd = tfp.distributions

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

    product = tf.squeeze(tf.matmul(w, tf.expand_dims(x, -1)))
    return tf.reduce_mean(tf.reduce_sum((product - y) ** 2, 1))

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


def mnist(layers,
          activation="sigmoid",
          batch_size=128,
          mode="train"):
  """Mnist classification with a multi-layer perceptron."""
  initializers = _nn_initializers

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
                     initializers=initializers)
  network = snt.Sequential([snt.BatchFlatten(), mlp])

  def build():
    indices = tf.random_uniform([batch_size], 0, data.num_examples, tf.int64)
    batch_images = tf.gather(images, indices)
    batch_labels = tf.gather(labels, indices)
    output = network(batch_images)
    return _xent_loss(output, batch_labels)

  return build


def mnist_conv(batch_norm=True,
               batch_size=128,
               mode="train"):

  # Data.
  data = mnist_dataset.load_mnist()
  data = getattr(data, mode)
  images = tf.constant(data.images, dtype=tf.float32, name="MNIST_images")
  images = tf.reshape(images, [-1, 28, 28, 1])
  labels = tf.constant(data.labels, dtype=tf.int64, name="MNIST_labels")

  def network(inputs, training=True):

      def _conv_activation(x):
          return tf.nn.max_pool(tf.nn.relu(x),
                                ksize=[1, 2, 2, 1],
                                strides=[1, 2, 2, 1],
                                padding="VALID")
      def conv_layer(inputs, strides, c_h, c_w, output_channels, padding, name):
          n_channels = int(inputs.get_shape()[-1])
          with tf.variable_scope(name) as scope:
              kernel1 = tf.get_variable('weights1',
                                        shape=[c_h, c_w, n_channels, output_channels],
                                        dtype=tf.float32,
                                        initializer=tf.random_normal_initializer(stddev=0.01)
                                        )
              
              biases1 = tf.get_variable('biases1', [output_channels], initializer=tf.constant_initializer(0.0))
          inputs = tf.nn.conv2d(inputs, kernel1, [1, strides, strides, 1], padding)
          inputs = tf.nn.bias_add(inputs, biases1)
          if batch_norm:
              inputs = tf.layers.batch_normalization(inputs, training=training)
          inputs = _conv_activation(inputs)
          return inputs

      inputs = conv_layer(inputs, 1, 3, 3, 16, "VALID", 'conv_layer1')
      inputs = conv_layer(inputs, 1, 5, 5, 32, "VALID", 'conv_layer2')
      inputs = tf.reshape(inputs, [batch_size, -1])
      fc_shape2 = int(inputs.get_shape()[1])
      weights = tf.get_variable("fc_weights",
                                shape=[fc_shape2, 10],
                                dtype=tf.float32,
                                initializer=tf.random_normal_initializer(stddev=0.01))
      bias = tf.get_variable("fc_bias",
                             shape=[10, ],
                             dtype=tf.float32,
                             initializer=tf.constant_initializer(0.0))
      return tf.nn.relu(tf.nn.bias_add(tf.matmul(inputs, weights), bias))

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


def cifar10(path,
            batch_norm=True,
            batch_size=128,
            num_threads=4,
            min_queue_examples=1000,
            mode="train"):
  """Cifar10 classification with a convolutional network."""

  # Data.
  _maybe_download_cifar10(path)
  if mode == "train":
    filenames = [os.path.join(path, CIFAR10_FOLDER, "data_batch_{}.bin".format(i)) for i in xrange(1, 6)]
  elif mode == "test":
    filenames = [os.path.join(path, CIFAR10_FOLDER, "test_batch.bin")]
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
  image = tf.math.divide(image, 255)

  queue = tf.RandomShuffleQueue(capacity=min_queue_examples + 3 * batch_size,
                                min_after_dequeue=min_queue_examples,
                                dtypes=[tf.float32, tf.int32],
                                shapes=[image.get_shape(), label.get_shape()])
  enqueue_ops = [queue.enqueue([image, label]) for _ in xrange(num_threads)]
  tf.train.add_queue_runner(tf.train.QueueRunner(queue, enqueue_ops))

  def network(inputs, training=True):

      def _conv_activation(x):
          return tf.nn.max_pool(tf.nn.relu(x),
                                ksize=[1, 2, 2, 1],
                                strides=[1, 2, 2, 1],
                                padding="VALID")
    
      def conv_layer(inputs, strides, c_h, c_w, output_channels, padding, name):
          n_channels = int(inputs.get_shape()[-1])
          with tf.variable_scope(name) as scope:
              kernel1 = tf.get_variable('weights1',
                                        shape=[c_h, c_w, n_channels, output_channels],
                                        dtype=tf.float32,
                                        initializer=tf.random_normal_initializer(stddev=0.01)
                                        )
            
              biases1 = tf.get_variable('biases1', [output_channels], initializer=tf.constant_initializer(0.0))
          inputs = tf.nn.conv2d(inputs, kernel1, [1, strides, strides, 1], padding)
          inputs = tf.nn.bias_add(inputs, biases1)
          if batch_norm:
              inputs = tf.layers.batch_normalization(inputs, training=training)
          inputs = _conv_activation(inputs)
          return inputs

      inputs = conv_layer(inputs, 2, 3, 3, 16, "VALID", 'conv_layer1')
      inputs = conv_layer(inputs, 2, 5, 5, 32, "VALID", 'conv_layer2')
      inputs = tf.reshape(inputs, [batch_size, -1])
      fc_shape2 = int(inputs.get_shape()[1])
      weights = tf.get_variable("fc_weights",
                                shape=[fc_shape2, 10],
                                dtype=tf.float32,
                                initializer=tf.random_normal_initializer(stddev=0.01))
      bias = tf.get_variable("fc_bias",
                             shape=[10, ],
                             dtype=tf.float32,
                             initializer=tf.constant_initializer(0.0))

      return tf.nn.relu(tf.nn.bias_add(tf.matmul(inputs, weights), bias))

  
  def build():
    image_batch, label_batch = queue.dequeue_many(batch_size)
    label_batch = tf.reshape(label_batch, [batch_size])
    output = network(image_batch)

    return _xent_loss(output, label_batch)

  return build


def LeNet(path,
          conv_channels=None,
          linear_layers=None,
          batch_norm=True,
          batch_size=128,
          num_threads=4,
          min_queue_examples=1000,
          mode="train"):

    # Data.
    _maybe_download_cifar10(path)

    # Read images and labels from disk.
    if mode == "train":
        filenames = [os.path.join(path, CIFAR10_FOLDER, "data_batch_{}.bin".format(i)) for i in xrange(1, 6)]
    elif mode == "test":
        filenames = [os.path.join(path, CIFAR10_FOLDER, "test_batch.bin")]
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
        return tf.nn.max_pool(tf.sigmoid(x),
                              ksize=[1, 2, 2, 1],
                              strides=[1, 2, 2, 1],
                              padding="VALID")

    conv = snt.nets.ConvNet2D(output_channels=conv_channels,
                              kernel_shapes=[5],
                              strides=[1],
                              paddings=[snt.VALID],
                              activation=_conv_activation,
                              activate_final=True,
                              initializers=_nn_initializers,
                              use_batch_norm=batch_norm)

    if batch_norm:
        linear_activation = lambda x: tf.sigmoid(snt.BatchNorm()(x, is_training=True))
    else:
        linear_activation = tf.sigmoid

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


def NAS(path,
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
        filenames = [os.path.join(path, CIFAR10_FOLDER, "data_batch_{}.bin".format(i)) for i in xrange(1, 6)]
    elif mode == "test":
        filenames = [os.path.join(path, CIFAR10_FOLDER, "test_batch.bin")]
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

    # Network
    def network(inputs, training=True):
        def conv_layer(inputs, strides, c_h, c_w, output_channels, padding, name):
            n_channels = int(inputs.get_shape()[-1])
            with tf.variable_scope(name) as scope:
                kernel1 = tf.get_variable('weights1',
                                          shape=[c_h, c_w, n_channels, output_channels],
                                          dtype=tf.float32,
                                          initializer=tf.random_normal_initializer(stddev=0.01)
                                          )

                biases1 = tf.get_variable('biases1', [output_channels], initializer=tf.constant_initializer(0.0))
            inputs = tf.nn.conv2d(inputs, kernel1, [1, strides, strides, 1], padding)
            inputs = tf.nn.bias_add(inputs, biases1)
            if batch_norm:
                inputs = tf.layers.batch_normalization(inputs, training=training)
            inputs = tf.nn.relu(inputs)
            return inputs

        def _pooling(x):
            return tf.nn.avg_pool(x,
                                  ksize=[1, 3, 3, 1],
                                  strides=[1, 1, 1, 1],
                                  padding="SAME")

        node0 = conv_layer(inputs, 1, 3, 3, 16, "SAME", 'node0')
        node0_onto_node2 = conv_layer(node0, 1, 3, 3, 16, "SAME", 'node0_onto_node2')
        node1 = conv_layer(node0, 1, 3, 3, 16, "SAME", 'node1')
        node1_onto_node3 = conv_layer(node1, 1, 3, 3, 16, "SAME", 'node1_onto_node3')
        node2 = _pooling(node1) + node0_onto_node2
        node3 = node2 + node1_onto_node3 + node0
        node_final = tf.reduce_mean(tf.reshape(node3, [batch_size, -1, 16]), axis=1)

        fc_shape2 = int(node_final.get_shape()[1])
        weights = tf.get_variable("fc_weights",
                                  shape=[fc_shape2, 10],
                                  dtype=tf.float32,
                                  initializer=tf.random_normal_initializer(stddev=0.01))
        bias = tf.get_variable("fc_bias",
                               shape=[10, ],
                               dtype=tf.float32,
                               initializer=tf.constant_initializer(0.0))
        return tf.nn.relu(tf.nn.bias_add(tf.matmul(node_final, weights), bias))

    def build():
        image_batch, label_batch = queue.dequeue_many(batch_size)
        label_batch = tf.reshape(label_batch, [batch_size])

        output = network(image_batch)
        return _xent_loss(output, label_batch)

    return build


def vgg16_cifar10(path,  # pylint: disable=invalid-name
            batch_norm=False,
            batch_size=128,
            num_threads=4,
            min_queue_examples=1000,
            mode="train"):
    """Cifar10 classification with a convolutional network."""
    
    # Data.
    _maybe_download_cifar10(path)
    # pdb.set_trace()
    # Read images and labels from disk.
    if mode == "train":
        filenames = [os.path.join(path,
                                  CIFAR10_FOLDER,
                                  "data_batch_{}.bin".format(i))
                     for i in xrange(1, 6)]
        is_training = True
    elif mode == "test":
        filenames = [os.path.join(path, CIFAR10_FOLDER, "test_batch.bin")]
        is_training = False
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
    image = tf.math.divide(image, 255)

    queue = tf.RandomShuffleQueue(capacity=min_queue_examples + 3 * batch_size,
                                  min_after_dequeue=min_queue_examples,
                                  dtypes=[tf.float32, tf.int32],
                                  shapes=[image.get_shape(), label.get_shape()])
    enqueue_ops = [queue.enqueue([image, label]) for _ in xrange(num_threads)]
    tf.train.add_queue_runner(tf.train.QueueRunner(queue, enqueue_ops))

    vgg = VGG16(0.5, 10)
    def build():
        image_batch, label_batch = queue.dequeue_many(batch_size)
        label_batch = tf.reshape(label_batch, [batch_size])
        # pdb.set_trace()
        output = vgg._build_model(image_batch)
        # print(output.shape)
        return _xent_loss(output, label_batch)
    
    return build



# ---------------------------------------
# Custom Confocal Microscopy Problems
# ---------------------------------------
def confocal_microscopy_3d(batch_size=128, num_points=5, ROI=[28, 28, 28], stddev=0.01, dtype=tf.float32, inference=False):
  if inference:
    def build():
      """Builds loss graph."""
      # Trainable variable.
      I_var = []
      x_var = []
      y_var = []
      z_var = []
      sigmaxy_var = []
      sigmaz_var = []
      
      for i in range(num_points):
        I_var.append(tf.get_variable(
          "I_var_%d"%i,
          shape=[batch_size, 1],
          dtype=dtype,
          initializer=tf.random_uniform_initializer()))
        
        x_var.append(tf.get_variable(
          "x_var_%d"%i,
          shape=[batch_size, 1],
          dtype=dtype,
          initializer=tf.random_uniform_initializer()))

        y_var.append(tf.get_variable(
          "y_var_%d"%i,
          shape=[batch_size, 1],
          dtype=dtype,
          initializer=tf.random_uniform_initializer()))

        z_var.append(tf.get_variable(
          "z_var_%d"%i,
          shape=[batch_size, 1],
          dtype=dtype,
          initializer=tf.random_uniform_initializer()))

        sigmaxy_var.append(tf.get_variable(
          "sigmaxy_var_%d"%i,
          shape=[batch_size, 1],
          dtype=dtype,
          initializer=tf.random_uniform_initializer()))

        sigmaz_var.append(tf.get_variable(
          "sigmaz_var_%d"%i,
          shape=[batch_size, 1],
          dtype=dtype,
          initializer=tf.random_uniform_initializer()))
      
      # predict
      def point_spread_function_3d(theta):
        '''
        Microscopic fluorescence point spread function
        Args:
          theta : Variables in the point spread function, [I0, x0, y0, z0, sigmaxy, sigmaz]
        Returns:
          I : fluorescence image, the size is [batch_size, size_x * size_y * sizez]
        '''
        priors_list = [tfd.Uniform(low=0.5, high=2.0), 
                  tfd.Uniform(low=0.5, high=ROI[0]-1), 
                  tfd.Uniform(low=0.5, high=ROI[1]-1), 
                  tfd.Uniform(low=0.5, high=ROI[2]-1), 
                  tfd.Uniform(low=2, high=4), 
                  tfd.Uniform(low=2, high=4)]
        xs = tf.linspace(0.0, float(ROI[0]-1), ROI[0])
        ys = tf.linspace(0.0, float(ROI[1]-1), ROI[1])
        zs = tf.linspace(0.0, float(ROI[2]-1), ROI[2])
        X, Y, Z = tf.meshgrid(xs, ys, zs)
        
        I0 = priors_list[0].quantile(tf.reshape(theta[0], [theta[0].shape[0], 1]))
        x0 = priors_list[1].quantile(tf.reshape(theta[1], [theta[1].shape[0], 1]))
        y0 = priors_list[2].quantile(tf.reshape(theta[2], [theta[2].shape[0], 1]))
        z0 = priors_list[3].quantile(tf.reshape(theta[3], [theta[3].shape[0], 1]))
        sigmaxy = priors_list[4].quantile(tf.reshape(theta[4], [theta[4].shape[0], 1]))
        sigmaz = priors_list[5].quantile(tf.reshape(theta[5], [theta[5].shape[0], 1]))

        xk = tf.reshape(X, [1, -1])
        yk = tf.reshape(Y, [1, -1])
        zk = tf.reshape(Z, [1, -1])

        I = I0 * ((-tf.math.erf((-0.5 - x0 + xk)/(tf.math.sqrt(2.0)*sigmaxy)) + tf.math.erf((0.5 - x0 + xk)/(tf.math.sqrt(2.0)*sigmaxy))) \
              * (-tf.math.erf((-0.5 - y0 + yk)/(tf.math.sqrt(2.0)*sigmaxy)) + tf.math.erf((0.5 - y0 + yk)/(tf.math.sqrt(2.0)*sigmaxy))) \
              * (-tf.math.erf((-0.5 - z0 + zk)/(tf.math.sqrt(2.0)*sigmaz)) + tf.math.erf((0.5 - z0 + zk)/(tf.math.sqrt(2.0)*sigmaz))))/8.0
        return I
    
      y_pred = tf.add_n([
        point_spread_function_3d([I_var[i], x_var[i], y_var[i], z_var[i], sigmaxy_var[i], sigmaz_var[i]])
        for i in range(num_points)])
      
      bg_var = tf.get_variable(
          "bg_var",
          shape=[batch_size, 1],
          dtype=dtype,  
          initializer=tf.random_normal_initializer(stddev=stddev))
      
      img_placeholder = tf.placeholder(dtype, shape=(batch_size, ROI[0]*ROI[1]*ROI[2]))
      obj = tf.reduce_mean(tf.math.reduce_sum((y_pred + bg_var - tf.math.l2_normalize(img_placeholder, axis=1)) ** 2, axis=1))
      return obj
  else:
    def build():
      """Builds loss graph."""
      # Trainable variable.
      I_var = []
      x_var = []
      y_var = []
      z_var = []
      sigmaxy_var = []
      sigmaz_var = []
      
      for i in range(num_points):
        I_var.append(tf.get_variable(
          "I_var_%d"%i,
          shape=[batch_size, 1],
          dtype=dtype,
          initializer=tf.random_uniform_initializer()))
        
        x_var.append(tf.get_variable(
          "x_var_%d"%i,
          shape=[batch_size, 1],
          dtype=dtype,
          initializer=tf.random_uniform_initializer()))

        y_var.append(tf.get_variable(
          "y_var_%d"%i,
          shape=[batch_size, 1],
          dtype=dtype,
          initializer=tf.random_uniform_initializer()))

        z_var.append(tf.get_variable(
          "z_var_%d"%i,
          shape=[batch_size, 1],
          dtype=dtype,
          initializer=tf.random_uniform_initializer()))

        sigmaxy_var.append(tf.get_variable(
          "sigmaxy_var_%d"%i,
          shape=[batch_size, 1],
          dtype=dtype,
          initializer=tf.random_uniform_initializer()))

        sigmaz_var.append(tf.get_variable(
          "sigmaz_var_%d"%i,
          shape=[batch_size, 1],
          dtype=dtype,
          initializer=tf.random_uniform_initializer()))

      # Non-trainable variables.
      I_sim = []
      x_sim  = []
      y_sim  = []
      z_sim  = []
      sigmaxy_sim = []
      sigmaz_sim = []

      for i in range(num_points):
        I_sim.append(tf.get_variable(
          "I_sim_%d"%i,
          shape=[batch_size, 1],
          dtype=dtype,
          initializer=tf.random_uniform_initializer(),
          trainable=False))
        
        x_sim.append(tf.get_variable(
          "x_sim_%d"%i,
          shape=[batch_size, 1],
          dtype=dtype,
          initializer=tf.random_uniform_initializer(),
          trainable=False))

        y_sim.append(tf.get_variable(
          "y_sim%d"%i,
          shape=[batch_size, 1],
          dtype=dtype,
          initializer=tf.random_uniform_initializer(),
          trainable=False))

        z_sim.append(tf.get_variable(
          "z_sim_%d"%i,
          shape=[batch_size, 1],
          dtype=dtype,
          initializer=tf.random_uniform_initializer(),
          trainable=False))

        sigmaxy_sim.append(tf.get_variable(
          "sigmaxy_sim_%d"%i,
          shape=[batch_size, 1],
          dtype=dtype,
          initializer=tf.random_uniform_initializer(),
          trainable=False))

        sigmaz_sim.append(tf.get_variable(
          "sigmaz_sim_%d"%i,
          shape=[batch_size, 1],
          dtype=dtype,
          initializer=tf.random_uniform_initializer(),
          trainable=False))
        
      # predict
      def point_spread_function_3d(theta):
        '''
        Microscopic fluorescence point spread function
        Args:
          theta : Variables in the point spread function, [I0, x0, y0, z0, sigmaxy, sigmaz]
        Returns:
          I : fluorescence image, the size is [batch_size, size_x * size_y * sizez]
        '''
        priors_list = [tfd.Uniform(low=0.5, high=2.0), 
                  tfd.Uniform(low=0.5, high=ROI[0]-1), 
                  tfd.Uniform(low=0.5, high=ROI[1]-1), 
                  tfd.Uniform(low=0.5, high=ROI[2]-1), 
                  tfd.Uniform(low=2, high=4), 
                  tfd.Uniform(low=2, high=4)]
        xs = tf.linspace(0.0, float(ROI[0]-1), ROI[0])
        ys = tf.linspace(0.0, float(ROI[1]-1), ROI[1])
        zs = tf.linspace(0.0, float(ROI[2]-1), ROI[2])
        X, Y, Z = tf.meshgrid(xs, ys, zs)
        
        I0 = priors_list[0].quantile(tf.reshape(theta[0], [theta[0].shape[0], 1]))
        x0 = priors_list[1].quantile(tf.reshape(theta[1], [theta[1].shape[0], 1]))
        y0 = priors_list[2].quantile(tf.reshape(theta[2], [theta[2].shape[0], 1]))
        z0 = priors_list[3].quantile(tf.reshape(theta[3], [theta[3].shape[0], 1]))
        sigmaxy = priors_list[4].quantile(tf.reshape(theta[4], [theta[4].shape[0], 1]))
        sigmaz = priors_list[5].quantile(tf.reshape(theta[5], [theta[5].shape[0], 1]))

        xk = tf.reshape(X, [1, -1])
        yk = tf.reshape(Y, [1, -1])
        zk = tf.reshape(Z, [1, -1])

        I = I0 * ((-tf.math.erf((-0.5 - x0 + xk)/(tf.math.sqrt(2.0)*sigmaxy)) + tf.math.erf((0.5 - x0 + xk)/(tf.math.sqrt(2.0)*sigmaxy))) \
              * (-tf.math.erf((-0.5 - y0 + yk)/(tf.math.sqrt(2.0)*sigmaxy)) + tf.math.erf((0.5 - y0 + yk)/(tf.math.sqrt(2.0)*sigmaxy))) \
              * (-tf.math.erf((-0.5 - z0 + zk)/(tf.math.sqrt(2.0)*sigmaz)) + tf.math.erf((0.5 - z0 + zk)/(tf.math.sqrt(2.0)*sigmaz))))/8.0
        return I
      
      y_pred = tf.add_n([
        point_spread_function_3d([I_var[i], x_var[i], y_var[i], z_var[i], sigmaxy_var[i], sigmaz_var[i]])
        for i in range(num_points)])
      
      y_sim = tf.add_n([
        point_spread_function_3d([I_sim[i], x_sim[i], y_sim[i], z_sim[i], sigmaxy_sim[i], sigmaz_sim[i]])
        for i in range(num_points)])
      
      bg_var = tf.get_variable(
          "bg_var",
          shape=[batch_size, 1],
          dtype=dtype,
          initializer=tf.random_normal_initializer(stddev=stddev))

      bg_sim = tf.get_variable(
          "bg_sim",
          shape=[batch_size, 1],
          dtype=dtype,
          initializer=tf.random_uniform_initializer(),
          trainable=False)
      obj = tf.reduce_mean(tf.math.reduce_sum((y_pred + bg_var - tf.math.l2_normalize(y_sim + bg_sim, axis=1)) ** 2, axis=1))
      return obj
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
   
    return tf.reduce_mean(product3)
    # return tf.reduce_mean(tf.reduce_mean(tf.reduce_sum((product - y) ** 2, 1)) - tf.reduce_mean(tf.reduce_sum(product2, 1)) + 10*num_dims)

  return build