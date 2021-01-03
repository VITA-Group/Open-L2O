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
# limitations under the License.
# ==============================================================================
"""Learning 2 Learn utils."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from timeit import default_timer as timer

import numpy as np
from six.moves import xrange

import problems
import random


def run_epoch(sess, cost_op, ops, reset, num_unrolls,
              scale=None, rd_scale=False, rd_scale_bound=3.0, assign_func=None, var_x=None,
              step=None, unroll_len=None,
              task_i=-1, data=None, label_pl=None, input_pl=None):
  """Runs one optimization epoch."""
  start = timer()
  sess.run(reset)
  if task_i == -1:
      if rd_scale:
        assert scale is not None
        r_scale = []
        for k in scale:
          r_scale.append(np.exp(np.random.uniform(-rd_scale_bound, rd_scale_bound,
                            size=k.shape)))
        assert var_x is not None
        k_value_list = []
        for k_id in range(len(var_x)):
          k_value = sess.run(var_x[k_id])
          k_value = k_value / r_scale[k_id]
          k_value_list.append(k_value)
        assert assign_func is not None
        assign_func(k_value_list)
        feed_rs = {p: v for p, v in zip(scale, r_scale)}
      else:
        feed_rs = {}
      feed_dict = feed_rs
      for i in xrange(num_unrolls):
        if step is not None:
            feed_dict[step] = i*unroll_len+1
        cost = sess.run([cost_op] + ops, feed_dict=feed_dict)[0]
  else:
      assert data is not None
      assert input_pl is not None
      assert label_pl is not None
      feed_dict = {}
      for ri in xrange(num_unrolls):
          for pl, dat in zip(label_pl, data["labels"][ri]):
              feed_dict[pl] = dat
          for pl, dat in zip(input_pl, data["inputs"][ri]):
              feed_dict[pl] = dat
          if step is not None:
              feed_dict[step] = ri * unroll_len + 1
          cost = sess.run([cost_op] + ops, feed_dict=feed_dict)[0]
  return timer() - start, cost


def run_eval_epoch(sess, cost_op, ops, num_unrolls, step=None, unroll_len=None):
  """Runs one optimization epoch."""
  start = timer()
  # sess.run(reset)
  total_cost = []
  feed_dict = {}
  for i in xrange(num_unrolls):
    if step is not None:
        feed_dict[step] = i * unroll_len + 1
    cost = sess.run([cost_op] + ops, feed_dict=feed_dict)[0]
    total_cost.append(cost)
  return timer() - start, total_cost


def print_stats(header, total_error, total_time, n):
  """Prints experiment statistics."""
  print(header)
  print("Log Mean Final Error: {:.2f}".format(np.log10(total_error / n)))
  print("Mean epoch time: {:.2f} s".format(total_time / n))


def get_default_net_config(path):
  return {
      "net": "CoordinateWiseDeepLSTM",
      "net_options": {
          "layers": (20, 20),
          "preprocess_name": "LogAndSign",
          "preprocess_options": {"k": 5},
          "scale": 0.01,
      },
      "net_path": path
  }


def get_config(problem_name, path=None, mode=None, num_hidden_layer=None, net_name=None):
  """Returns problem configuration."""
  if problem_name == "simple":
    problem = problems.simple()
    net_config = {"cw": {
        "net": "CoordinateWiseDeepLSTM",
        "net_options": {"layers": (), "initializer": "zeros"},
        "net_path": path
    }}
    net_assignments = None
  elif problem_name == "simple-multi":
    problem = problems.simple_multi_optimizer()
    net_config = {
        "cw": {
            "net": "CoordinateWiseDeepLSTM",
            "net_options": {"layers": (), "initializer": "zeros"},
            "net_path": path
        },
        "adam": {
            "net": "Adam",
            "net_options": {"learning_rate": 0.01}
        }
    }
    net_assignments = [("cw", ["x_0"]), ("adam", ["x_1"])]
  elif problem_name == "quadratic":
    problem = problems.quadratic(batch_size=128, num_dims=10)
    net_config = {"cw": {
        "net": "CoordinateWiseDeepLSTM",
        "net_options": {"layers": (20, 20)},
        "net_path": path
    }}
    net_assignments = None
  ### our tests
  elif problem_name == "mnist":
    if mode is None:
        mode = "train" if path is None else "test"
    problem = problems.mnist(layers=(20,), activation="sigmoid", mode=mode)
    net_config = {"cw": get_default_net_config(path)}
    net_assignments = None
  elif problem_name == "mnist_relu":
    if mode is None:
        mode = "train" if path is None else "test"
    problem = problems.mnist(layers=(20,), activation="relu", mode=mode)
    net_config = {"cw": get_default_net_config(path)}
    net_assignments = None
  elif problem_name == "mnist_deeper":
    if mode is None:
        mode = "train" if path is None else "test"
    num_hidden_layer = 2
    problem = problems.mnist(layers=(20,) * num_hidden_layer, activation="sigmoid", mode=mode)
    net_config = {"cw": get_default_net_config(path)}
    net_assignments = None
  elif problem_name == "mnist_conv":
    if mode is None:
        mode = "train" if path is None else "test"
    problem = problems.mnist_conv(mode=mode, batch_norm=True)
    net_config = {"cw": get_default_net_config(path)}
    net_assignments = None
  elif problem_name == "cifar_conv":
    if mode is None:
        mode = "train" if path is None else "test"
    problem = problems.cifar10("cifar10", mode=mode)
    net_config = {"cw": get_default_net_config(path)}
    net_assignments = None
  elif problem_name == "lenet":
    if mode is None:
        mode = "train" if path is None else "test"
    problem = problems.LeNet("cifar10",
                             conv_channels=(6, 16),
                             linear_layers=(120, 84),
                             mode=mode)
    net_config = {"cw": get_default_net_config(path)}
    net_assignments = None
  elif problem_name == "nas":
    if mode is None:
        mode = "train" if path is None else "test"
    problem = problems.NAS("cifar10", mode=mode)
    net_config = {"cw": get_default_net_config(path)}
    net_assignments = None
  ###
  elif problem_name == "vgg16":
    mode = "train" if path is None else "test"
    problem = problems.vgg16_cifar10("cifar10",
                               mode=mode)
    net_config = {"cw": get_default_net_config(path)}
    net_assignments = None
  elif problem_name == "cifar-multi":
    mode = "train" if path is None else "test"
    problem = problems.cifar10("cifar10",
                               conv_channels=(16, 16, 16),
                               linear_layers=(32,),
                               mode=mode)
    net_config = {
        "conv": get_default_net_config(path),
        "fc": get_default_net_config(path)
    }
    conv_vars = ["conv_net_2d/conv_2d_{}/w".format(i) for i in xrange(3)]
    fc_vars = ["conv_net_2d/conv_2d_{}/b".format(i) for i in xrange(3)]
    fc_vars += ["conv_net_2d/batch_norm_{}/beta".format(i) for i in xrange(3)]
    fc_vars += ["mlp/linear_{}/w".format(i) for i in xrange(2)]
    fc_vars += ["mlp/linear_{}/b".format(i) for i in xrange(2)]
    fc_vars += ["mlp/batch_norm/beta"]
    net_assignments = [("conv", conv_vars), ("fc", fc_vars)]
  elif problem_name == "confocal_microscopy_3d":
    problem = problems.confocal_microscopy_3d(batch_size=32, num_points=5)
    net_config = {"cw": {
        "net": "CoordinateWiseDeepLSTM",
        "net_options": {"layers": (20, 20)},
        "net_path": path
    }}
    net_assignments = None
  elif problem_name == "square_cos":
    problem = problems.square_cos(batch_size=128, num_dims=2)
    net_config = {"cw": {
        "net": "CoordinateWiseDeepLSTM",
        "net_options": {"layers": (20, 20)},
        "net_path": path
    }}
    net_assignments = None
  else:
    raise ValueError("{} is not a valid problem".format(problem_name))

  if net_name == "RNNprop":
      default_config = {
              "net": "RNNprop",
              "net_options": {
                  "layers": (20, 20),
                  "preprocess_name": "fc",
                  "preprocess_options": {"dim": 20},
                  "scale": 0.01,
                  "tanh_output": True
              },
              "net_path": path
          }
      net_config = {"rp": default_config}

  return problem, net_config, net_assignments
