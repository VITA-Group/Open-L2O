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
"""Learning 2 Learn preprocessing modules."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
from meta_rnnprop_train import _make_with_custom_variables
import random

def opt_variables_initializer(opt, var_list, if_adam=False):
  vars = [opt.get_slot(var, name)
               for name in opt.get_slot_names()
               for var in var_list if var is not None]
  if if_adam:
    vars.extend(list(opt._get_beta_accumulators()))
  return tf.variables_initializer(vars)


class data_loader():
    def __init__(self, make_loss, x, constants, subsets, scale, optimizers, unroll_len):
        self.unroll_len = unroll_len
        self.optimizers = optimizers.split(",")
        self.num_subsets = len(subsets)

        self.x = x
        self.x_flat = [self.flatten_and_concat([x[i] for i in subset]) for subset in subsets]
        self.scale = scale
        scaled_x = [x[k] * scale[k] for k in range(len(scale))]
        self.loss = _make_with_custom_variables(make_loss, scaled_x)
        self.gradients = tf.gradients(self.loss, x)
        self.gradients_flat = [self.flatten_and_concat([self.gradients[i] for i in subset])
                               for subset in subsets]
        self.reset_x = tf.variables_initializer(x+constants)
        if "adam" in self.optimizers:
            self.adam = tf.train.AdamOptimizer(0.01)
            self.update_adam = self.adam.apply_gradients(zip(self.gradients, x))
            self.reset_adam = opt_variables_initializer(self.adam, x, True)
        if "rmsprop" in self.optimizers:
            self.rmsprop = tf.train.RMSPropOptimizer(0.01)
            self.update_rmsprop = self.rmsprop.apply_gradients(zip(self.gradients, x))
            self.reset_rmsprop = opt_variables_initializer(self.rmsprop, x)
        if "nag" in self.optimizers:
            self.nag = tf.train.MomentumOptimizer(0.01, 0.9, use_nesterov=True)
            self.update_nag = self.nag.apply_gradients(zip(self.gradients, x))
            self.reset_nag = opt_variables_initializer(self.nag, x)

    def flatten_and_concat(self, var_list):
        var_flat = []
        for var in var_list:
            var = tf.squeeze(tf.reshape(var, shape=(-1, 1)))
            var_flat.append(var)
        var_cat = tf.concat(var_flat, axis=0)
        return var_cat

    def get_data(self, task_i, sess, num_unrolls, assign_func, rd_scale_bound, if_scale=True, mt_k=1):
        opt_name = self.optimizers[task_i]
        update = getattr(self, "update_"+opt_name)

        # init
        sess.run(self.reset_x)

        # reset
        opt_reset = getattr(self, "reset_{}".format(opt_name))
        sess.run(opt_reset)

        # scale
        if if_scale:
            r_scale = []
            for k in self.scale:
                r_scale.append(np.exp(np.random.uniform(-rd_scale_bound, rd_scale_bound,
                                                        size=k.shape)))
            feed_rs = {p: v for p, v in zip(self.scale, r_scale)}
            k_value_list = []
            for k_id in range(len(self.scale)):
                k_value = sess.run(self.x[k_id])
                k_value = k_value / r_scale[k_id]
                k_value_list.append(k_value)
            assert assign_func is not None
            assign_func(k_value_list)
        else:
            feed_rs = {}

        # get updates
        data = {"inputs": [], "labels": []}
        x_prev = sess.run(self.x_flat)

        for ri in range(num_unrolls):
            inputs = []  # (unroll_length, num_subsets, num_params)
            labels = []
            for stepi in range(self.unroll_len):
                *gs, _ = sess.run(self.gradients_flat + [update], feed_dict=feed_rs)
                inputs.append(gs)
                for ki in range(mt_k-1):
                    sess.run(update, feed_dict=feed_rs)
                x_cur = sess.run(self.x_flat)
                x_diff = [cur-prev for cur, prev in zip(x_cur, x_prev)]
                x_prev = x_cur
                labels.append(x_diff)
            input_subsets = []  # (num_subsets, unroll_length, num_params)
            label_subsets = []
            for i in range(self.num_subsets):
                ipt = np.concatenate([ipt[i][None, :] for ipt in inputs], axis=0)
                input_subsets.append(ipt)
                lb = np.concatenate([lb[i][None, :] for lb in labels], axis=0)
                label_subsets.append(lb)
            data["inputs"].append(input_subsets)
            data["labels"].append(label_subsets)
        return data