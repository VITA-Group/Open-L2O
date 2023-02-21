
import tensorflow as tf
import numpy as np
import pickle
import random

def opt_variables_initializer(opt, var_list, if_adam=False):
  vars = [opt.get_slot(var, name)
               for name in opt.get_slot_names()
               for var in var_list if var is not None]
  if if_adam:
    vars.extend(list(opt._get_beta_accumulators()))
  return tf.variables_initializer(vars+var_list)


class mt_utils():
  def __init__(self, problem, opt_names="adam"):
    self.graph_local = tf.Graph()
    with self.graph_local.as_default():
      # create variables
      vars = []
      for ti, shape in enumerate(problem.param_shapes):
        var = tf.get_variable(name="local_var_{}".format(ti), shape=shape)
        vars.append(var)
      self.vars = vars
      # placeholder
      self.data_pl = tf.placeholder(tf.float32)
      self.labels_pl = tf.placeholder(tf.float32)
      self.lr_pl = tf.placeholder(tf.float32)
      # obj
      self.obj = problem.objective(self.vars, self.data_pl, self.labels_pl)
      # gradients
      self.grads = tf.gradients(self.obj, self.vars)
      # optmizer
      if 'adam' in opt_names:
        self.opt_adam = tf.train.AdamOptimizer(self.lr_pl)
        self.update_adam = self.opt_adam.apply_gradients(zip(self.grads, self.vars))
        self.reset_adam = opt_variables_initializer(self.opt_adam, self.vars, True)
      if 'rmsprop' in opt_names:
        self.opt_rmsprop = tf.train.RMSPropOptimizer(self.lr_pl)
        self.update_rmsprop = self.opt_rmsprop.apply_gradients(zip(self.grads, self.vars))
        self.reset_rmsprop = opt_variables_initializer(self.opt_rmsprop, self.vars, False)
      if 'nag' in opt_names:
        self.opt_nag = tf.train.MomentumOptimizer(self.lr_pl, 0.9, use_nesterov=True)
        self.update_nag = self.opt_nag.apply_gradients(zip(self.grads, self.vars))
        self.reset_nag = opt_variables_initializer(self.opt_nag, self.vars, False)
      self.sess = tf.Session()

  def get_mt_labels(self, init_tensors, data, labels, mini_batches, dataset_batches, opt_name="adam", k=1):
    label_range = np.max(labels) + 1
    rd_lr = 0.01
    print ("mt opt name: {}".format(opt_name))
    print ("mt opt learning rate = {}".format(rd_lr))
    with self.graph_local.as_default():
      update = getattr(self, "update_{}".format(opt_name))
      reset = getattr(self, "reset_{}".format(opt_name))
      self.sess.run(reset)
      for ti, tensor in enumerate(init_tensors):
         self.sess.run(self.vars[ti].assign(tensor))
      with tf.control_dependencies(self.vars):
        # loop
        mt_labels = []  # (num_unrolls, unroll_len, num_params, param_shape)
        num_unrolls = len(dataset_batches)
        for unroll_itr in range(num_unrolls):
          batches = dataset_batches[unroll_itr]
          mini_batch = mini_batches[unroll_itr]
          mini_data = data[mini_batch]
          mini_labels = labels[mini_batch]
          mini_labels = np.eye(label_range)[mini_labels]
          unroll_len = len(batches)//k
          assert len(batches) % k == 0
          mt_labels_roll = []  # (unroll_len, num_params, param_shape)
          x_prev = self.sess.run(self.vars)
          for itr in range(unroll_len):
            for ki in range(k):
              batch = batches[itr*k+ki]
              data_batch = mini_data[batch]
              labels_batch = mini_labels[batch]
              feed = {self.data_pl: data_batch, self.labels_pl: labels_batch, self.lr_pl: rd_lr}
              self.sess.run(update, feed_dict=feed)
            x_cur = self.sess.run(self.vars)
            x_update = [x2-x1 for x1, x2 in zip(x_cur, x_prev)]  # (num_params, param_shape)
            x_prev = x_cur
            mt_labels_roll.append(x_update)
          mt_labels.append(mt_labels_roll)
        return mt_labels






