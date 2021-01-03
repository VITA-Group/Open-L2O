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
"""Learning to learn (meta) optimizer."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import contextlib
import os
import pdb
import pickle

import mock
import sonnet as snt
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.util import nest

import networks


def _nested_assign(ref, value):
  """Returns a nested collection of TensorFlow assign operations.

  Args:
    ref: Nested collection of TensorFlow variables.
    value: Values to be assigned to the variables. Must have the same structure
        as `ref`.

  Returns:
    Nested collection (same structure as `ref`) of TensorFlow assign operations.

  Raises:
    ValueError: If `ref` and `values` have different structures.
  """
  if isinstance(ref, list) or isinstance(ref, tuple):
    if len(ref) != len(value):
      raise ValueError("ref and value have different lengths.")
    result = [_nested_assign(r, v) for r, v in zip(ref, value)]
    if isinstance(ref, tuple):
      return tuple(result)
    return result
  else:
    return tf.assign(ref, value)


def _nested_variable(init, name=None, trainable=False):
  """Returns a nested collection of TensorFlow variables.

  Args:
    init: Nested collection of TensorFlow initializers.
    name: Variable name.
    trainable: Make variables trainable (`False` by default).

  Returns:
    Nested collection (same structure as `init`) of TensorFlow variables.
  """
  if isinstance(init, list) or isinstance(init, tuple):
    result = [_nested_variable(i, name, trainable) for i in init]
    if isinstance(init, tuple):
      return tuple(result)
    return result
  else:
    return tf.Variable(init, name=name, trainable=trainable)


def _nested_tuple(elems):
  if isinstance(elems, list) or isinstance(elems, tuple):
    result = tuple([_nested_tuple(x) for x in elems])
    return result
  else:
    return elems


def _wrap_variable_creation(func, custom_getter):
  """Provides a custom getter for all variable creations."""
  original_get_variable = tf.get_variable
  def custom_get_variable(*args, **kwargs):
    if hasattr(kwargs, "custom_getter"):
      raise AttributeError("Custom getters are not supported for optimizee "
                           "variables.")
    return original_get_variable(*args, custom_getter=custom_getter, **kwargs)

  # Mock the get_variable method.
  with mock.patch("tensorflow.get_variable", custom_get_variable):
    return func()


def _get_variables(func):
  """Calls func, returning any variables created, but ignoring its return value.

  Args:
    func: Function to be called.

  Returns:
    A tuple (variables, constants) where the first element is a list of
    trainable variables and the second is the non-trainable variables.
  """
  variables = []
  constants = []

  def custom_getter(getter, name, **kwargs):
    trainable = kwargs["trainable"]
    kwargs["trainable"] = False
    variable = getter(name, **kwargs)
    if trainable:
      variables.append(variable)
    else:
      constants.append(variable)
    return variable

  with tf.name_scope("unused_graph"):
    _wrap_variable_creation(func, custom_getter)

  return variables, constants


def _make_with_custom_variables(func, variables):
  """Calls func and replaces any trainable variables.

  This returns the output of func, but whenever `get_variable` is called it
  will replace any trainable variables with the tensors in `variables`, in the
  same order. Non-trainable variables will re-use any variables already
  created.

  Args:
    func: Function to be called.
    variables: A list of tensors replacing the trainable variables.

  Returns:
    The return value of func is returned.
  """
  variables = collections.deque(variables)

  def custom_getter(getter, name, **kwargs):
    if kwargs["trainable"]:
      return variables.popleft()
    else:
      kwargs["reuse"] = True
      return getter(name, **kwargs)

  return _wrap_variable_creation(func, custom_getter)


MetaLoss = collections.namedtuple("MetaLoss", "loss, update, reset, fx, x")
MetaStep = collections.namedtuple("MetaStep", "step, update, reset, fx, x")


def _make_nets(variables, config, net_assignments):
  """Creates the optimizer networks.

  Args:
    variables: A list of variables to be optimized.
    config: A dictionary of network configurations, each of which will be
        passed to networks.Factory to construct a single optimizer net.
    net_assignments: A list of tuples where each tuple is of the form (netid,
        variable_names) and is used to assign variables to networks. netid must
        be a key in config.

  Returns:
    A tuple (nets, keys, subsets) where nets is a dictionary of created
    optimizer nets such that the net with key keys[i] should be applied to the
    subset of variables listed in subsets[i].

  Raises:
    ValueError: If net_assignments is None and the configuration defines more
        than one network.
  """
  # create a dictionary which maps a variable name to its index within the
  # list of variables.
  name_to_index = dict((v.name.split(":")[0], i)
                       for i, v in enumerate(variables))

  if net_assignments is None:
    if len(config) != 1:
      raise ValueError("Default net_assignments can only be used if there is "
                       "a single net config.")

    with tf.variable_scope("vars_optimizer"):
      key = next(iter(config))
      kwargs = config[key]
      net = networks.factory(**kwargs)

    nets = {key: net}
    keys = [key]
    subsets = [range(len(variables))]
  else:
    nets = {}
    keys = []
    subsets = []
    with tf.variable_scope("vars_optimizer"):
      for key, names in net_assignments:
        if key in nets:
          raise ValueError("Repeated netid in net_assigments.")
        nets[key] = networks.factory(**config[key])
        subset = [name_to_index[name] for name in names]
        keys.append(key)
        subsets.append(subset)
        print("Net: {}, Subset: {}".format(key, subset))

  # subsets should be a list of disjoint subsets (as lists!) of the variables
  # and nets should be a list of networks to apply to each subset.
  return nets, keys, subsets


class MetaOptimizer(object):
  """Learning to learn (meta) optimizer.

  Optimizer which has an internal RNN which takes as input, at each iteration,
  the gradient of the function being minimized and returns a step direction.
  This optimizer can then itself be optimized to learn optimization on a set of
  tasks.
  """

  def __init__(self, num_mt, **kwargs):
    """Creates a MetaOptimizer.

    Args:
      **kwargs: A set of keyword arguments mapping network identifiers (the
          keys) to parameters that will be passed to networks.Factory (see docs
          for more info).  These can be used to assign different optimizee
          parameters to different optimizers (see net_assignments in the
          meta_loss method).
    """
    self._nets = None
    self.num_mt = num_mt

    if not kwargs:
      # Use a default coordinatewise network if nothing is given. this allows
      # for no network spec and no assignments.
      self._config = {
          "coordinatewise": {
              "net": "CoordinateWiseDeepLSTM",
              "net_options": {
                  "layers": (20, 20),
                  "preprocess_name": "LogAndSign",
                  "preprocess_options": {"k": 5},
                  "scale": 0.01,
              }}}
    else:
      self._config = kwargs

  def save(self, sess, path=None, index=None):
    """Save meta-optimizer."""
    result = {}
    for k, net in self._nets.items():
      if path is None:
        filename = None
        key = k
      elif index is not None:
        filename = os.path.join(path, "{}.l2l-{}".format(k, index))
        key = filename
      else:
        filename = os.path.join(path, "{}.l2l".format(k))
        key = filename
      net_vars = networks.save(net, sess, filename=filename)
      result[key] = net_vars
    return result

  def restorer(self):
    self.restore_pl = {}
    self.assigns = {}
    for k, net in self._nets.items():
      vars = snt.get_variables_in_module(net)
      self.restore_pl[k] = collections.defaultdict(dict)
      self.assigns[k] = collections.defaultdict(dict)
      for v in vars:
        split = v.name.split(":")[0].split("/")
        module_name = split[-2]
        variable_name = split[-1]
        self.restore_pl[k][module_name][variable_name] =\
          tf.placeholder(name="{}_{}_pl".format(module_name, variable_name), shape=v.get_shape(), dtype=v.dtype)
        self.assigns[k][module_name][variable_name] = tf.assign(v, self.restore_pl[k][module_name][variable_name])

  def restore(self, sess, path, index):
    ops = []
    feed = {}
    for k, net in self._nets.items():
      filename = os.path.join(path, "{}.l2l-{}".format(k, index))
      data = pickle.load(open(filename, "rb"))
      vars = snt.get_variables_in_module(net)
      for v in vars:
        split = v.name.split(":")[0].split("/")
        module_name = split[-2]
        variable_name = split[-1]
        feed[self.restore_pl[k][module_name][variable_name]] = data[module_name][variable_name]
        ops.append(self.assigns[k][module_name][variable_name])
    sess.run(ops, feed_dict=feed)

  def meta_loss(self,
                make_loss,
                len_unroll,
                net_assignments=None,
                second_derivatives=False):
    """Returns an operator computing the meta-loss.

    Args:
      make_loss: Callable which returns the optimizee loss; note that this
          should create its ops in the default graph.
      len_unroll: Number of steps to unroll.
      net_assignments: variable to optimizer mapping. If not None, it should be
          a list of (k, names) tuples, where k is a valid key in the kwargs
          passed at at construction time and names is a list of variable names.
      second_derivatives: Use second derivatives (default is false).

    Returns:
      namedtuple containing (loss, update, reset, fx, x), ...
    """

    # Construct an instance of the problem only to grab the variables. This
    # loss will never be evaluated.
    # pdb.set_trace()
    
    x, constants = _get_variables(make_loss)

    print("Optimizee variables")
    print([op.name for op in x])
    print("Problem variables")
    print([op.name for op in constants])

    # create scale placeholder here
    scale = []
    for k in x:
      scale.append(tf.placeholder_with_default(tf.ones(shape=k.shape), shape=k.shape, name=k.name[:-2] + "_scale"))

    # Create the optimizer networks and find the subsets of variables to assign
    # to each optimizer.
    nets, net_keys, subsets = _make_nets(x, self._config, net_assignments)
    print('nets', nets)
    print('subsets', subsets)
    # Store the networks so we can save them later.
    self._nets = nets

    # Create hidden state for each subset of variables.
    state = []
    with tf.name_scope("states"):
      for i, (subset, key) in enumerate(zip(subsets, net_keys)):
        net = nets[key]
        with tf.name_scope("state_{}".format(i)):
          state.append(_nested_variable(
              [net.initial_state_for_inputs(x[j], dtype=tf.float32)
               for j in subset],
              name="state", trainable=False))

    def update(net, fx, x, state):
      """Parameter and RNN state update."""
      with tf.name_scope("gradients"):
        gradients = tf.gradients(fx, x)

        # Stopping the gradient here corresponds to what was done in the
        # original L2L NIPS submission. However it looks like things like
        # BatchNorm, etc. don't support second-derivatives so we still need
        # this term.
        if not second_derivatives:
          gradients = [tf.stop_gradient(g) for g in gradients]

      with tf.name_scope("deltas"):
        deltas, state_next = zip(*[net(g, s) for g, s in zip(gradients, state)])
        state_next = _nested_tuple(state_next)
        state_next = list(state_next)

      return deltas, state_next

    def time_step(t, fx_array, x, state):
      """While loop body."""
      x_next = list(x)
      state_next = []

      with tf.name_scope("fx"):
        scaled_x = [x[k] * scale[k] for k in range(len(scale))]
        fx = _make_with_custom_variables(make_loss, scaled_x)
        fx_array = fx_array.write(t, fx)

      with tf.name_scope("dx"):
        for subset, key, s_i in zip(subsets, net_keys, state):
          x_i = [x[j] for j in subset]
          deltas, s_i_next = update(nets[key], fx, x_i, s_i)

          for idx, j in enumerate(subset):
            delta = deltas[idx]
            x_next[j] += delta
          state_next.append(s_i_next)

      with tf.name_scope("t_next"):
        t_next = t + 1

      return t_next, fx_array, x_next, state_next

    # Define the while loop.
    fx_array = tf.TensorArray(tf.float32, size=len_unroll+1,
                              clear_after_read=False)
    _, fx_array, x_final, s_final = tf.while_loop(
        cond=lambda t, *_: t < len_unroll,
        body=time_step,
        loop_vars=(0, fx_array, x, state),
        parallel_iterations=1,
        swap_memory=True,
        name="unroll")

    with tf.name_scope("fx"):
      scaled_x_final = [x_final[k] * scale[k] for k in range(len(scale))]
      fx_final = _make_with_custom_variables(make_loss, scaled_x_final)
      fx_array = fx_array.write(len_unroll, fx_final)

    loss = tf.reduce_sum(fx_array.stack(), name="loss")

    ##################################
    ### multi task learning losses ###
    ##################################
    # state (num_subsets, num_x, (num_layers, (h,c)))
    # state_reshape (num_mt, num_subsets, (num_layers, (h, c)))
    state_reshape = []
    num_layers = len(state[0][0])
    for mti in range(self.num_mt):
      state_reshape_mti = []
      for state_subset in state:
        state_layers = ()
        for li in range(num_layers):
          h = tf.concat([st_x[li][0] for st_x in state_subset], axis=0)
          c = tf.concat([st_x[li][1] for st_x in state_subset], axis=0)
          h = tf.Variable(h, name="state_reshape_h", trainable=False)
          c = tf.Variable(c, name="state_reshape_c", trainable=False)
          state_layers += ((h, c),)
        state_reshape_mti.append(state_layers)
      state_reshape.append(state_reshape_mti)
    if self.num_mt > 0:
      shapes = [st_subset[0][0].get_shape().as_list()[0] for st_subset in state_reshape[0]]
    else:
      shapes = []
    num_params_total = sum(shapes)
    print("number of parameters = {}".format(num_params_total))

    # placeholder (num_mt, num_subsets, len_unroll, num_params)
    mt_labels = []
    mt_inputs = []
    for i in range(self.num_mt):
      mt_labels.append(
        [tf.placeholder(dtype=tf.float32, shape=(len_unroll, shapes[j]),
                        name="mt{}_label_subset{}".format(i, j))
         for j in range(len(subsets))]
      )
      mt_inputs.append(
        [tf.placeholder(dtype=tf.float32, shape=(len_unroll, shapes[j]),
                        name="mt{}_input_subset{}".format(i, j))
         for j in range(len(subsets))]
      )

    # loop
    def time_step_mt(mti):
      def time_step_func(t, loss_array, states):
        loss_t_sum = 0.0
        state_next = []
        for si, (k, st) in enumerate(zip(net_keys, states)):
          net = nets[k]
          g_input = tf.gather(mt_inputs[mti][si], indices=t, axis=0)
          g_label = tf.gather(mt_labels[mti][si], indices=t, axis=0)
          delta, state_next_si = net(g_input, st)
          loss_t_sum += tf.reduce_sum((g_label - delta) * (g_label - delta)) * 0.5
          state_next_si = _nested_tuple(state_next_si)
          state_next.append(state_next_si)
        loss_t = loss_t_sum / num_params_total
        loss_array = loss_array.write(t, loss_t)
        t_next = t + 1
        return t_next, loss_array, state_next

      return time_step_func

    loss_arrays = [tf.TensorArray(tf.float32, size=len_unroll, clear_after_read=False)
                   for _ in range(self.num_mt)]
    state_reshape_final = []
    for mti in range(self.num_mt):
      loss_array = loss_arrays[mti]
      _, loss_array, state_reshape_final_mti = tf.while_loop(
        cond=lambda t, *_: t < len_unroll,
        body=time_step_mt(mti),
        loop_vars=(0, loss_array, state_reshape[mti]),
        parallel_iterations=1,
        swap_memory=True,
        name="unroll_mt")
      loss_arrays[mti] = loss_array
      state_reshape_final.append(state_reshape_final_mti)

    # loss
    loss_mt = [tf.reduce_sum(loss_array.stack(), name="loss_mt{}".format(i))
               for i, loss_array in enumerate(loss_arrays)]

    # Reset the state; should be called at the beginning of an epoch.
    with tf.name_scope("reset"):
      variables = (nest.flatten(state) +
                   x + constants)
      # Empty array as part of the reset process.
      reset = [tf.variables_initializer(variables), fx_array.close()]

      # mt
      variables_mt = [nest.flatten(state_reshape[mti]) for mti in range(self.num_mt)]
      reset_mt = [[tf.variables_initializer(variables_mt[mti]), loss_arrays[mti].close()]
                  for mti in range(self.num_mt)]

    # Operator to update the parameters and the RNN state after our loop, but
    # during an epoch.
    with tf.name_scope("update"):
      update = (nest.flatten(_nested_assign(x, x_final)) +
                nest.flatten(_nested_assign(state, s_final)))
      update_mt = [(nest.flatten(_nested_assign(state_reshape[mti], state_reshape_final[mti])))
                   for mti in range(self.num_mt)]

    # Log internal variables.
    for k, net in nets.items():
      print("Optimizer '{}' variables".format(k))
      print([op for op in snt.get_variables_in_module(net)])

    return MetaLoss(loss, update, reset, fx_final, x_final), scale, x, constants, subsets,\
           loss_mt, update_mt, reset_mt, mt_labels, mt_inputs

  def meta_minimize(self, make_loss, len_unroll, learning_rate=0.01, **kwargs):
    """Returns an operator minimizing the meta-loss.

    Args:
      make_loss: Callable which returns the optimizee loss; note that this
          should create its ops in the default graph.
      len_unroll: Number of steps to unroll.
      learning_rate: Learning rate for the Adam optimizer.
      **kwargs: keyword arguments forwarded to meta_loss.

    Returns:
      namedtuple containing (step, update, reset, fx, x), ...
    """

    info, scale, x, constants, subsets, loss_mt, update_mt, reset_mt, mt_labels, mt_inputs = \
        self.meta_loss(make_loss, len_unroll, **kwargs)
    optimizer = tf.train.AdamOptimizer(learning_rate)
    step = optimizer.minimize(info.loss)

    # mt
    optimizer_mt = []
    steps_mt = []
    for loss_mti in loss_mt:
      optimizer_mt.append(tf.train.AdamOptimizer(learning_rate))
      steps_mt.append(optimizer_mt[-1].minimize(loss_mti))

    self.restorer()

    return MetaStep(step, *info[1:]), scale, x, constants, subsets, \
           loss_mt, steps_mt, update_mt, reset_mt, mt_labels, mt_inputs