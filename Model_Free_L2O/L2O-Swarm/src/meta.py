
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import contextlib
import os

import mock
import sonnet as snt
import tensorflow as tf
import entropy_loss

import numpy as np

from tensorflow.python.framework import ops
from tensorflow.python.util import nest

import networks
import pickle
import pdb


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


MetaLoss = collections.namedtuple("MetaLoss", "loss, update, reset, fx, x, test")
MetaStep = collections.namedtuple("MetaStep", "step, update, reset, fx, x, test, fc_weights, fc_bias, fc_va")


def _make_nets(variables, config, net_assignments):
  """Creates the optimizer networks.

  Args:object() takes no parameters
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

  def __init__(self, **kwargs):
    """Creates a MetaOptimizer.

    Args:
      **kwargs: A set of keyword arguments mapping network identifiers (the
          keys) to parameters that will be passed to networks.Factory (see docs
          for more info).  These can be used to assign different optimizee
          parameters to different optimizers (see net_assignments in the
          meta_loss method).
    """
    self._nets = None
    #self.dims=2
    self.num_lstm = 4
#    pdb.set_trace()
#    trainable = tf.constant(False, dtype = tf.bool)
    #intra attention x_minimal verification 
#    self.fx_minimal = [tf.Variable(float("inf"),trainable = False) for i in range(self.num_lstm)]
    self.fx_minimal = tf.Variable(float("inf"),trainable = False)
    #intra attention x_minimal if fx_minimal is satisfied
    
#    delta_shape=([784,20],[20,],[20,10],[10,])
#    tensors=[]
#    for i in range(4):
#        tensor = tf.Variable(tf.zeros(delta_shape[i]), trainable = False)
#        tensors.append(tensor)
#    self.x_minimal = [tensors for i in range(self.num_lstm)]
#    self.pre_deltas = [tensors for i in range(self.num_lstm)]
#    self.pre_gradients = [tensors for i in range(self.num_lstm)]
    self.x_minimal = []
    self.pre_deltas = []
    self.pre_gradients = []
    self.intra_features = 4
    self.fc_kernel = []
    self.fc_bias = []
    self.fc_va = []
#    fc_kernel_shape = ([20, 15680*2], [20, 20*2], [20, 200*2], [10, 10*2])
#    fc_bias_shape = ([20, self.intra_features ], [20, self.intra_features], [20, self.intra_features], [10, self.intra_features])
#    fc_va_shape=([1,20],[1,20],[1,20],[1,10])
#    for i in range(4):
#      sub_fc_kernel = tf.Variable(tf.random_normal(fc_kernel_shape[i]))
#      sub_fc_bias = tf.Variable(tf.random_normal(fc_bias_shape[i]))
#      sub_fc_va = tf.Variable(tf.ones(fc_va_shape[i]), trainable = False)
#      self.fc_kernel.append(sub_fc_kernel)
#      self.fc_bias.append(sub_fc_bias)
#      self.fc_va.append(sub_fc_va)

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

  def save(self, sess, path=None):
    """Save meta-optimizer."""
    result = {}
    for k, net in self._nets.items():
      if path is None:
        filename = None
        key = k
      else:
        filename = os.path.join(path, "{}.l2l".format(k))
        key = filename
      net_vars = networks.save(net, sess, filename=filename)
      result[key] = net_vars
    return result
  
  
  def meta_loss(self,
                make_loss,
                len_unroll,
                net_assignments=None,
                model_path = None,
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
      namedtuple containing (loss, update, reset, fx, x)
    """

    # Construct an instance of the problem only to grab the variables. This
    # loss will never be evaluated.
    
    sub_x, sub_constants=_get_variables(make_loss)
    print (sub_x, sub_constants)
#    pdb.set_trace()
#    print(len(sub_x))
    
    def intra_init(x):
      
#      print(self.x_minimal)
      fc_columns = 10
#      pdb.set_trace()
      
      if model_path == None:
        for i in range(len(x)):
#      fc_kernel_shape = ([20, 15680*2], [20, 20*2], [20, 200*2], [10, 10*2])
#      fc_bias_shape = ([20, self.intra_features ], [20, self.intra_features], [20, self.intra_features], [10, self.intra_features])
#      fc_va_shape=([1,20],[1,20],[1,20],[1,10])
#      pdb.set_trace()
      
#        x[i] = tf.reshape(x[i],[1,-1])
#        print(x[i])
          fc_shape = tf.reshape(x[i],[1,-1]).get_shape()
          print(fc_shape)
#            print(x[i])
#            print(size)
#            res =tf.Variable([fc_columns-1,0],trainable=False)
#            size = size + res
#            print(size)
          fc_kernel = np.random.rand(fc_columns, fc_shape[1]*2)
          print(fc_kernel.shape)
          fc_bias_shape = [fc_columns, self.intra_features ]
          fc_va_shape = [1,fc_columns]
#           ker = tf.Variable(tf.random_normal(fc_shape),trainable = False)
#           sub_ker = tf.concat([ker, ker],axis = 1)
#           fc_kernel = tf.concat([sub_ker for i in range(fc_columns)], axis = 0)
#           kernel_shape = tf.shape(fc_kernel)
#           print(kernel_shape)
          sub_fc_kernel = tf.Variable(tf.random_normal(fc_kernel.shape))
#           print(sub_fc_kernel)
          sub_fc_bias = tf.Variable(tf.random_normal(fc_bias_shape))
#           print(sub_fc_bias)
          sub_fc_va = tf.Variable(tf.ones(fc_va_shape), trainable = False)
#           print(sub_fc_va)
          self.fc_kernel.append(sub_fc_kernel)
          self.fc_bias.append(sub_fc_bias)
          self.fc_va.append(sub_fc_va)
      else:
        with open('./{}/loss_record.pickle'.format(model_path),'rb') as loss:
          data = pickle.load(loss)
        self.fc_kernel = [tf.Variable(item) for item in data['fc_weights']]
        self.fc_bias = [tf.Variable(item) for item in data['fc_bias']]
        for i in range(len(x)):
          fc_va_shape = [1,fc_columns]
          sub_fc_va = tf.Variable(tf.ones(fc_va_shape), trainable = False)
          self.fc_va.append(sub_fc_va)
    
    intra_init(sub_x)

    def vars_init(x):
#      pdb.set_trace()
      variables = []
      for i in range(len(x)):
        vars_shape = x[i].get_shape()
#        vars_shape = vars_shape.tolist()
        print(vars_shape)
        vars = np.random.rand(self.num_lstm,*vars_shape)
        print(vars.shape)
#        vars_shape = [item for item in vars_shape]
#        vars_shape.insert(0, self.num_lstm)
#        print(vars_shape)
#        var = tf.Variable(tf.ones(vars_shape), trainable = False)
#        sub_var = tf.stack([var for j in range(self.num_lstm)], axis = 0)
#        sub_var_shape = tf.shape(sub_var)
        sub_vars = tf.Variable(tf.random.normal(vars.shape, mean=0.0, stddev=0.01), trainable = False)
        #sub_vars = tf.Variable(tf.random.uniform(vars.shape, minval=-3, maxval=3), trainable = False)
#        sub_vars = tf.stack(sub_vars, axis = 0)
        variables.append(sub_vars)
      return variables
    
    x = vars_init(sub_x)
    constants = vars_init(sub_constants)
    self.x_minimal = vars_init(sub_x)
    print(sub_x, x[0].shape)
    print(constants)
    #exit(0)
    '''
    def intra_vars_init(x):
      variables = []
      for i in range(len(x)):
        sub_vars = []
        for j in range(self.num_lstm):
          vars_shape = tf.shape(x[i])
          var = tf.Variable(tf.random_normal(vars_shape, mean = 0, stddev = 0.01), trainable = False)
          sub_vars.append(var)
        sub_vars = tf.stack(sub_vars, axis = 0)
        variables.append(sub_vars)
      return variables
    '''
    self.pre_deltas = vars_init(sub_x)
    self.pre_gradients = vars_init(sub_x)
    
#    x=[sub_x for i in range(self.num_lstm)]
#    constants=[sub_constants for i in range(self.num_lstm)]
    
    
    print("x.length",len(x), x[0].shape, type(x[0]))
    #np.savetxt("xlength", x)
    print("Optimizee variables")
    print([op.name for op in sub_x])
    print("Problem variables")
    print([op.name for op in sub_constants])
#
#    fc_kernel = []
#    fc_bias = []
#    fc_va = []
#    fc_kernel_shape = ([20, 15680*2], [20, 20*2], [20, 200*2], [10, 10*2])
#    fc_bias_shape = ([20, self.intra_features ], [20, self.intra_features], [20, self.intra_features], [10, self.intra_features])
#    fc_va_shape=([1,20],[1,20],[1,20],[1,10])
#    for i in range(4):
#      sub_fc_kernel = tf.Variable(tf.random_normal(fc_kernel_shape[i]))
#      sub_fc_bias = tf.Variable(tf.random_normal(fc_bias_shape[i]))
#      sub_fc_va = tf.Variable(tf.ones(fc_va_shape[i]), trainable = False)
#      fc_kernel.append(sub_fc_kernel)
#      fc_bias.append(sub_fc_bias)
#      fc_va.append(sub_fc_va)

#    x_loop = x
    # Create the optimizer networks and find the subsets of variables to assign
    # to each optimizer.
    nets, net_keys, subsets = _make_nets(sub_x, self._config, net_assignments) 

    # Store the networks so we can save them later.
    self._nets = nets

    # Create hidden state for each subset of variables.
    '''
    if len(subsets) > 1:
      state = []
    else:
      state=[[] for i in range(len(subsets))]
    '''
    state = []
    with tf.name_scope("states"):
      for i, (subset, key) in enumerate(zip(subsets, net_keys)):
        net = nets[key]
        with tf.name_scope("state_{}".format(i)):
          state.append(_nested_variable(
              [net.initial_state_for_inputs(tf.stack(x[j], axis = 0), dtype=tf.float32)
               for j in subset],
              name="state", trainable=False))
      '''
      if len(subsets) > 1:
        state.append(single_state) 
      else:
        for i in range(len(subsets))
          state[i].append(single_state[i]) 
      '''
#    pdb.set_trace()
#    print(x)
#    print(state)
    
    
    def update(t, net, gradients, x, attraction, state):
      """Parameter and RNN state update."""
      
   
      def attraction_init(mat):
#        pdb.set_trace()
        alpha = 1
        print(mat)
#        mat_split = tf.split(mat_x, self.num_lstm)
#        mat_split = [tf.reshape(item, [1, -1]) for item in mat_split]
        norm = []
        for i in range(self.num_lstm):
          print(mat[i])
          mat_norm = tf.reshape(mat[i], [self.num_lstm, -1])
          print(mat_norm)
          mat_l2 = tf.reduce_sum(tf.multiply(mat_norm, mat_norm), axis = 1)
          print(mat_l2)
          mat_l2 = -alpha*tf.reshape(mat_l2, [1, -1])
          print(mat_l2)
          mat_softmax = tf.nn.softmax(mat_l2)
          print(mat_softmax)
          norm.append(tf.matmul(mat_softmax, mat_norm))
#          sub_norm = tf.concat([mat_split[i] for j in range(self.num_lstm)], axis = 0)
#          sub_norm = tf.tile(mat_split[i], [self.num_lstm, 1])
#          sub_norm = sub_norm - tf.reshape(tf.stack(mat_y[i], \
#                                        axis= 0) , [self.num_lstm,-1])
#          attraction_norm = -alpha*tf.matmul(mat_split[i],tf.transpose(sub_norm))
#          attraction_intra = tf.nn.softmax(attraction_norm)
#          sub_attraction = tf.matmul(attraction_intra, sub_norm)
#          norm.append(sub_attraction)
        print(norm)
        attraction = tf.reshape(tf.stack(norm, axis = 0), tf.shape(mat[0]))
        print(attraction)
        return attraction
      
        
#      pdb.set_trace()
      print(attraction)
      x_attraction = [attraction_init(sub_attraction) for sub_attraction in attraction]
      print(x_attraction)
      def inter_attention(mat_a,mat_b):
#        pdb.set_trace()
        l=1
        gama=1/self.num_lstm
#        origin = tf.reshape(mat_a, [self.num_lstm, -1])
        origin = mat_a
        print(origin)
        grad = mat_b 
        print(grad)
        def x_res_init(mat):
#          pdb.set_trace()
          mat_split=tf.split(mat, self.num_lstm)
          norm = []
          for i in range(self.num_lstm):
#            sub_norm = tf.concat([mat_split[i] for j in range(self.num_lstm)], axis = 0)
            sub_norm = tf.tile(mat_split[i], [self.num_lstm, 1])
            sub_norm = sub_norm - mat
            norm.append(tf.matmul(mat_split[i],tf.transpose(sub_norm)))
          result = (-1/2*l)*tf.concat(norm, axis = 0)
          print(result)
          normalise = tf.nn.softmax(tf.transpose(result))
          return normalise

        def attention(Mat_a,Mat_b):
          result = gama*tf.matmul(Mat_a,Mat_b)
          Mat_b = tf.add(Mat_b,result)
          return Mat_b
#        pdb.set_trace()
        matmul1 = tf.matmul(grad,tf.transpose(grad))
        print(matmul1)
#        matmul2 = tf.matmul(origin,tf.transpose(origin))
#        matmul2 = tf.exp(matmul2/2*l)
#        print(matmul2)
        softmax_grad = tf.nn.softmax(tf.transpose(matmul1))
        print(softmax_grad)
#        softmax_grad = tf.transpose(softmax_grad)
        softmax_origin = x_res_init(origin)
        print(softmax_origin)
#        softmax_origin = tf.transpose(softmax_origin)
        input_mul_x = tf.matmul(softmax_grad,softmax_origin)
        e_ij = attention(input_mul_x,grad)
#        e_ij = attention(softmax_grad,grad)
        print(e_ij)
        return e_ij
      
      def intra_attention( grad, pre_grad, x, x_min, sub_x_attraction, ht, fc_kernel, fc_bias, fc_va):
#        pdb.set_trace()
#        print(x)
        
#        shape=([1,15680],[1,20],[1,200],[1,10])
#        reshape=([784,20],[20,],[20,10],[10,])
        beta = 0.9
#        sub_grad = tf.unstack(grad, axis = 0)
#        momentum = tf.unstack(beta*pre_grad, axis = 0)
#        intra_dim = len(grad.get_shape()) - 1
#        print(intra_dim)
        intra_feature = tf.concat([grad, beta*pre_grad, x - x_min, sub_x_attraction], axis=0)
        intra_feature = tf.reshape(intra_feature,[self.num_lstm*self.intra_features, -1])
#        intra_feature = tf.reshape(intra_feature, [self.intra_features, -1])
        print(intra_feature)
        ht_concat = tf.concat([ht for i in range(self.intra_features)], axis = 0)
        ht_concat = tf.reshape(ht_concat, [self.num_lstm*self.intra_features, -1])
        print(ht_concat)
        intra_concat = tf.concat([intra_feature, ht_concat], axis = 1)
        print(intra_concat)
#        intra_concat = tf.reshape(intra_concat, [self.intra_features, -1])
#        print(intra_concat)
        intra_fc_bias = tf.tile(fc_bias, [1, self.num_lstm])
        print(intra_fc_bias)
        intra_fc = tf.tanh(tf.matmul(fc_kernel,tf.transpose(intra_concat)) + intra_fc_bias)
        print(intra_fc)
        va = fc_va
        print(va)
        b_ij = tf.matmul(va,intra_fc)
        print(b_ij)
        p_ij = tf.nn.softmax(tf.reshape(b_ij, [self.num_lstm, -1]))
        print(p_ij)
        p_ij = tf.reshape(p_ij, [self.num_lstm*self.intra_features, 1])
        print(p_ij)
        gradient = tf.multiply(p_ij, intra_feature)
        gradient = tf.reshape(gradient, [self.intra_features, -1])
        gradient = tf.reduce_sum(gradient, axis = 0)
        print(gradient)
        gradient = tf.reshape(gradient, tf.shape(grad))
        print(gradient)
#        gradient = tf.reshape(gradient, reshape[i])
#        intra_fc_bias = tf.concat([fc_bias for i in range(self.num_lstm)], axis = 1)
#        sub_x = tf.reshape(x, [1,-1])
#        sub_x_min = tf.reshape(x_min, [1,-1])
#        sub_ht = tf.unstack(ht, axis = 0)
#        x_res = tf.unstack((x - x_min) ,axis = 0)
#        gradients = []
#        for j in range(self.num_lstm):
#          intra_feature = tf.concat([sub_grad[j], momentum[j], x_res[j]],axis=0)
#          print(intra_feature)
#          intra_feature = tf.reshape(intra_feature, [self.intra_features, -1])
#          print(intra_feature)
#          ht_concat = tf.concat([sub_ht[j] for i in range(self.intra_features)],axis = 0)
#          print(ht_concat)
#          ht_concat = tf.reshape(ht_concat, [self.intra_features, -1])
#          print(ht_concat)
##          grad_concat = tf.concat([sub_grad,sub_ht],axis=1)            
##          moment_concat = tf.concat([sub_moment,sub_ht],axis=1)
##          x_res_concat = tf.concat([x_res,sub_ht],axis=1)
##          intra_concat = tf.concat([grad_concat,moment_concat,x_res_concat],axis=0)
#          intra_concat = tf.concat([intra_feature, ht_concat], axis = 1)
#          print(intra_concat)
##        intra_concat = tf.transpose(intra_concat)
#          intra_fc=tf.tanh(tf.matmul(fc_kernel,tf.transpose(intra_concat)) + fc_bias)
#          va = fc_va
#          b_ij = tf.matmul(va,intra_fc)
#          p_ij = tf.nn.softmax(b_ij)
#          gradient = tf.matmul(p_ij, intra_feature)
#          gradient = tf.reshape(gradient, tf.shape(grad[j]))
#          gradients.append((gradient))
##        print(tf.stack(gradientsgrients, axis = 0)
        return gradient
      with tf.name_scope("gradients"):
#        pdb.set_trace()
#          print(fx[i])
        print(x)
            
        # Stopping the gradient here corresponds to what was done in the
        # original L2L NIPS submission. However it looks like things like
        # BatchNorm, etc. don't support second-derivatives so we still need
        # this term.
        if not second_derivatives:
          gradients = [tf.stop_gradient(g) for g in gradients]
        print(gradients)
#      with tf.name_scope("intra_attention"):
#           
#        print(sub_gradients)
#        print(x_min)
#        print(ht)
#        sub_gradients = intra_attention(sub_gradients, pre_grads, x[i], x_min, ht)

        
#      pdb.set_trace()
      print(gradients)
      with tf.name_scope("inter_attention"):
##x to matrix
        for i in range(len(x)):
#          xi = tf.stack(x[i], axis = 0)
          shape = tf.shape(x[i])
          mat_x = tf.reshape(x[i], [self.num_lstm,-1])
          
          mat_grads = tf.reshape(gradients[i], [self.num_lstm,-1])
          
##inter-attention
          inter_grads=inter_attention(mat_x, mat_grads)
          gradients[i] = tf.reshape(inter_grads, shape)

        
        print('mnist_gradients',gradients)


      with tf.name_scope("deltas"):
        
#        pdb.set_trace()
        x_min = self.x_minimal
        ht = self.pre_deltas
        pre_grads = self.pre_gradients
        deltas, state_next = zip(*[net(intra_attention( grad, pre_grad, x, x_min, sub_x_attraction, 
        ht, fc_kernel, fc_bias , fc_va), s)
        for grad, pre_grad, x, x_min, sub_x_attraction, ht, fc_kernel, fc_bias , fc_va, s in zip(gradients,
        pre_grads, x, x_min, x_attraction, ht, self.fc_kernel, self.fc_bias , self.fc_va, state)])
        self.pre_deltas = deltas
        print(deltas)
        print(state_next)
        state_next = list(state_next)
        self.pre_gradients=gradients
        print(state_next)
      
#      print(state_next)
      return deltas, state_next
#time_step的参数初始化返回的x_now代表当前mnist网络的x，x_next代表用lstm进行梯度更新后的mnist网络x   
#intra&inter time-step
#    pdb.set_trace()
#    print(x)
    def time_step(t, fx_array, x, x_array, state):
      """While loop body."""
#      pdb.set_trace()
#      print(x)
      x_next = list(x)
      state_next = []

#      fx_x = [tf.unstack(tensor, axis = 0) for tensor in x]
      
      with tf.name_scope("fx"):
        update_fx = []
        gradients = []
        for z in range(self.num_lstm):
          sub_x = [item[z] for item in x]
          sub_fx_batch = _make_with_custom_variables(make_loss, sub_x)
          sub_fx = tf.reduce_mean(sub_fx_batch)

          x_array = x_array.write(t*self.num_lstm + z, sub_x[0])
          fx_array = fx_array.write(t*self.num_lstm + z, sub_fx_batch)
          sub_gradients = tf.gradients(sub_fx, sub_x)
          print(sub_gradients)
          update_fx.append(sub_fx)
          gradients.append(sub_gradients)
        
#        pdb.set_trace()
        print(gradients)
        gradients = zip(*gradients)
        gradients = [tf.stack(list(gradient), axis = 0)for gradient in gradients]
        print(gradients)
#        pdb.set_trace()
        attraction = []
        for j in range(len(x)):
          attraction_x = []
          for ind1, item1 in enumerate(update_fx):
            sub_attraction_x = []
            for ind2, item2 in enumerate(update_fx):
#              def f1(): return sub_attraction_x
#              def f2(): sub_attraction_x = sub_attraction_x.remove(sub_attraction_x[ind2]) return sub_attraction_x
#              sub_attraction_x = tf.cond(tf.greater(item1, item2), lambda:f1(), lambda:f2())
              return_value = tf.where(tf.greater(item1, item2), x[j][ind1] - x[j][ind1], x[j][ind1] - x[j][ind2])
              print(return_value)
              sub_attraction_x.append(return_value)
              print(sub_attraction_x)
            attraction_x.append(tf.stack(sub_attraction_x, axis = 0))
            print(attraction_x)
          attraction.append(attraction_x)
        print(attraction)
#        print(x)
##        test_xg = [tf.unstack(tensor, axis = 0) for tensor in x]
#        for k in range(self.num_lstm):
#          test_x = [item[k] for item in x]
##          test_fx = _make_with_custom_variables(make_loss, test_x)
#          test_gradients = tf.gradients(update_fx[k], test_x)
#          print(test_gradients)
#          fx_array = fx_array.write(t*self.num_lstm + z, sub_fx)
        fx_sum = tf.reduce_sum(tf.stack(update_fx))
        def f1(): return self.fx_minimal, self.x_minimal
        def f2(): return fx_sum , x
        self.fx_minimal, self.x_minimal = tf.cond(tf.greater(fx_sum, self.fx_minimal), lambda:f1(), lambda:f2())
      with tf.name_scope("dx"):
        for subset, key, s_i in zip(subsets, net_keys, state):
          
#          pdb.set_trace()   
#          print(update_fx)
          x_i = [x[j] for j in subset]
          
#          print(x_i)
          deltas, s_i_next = update(t, nets[key], gradients, x_i, attraction, s_i)

          ratio=1.
          for idx, j in enumerate(subset):
            x_next[j] += deltas[idx]*ratio
          state_next.append(s_i_next)

      with tf.name_scope("t_next"):
        t_next = t + 1

      
        
      return t_next, fx_array, x_next, x_array, state_next

    
    # Define the while loop.
    fx_array = tf.TensorArray(tf.float32, size=(len_unroll + 1)*self.num_lstm,
                              clear_after_read=False)

    # we need x_array for calculating the entropy loss
    x_array = tf.TensorArray(tf.float32, size=(len_unroll + 1)*self.num_lstm,
                              clear_after_read=False)

    _, fx_array, x_final, x_array, s_final = tf.while_loop(
        cond=lambda t, *_: t < len_unroll,
        body=time_step,
        loop_vars=(0, fx_array, x, x_array, state),
        parallel_iterations=1,
        swap_memory=True,
        name="unroll")

    with tf.name_scope("fx"):
#      pdb.set_trace()
#      print('x_final',x_final)
      final_x = [tf.unstack(tensor, axis = 0) for tensor in x_final]
      print ("final_x", final_x[0][0].shape, x_final[0].shape)
      with tf.name_scope("fx"):
        fx_final = []
        for z in range(self.num_lstm):
          sub_x = [item[z] for item in final_x]
          print ("sub_x", type(sub_x[0]))
          sub_fx_final_batch = _make_with_custom_variables(make_loss, sub_x)
          sub_fx_final = tf.reduce_mean(sub_fx_final_batch)
          print ('sub_x[0]', sub_x[0].shape, sub_fx_final.shape)
          

          fx_array = fx_array.write(len_unroll*self.num_lstm + z, sub_fx_final_batch)
          x_array = x_array.write(len_unroll*self.num_lstm + z, sub_x[0])
          fx_final.append(sub_fx_final)
         


    print (x[0].shape, x_final[0].shape,  len(fx_final),'xinfal11', len_unroll)
    
    
    loss = entropy_loss.self_loss(x_array.stack(), fx_array.stack(), (len_unroll + 1)*self.num_lstm)
    #loss = tf.reduce_mean(tf.reduce_sum(fx_array.stack(), -1))
    #print (loss.shape)
    #exit(0)
    

    # Reset the state; should be called at the beginning of an epoch.
    
    # Reset the state; should be called at the beginning of an epoch.
    with tf.name_scope("reset"):
#      pdb.set_trace()
      variables = (nest.flatten(state) +
                   x + constants)
#      print(variables)
#      print(x)
      # Empty array as part of the reset process.
      reset = [tf.variables_initializer(variables), fx_array.close(), x_array.close()]

    # Operator to update the parameters and the RNN state after our loop, but
    # during an epoch.
    with tf.name_scope("update"):
      update = (nest.flatten(_nested_assign(x, x_final)) +
                nest.flatten(_nested_assign(state, s_final)))

    # Log internal variables.
    for k, net in nets.items():
      

      print("Optimizer '{}' variables".format(k))
      print([op.name for op in snt.get_variables_in_module(net)])
    
    print(fx_final)
    

    return MetaLoss(loss, update, reset, fx_final, x_final, constants)

  def meta_minimize(self, make_loss, len_unroll, learning_rate=0.01, **kwargs):
    """Returns an operator minimizing the meta-loss.

    Args:
      make_loss: Callable which returns the optimizee loss; note that this
          should create its ops in the default graph.
      len_unroll: Number of steps to unroll.
      learning_rate: Learning rate for the Adam optimizer.
      **kwargs: keyword arguments forwarded to meta_loss.

    Returns:
      namedtuple containing (step, update, reset, fx, x)
    """
    info = self.meta_loss(make_loss, len_unroll, **kwargs)
    optimizer = tf.train.AdamOptimizer(learning_rate)
    regular = tf.contrib.layers.apply_regularization(tf.contrib.layers.l2_regularizer(1e-4), tf.trainable_variables())
#    gradients = optimizer.compute_gradients(info.loss+regular)
#    capped_gradients = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gradients if grad is not None]
#    train_op = optimizer.apply_gradients(capped_gradients)
#    train_op = optimizer.apply_gradients(gradients)
    step = optimizer.minimize(info.loss+regular)

    return MetaStep(step, *info[1:], self.fc_kernel, self.fc_bias, self.fc_va)
