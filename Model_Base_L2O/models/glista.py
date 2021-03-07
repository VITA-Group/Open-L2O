import numpy as np
from numpy import linalg as LA
import tensorflow.compat.v2 as tf
from tensorflow.compat.v2 import keras
from .utils import shrink_ss


class GlistaCell(keras.layers.Layer):
  """Glista cell."""

  def __init__(self,
               A,
               W,
               step_size,
               theta,
               q,
               D_gain,
               gain_func,
               alti,
               alti_over,
               layer_id,
               name=None):
    super(GlistaCell, self).__init__(name=name)
    self._A = A
    self._W = W  # A.T / scale
    self._M = self._A.shape[0]
    self._N = self._A.shape[1]
    self.step_size = step_size
    self.theta = theta
    self.q = q
    self.D_gain = D_gain
    self.gain_func = gain_func
    self.alti = alti
    self.alti_over = alti_over
    self.layer_id = layer_id

    def reweight_function(x, D, theta, alti_):
      reweight = 1.0 + alti_*theta*tf.nn.relu(1-tf.nn.relu(D*tf.abs(x)))
      return reweight
    def reweight_inverse(x, D, theta, alti):
      reweight = 1.0 + alti * theta * 0.2/(0.001 + tf.abs(D*x))
      return reweight
    def reweight_exp(x, D, theta, alti):
      reweight = 1.0 + alti * theta * tf.exp(-D * tf.abs(x))
      return reweight
    def reweight_sigmoid(x, D, theta, alti):
      reweight = 1.0 + alti * theta * tf.nn.sigmoid(-D * tf.abs(x))
      return reweight

    def gain(x, D, theta, alti, gain_func):
      if gain_func == 'relu':
          use_function = reweight_function
      if gain_func == 'inv':
          use_function = reweight_inverse
      elif gain_func == 'exp':
          use_function = reweight_exp
      elif gain_func == 'sigm':
          use_function = reweight_sigmoid
      elif gain_func == 'none':
          return 1.0
      return use_function(x, D, theta, alti)
    self.gain = gain

    def overshoot(alti, Part_1, Part_2):
      return 1.0 - alti * Part_1 * Part_2
    self.overshoot = overshoot

  def call(self, inputs):
    y = inputs[:, :self._M]

    if self.layer_id == 0:
      xk = 0.0
      Part_2_inv = self.theta
    else:
      xk = inputs[:, -self._N:]
      cindex = inputs[:, -(self._N+self._N):-self._N]
      Part_2_inv = self.theta * cindex

    # Gain gate
    in_ = self.gain(xk, self.D_gain, 1.0, self.alti, self.gain_func)

    res = y - tf.matmul(in_ * xk, self._A, transpose_b=True)
    zk = in_ * xk + self.step_size * tf.matmul(res, self._W, transpose_b=False)
    xk_title, cindex = shrink_ss(zk, self.theta, self.q, return_index=True)

    # Overshoot gate
    Part_1_inv = 1.0 / (tf.abs(xk_title - xk) + 0.1)
    g_ = self.overshoot(self.alti_over, Part_1_inv, Part_2_inv)

    output = g_ * xk_title + (1 - g_) * xk

    return tf.concat([inputs, cindex, output], 1)


class Glista(keras.Sequential):
  """Glista model."""

  def __init__(self,
               A,
               T,
               lam,
               q_per_layer,
               maxq,
               share_W=False,
               D=None,
               alti=5,
               gain_func='relu',
               name="Glista"):
    super(Glista, self).__init__(name=name)
    self._A = tf.constant(A.astype(np.float32), name=name+'_A_const')
    self._T = int(T)
    self._lam = lam
    self.share_W = share_W
    self.gain_func = gain_func
    self._alti = alti
    self._M = self._A.shape[0]
    self._N = self._A.shape[1]

    self._scale = 1.001 * np.linalg.norm(A, ord=2)**2
    self._theta = (self._lam / self._scale).astype(np.float32)

    self.q = np.clip([(t+1) * q_per_layer for t in range(self._T)], 0.0, maxq)
    print(self.q)

    _W = self._A / self._scale
    _D_gain = np.ones((1, self._N), dtype=np.float32)

    if share_W:
      self._W = tf.Variable(_W, trainable=True, name=name + "_W")
    else:
      self._W = [
          tf.Variable(_W, trainable=True, name=name + "_W" + str(i + 1))
          for i in range(self._T)
      ]

    self.theta = [
        tf.Variable(self._theta, trainable=True, name=name + "_theta" + str(i + 1))
        for i in range(self._T)
    ]
    if share_W:
      self.step_size = [
          tf.Variable(1.0, trainable=True,
                      name=name + "_step_size" + str(i + 1)) for i in range(self._T)
      ]
    else:
      self.step_size = [1.0] * self._T

      self.alti = [
          tf.Variable(self._alti, trainable=True,
                      name=name + "_alti" + str(i + 1)) for i in range(self._T)
      ]
      self.alti_over = [
          tf.Variable(self._alti, trainable=True,
                      name=name + "_alti_over" + str(i + 1)) for i in range(self._T)
      ]
      self.D_gain = [
          tf.Variable(_D_gain, trainable=True,
                      name=name + "_D_gain" + str(i + 1)) for i in range(self._T)
      ]

    if D is not None:
      self._D = D
      self._W_D_constant = tf.Variable(self._D, trainable=False, name=name + "_W_D_constant")
      self._W_D = tf.Variable(self._D, trainable=True, name=name + "_W_D")
    else:
      self._D = None
      self._W_D = None

  def create_cell(self, layer_id):
    if self.share_W:
      w = self._W
    else:
      w = self._W[layer_id]

    if self._D is None:
      w_d = self._W_D
      F = 0
    else:
      F = self._D.shape[0]
      if layer_id == self._T - 1:
        w_d = self._W_D
      else:
        w_d = self._W_D_constant

    cell = GlistaCell(self._A,
                      self._W[layer_id],
                      self.step_size[layer_id],
                      self.theta[layer_id],
                      self.q[layer_id],
                      self.D_gain[layer_id],
                      self.gain_func,
                      self.alti[layer_id],
                      self.alti_over[layer_id],
                      layer_id,
                      "Glista_layer" + str(layer_id + 1))

    self.add(cell)

