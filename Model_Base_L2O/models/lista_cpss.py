import numpy as np
from numpy import linalg as LA
import tensorflow.compat.v2 as tf
from tensorflow.compat.v2 import keras
from .utils import shrink_ss


class ListaCpssCell(keras.layers.Layer):
  """ListaCpss cell."""

  def __init__(self,
               A,
               W,
               step_size,
               theta,
               q,
               layer_id,
               name=None):
    super(ListaCpssCell, self).__init__(name=name)
    self._A = A.astype(np.float32)
    self.W = W
    self.step_size = step_size
    self.theta = theta
    self.layer_id = layer_id
    self.q = q
    self._M = self._A.shape[0]
    self._N = self._A.shape[1]

  def call(self, inputs):
    if self.layer_id == 0:
      res = inputs[:, :self._M]
      xk = 0.0
    else:
      y = inputs[:, :self._M]
      xk = inputs[:, -self._N:]
      res = y - tf.matmul(xk, self._A, transpose_b=True)
    output = xk + self.step_size * tf.matmul(res, self.W, transpose_b=False)
    output = shrink_ss(output, self.theta, self.q)
    return tf.concat([inputs, output], 1)


class ListaCpss(keras.Sequential):
  """ListaCpss model."""

  def __init__(self,
               A,
               T,
               lam,
               q_per_layer,
               maxq,
               share_W=False,
               D=None,
               name="ListaCpss"):
    super(ListaCpss, self).__init__(name=name)
    self._A = A.astype(np.float32)
    self._T = int(T)
    self._lam = lam
    self.share_W = share_W
    self._M = self._A.shape[0]
    self._N = self._A.shape[1]

    self._scale = 1.001 * np.linalg.norm(A, ord=2)**2
    self._theta = (self._lam / self._scale).astype(np.float32)

    self.q = np.clip([(t+1) * q_per_layer for t in range(self._T)], 0.0, maxq)
    print(self.q)

    if share_W:
      self._W = tf.Variable(self._A, trainable=True, name=name + "_W")
    else:
      self._W = [
          tf.Variable(self._A, trainable=True, name=name + "_W" + str(i + 1))
          for i in range(self._T)
      ]
    self.theta = [
        tf.Variable(self._theta, trainable=True, name=name + "_theta" + str(i + 1))
        for i in range(self._T)
    ]
    if share_W:
      self.step_size = [
          tf.Variable(1.0, trainable=True, name=name + "_step_size" + str(i + 1))
          for i in range(self._T)
      ]
    else:
      self.step_size = [1.0] * self._T

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

    cell = ListaCpssCell(self._A,
                         w,
                         self.step_size[layer_id],
                         self.theta[layer_id],
                         self.q[layer_id],
                         layer_id,
                         "ListaCpss_layer" + str(layer_id + 1))

    self.add(cell)

