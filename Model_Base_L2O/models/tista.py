import numpy as np
from numpy import linalg as LA
import tensorflow.compat.v2 as tf
from tensorflow.compat.v2 import keras
# from .utils import shrink_tista
import math


class TistaCell(keras.layers.Layer):
  """Tista cell."""

  def __init__(self,
               A,
               W,
               step_size,
               # theta,
               p,
               alpha2,
               sigma2,
               layer_id,
               name=None):
    super(TistaCell, self).__init__(name=name)
    self._A = A.astype(np.float32)
    self._taa = np.trace(np.matmul(A.T, A))
    pinv_A = np.linalg.pinv(A)
    self._tww = np.trace(np.matmul(pinv_A.T, pinv_A))
    self._p = p
    self._alpha2 = alpha2
    self._sigma2 = sigma2
    self.W = W
    self.step_size = step_size
    # self.theta = theta
    self.layer_id = layer_id
    self._M = self._A.shape[0]
    self._N = self._A.shape[1]

  def gauss(self, x, var):
    return tf.math.exp(-x**2.0/(2.0*var)) / tf.math.pow(2.0*math.pi*var,0.5)

  def shrink_tista(self, y, tau2):
    return (y * self._alpha2 / (self._alpha2 + tau2)) * self._p * \
            self.gauss(y, (self._alpha2+tau2)) / \
            ((1-self._p) * self.gauss(y, tau2) + self._p * self.gauss(y, (self._alpha2+tau2)))

  def eval_tau2(self, t):
    v2 = (tf.math.reduce_sum(t**2.0, axis=1, keepdims=True) - self._M*self._sigma2) / self._taa
    v2 = tf.clip_by_value(v2, clip_value_min=1e-9, clip_value_max=1000000)
    tau2 = (v2/self._N) * (self._N + (self.step_size**2 - 2.0*self.step_size)*self._M) + \
            self.step_size**2.0 * self._tww * self._sigma2 / self._N
    # tau2 = (tau2.expand(N, batch_size)).t()
    return tau2

  def call(self, inputs):
    y = inputs[:, :self._M]
    if self.layer_id == 0:
      xk = 0.0
      res = y
    else:
      xk = inputs[:, -self._N:]
      res = y - tf.matmul(xk, self._A, transpose_b=True)

    tau2 = self.eval_tau2(res)
    output = xk + self.step_size * tf.matmul(res, self.W, transpose_b=False)
    output = self.shrink_tista(output, tau2)

    return tf.concat([inputs, output], 1)


class Tista(keras.Sequential):
  """Lista model."""

  def __init__(self,
               A,
               T,
               lam,
               sigma2,
               share_W=False,
               D=None,
               name="Tista"):
    super(Tista, self).__init__(name=name)
    self._A = A.astype(np.float32)
    self._T = int(T)
    self._lam = lam
    self.sigma2 = sigma2
    self.share_W = share_W
    self._M = self._A.shape[0]
    self._N = self._A.shape[1]

    self._scale = 1.001 * np.linalg.norm(A, ord=2)**2
    # self._theta = (self._lam / self._scale).astype(np.float32)

    if share_W:
      self._W = tf.Variable(self._A, trainable=True, name=name + "_W")
    else:
      self._W = [
          tf.Variable(self._A, trainable=True, name=name + "_W" + str(i + 1))
          for i in range(self._T)
      ]
    # self.theta = [
    #     tf.Variable(self._theta, trainable=True, name=name + "_theta" + str(i + 1))
    #     for i in range(self._T)
    # ]
    self.step_size = [
        tf.Variable(1.0, trainable=True, name=name + "_step_size" + str(i + 1))
        for i in range(self._T)
    ]
    self.p = [
        tf.Variable(0.1, trainable=True, name=name + "_p" + str(i + 1))
        for i in range(self._T)
    ]
    self.alpha2 = [
        tf.Variable(1.0, trainable=True, name=name + "_alpha2" + str(i + 1))
        for i in range(self._T)
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

    cell = TistaCell(self._A,
                     w,
                     self.step_size[layer_id],
                     self.p[layer_id],
                     self.alpha2[layer_id],
                     self.sigma2,
                     # self.theta[layer_id],
                     layer_id,
                     "Tista_layer" + str(layer_id + 1))

    self.add(cell)

