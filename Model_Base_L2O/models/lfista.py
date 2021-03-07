import numpy as np
from numpy import linalg as LA
import tensorflow.compat.v2 as tf
from tensorflow.compat.v2 import keras
from .utils import shrink_free
import math


class LfistaCell(keras.layers.Layer):
  """Lfista cell."""

  def __init__(self,
               A,
               Wg,
               Wm,
               We,
               theta,
               layer_id,
               name=None):
    super(LfistaCell, self).__init__(name=name)
    self._A = A.astype(np.float32)
    self.Wg = Wg
    self.Wm = Wm
    self.We = We
    self.theta = theta
    self.layer_id = layer_id
    self._M = self._A.shape[0]
    self._N = self._A.shape[1]

  def call(self, inputs):
    # output = B * y 
    output = tf.matmul(inputs[:, :self._M], self.We, transpose_b=True)
    # if the current layer is not the first layer, take the ouput of the
    # last layer as the input.
    if self.layer_id > 0:
      inputs_ = inputs[:, -self._N:]
      output += tf.matmul(inputs_, self.Wg, transpose_b=True)
    if self.layer_id > 1:
      prev_inputs_ = inputs[:, -self._N*2:-self._N]
      output += tf.matmul(prev_inputs_, self.Wm, transpose_b=True)

    output = shrink_free(output, self.theta)
    return tf.concat([inputs, output], 1)


class Lfista(keras.Sequential):
  """Lfista model."""

  def __init__(self,
               A,
               T,
               lam,
               share_W=False,
               D=None,
               name="Lfista"):
    super(Lfista, self).__init__(name=name)
    self._A = A.astype(np.float32)
    self._T = int(T)
    self._lam = lam
    self.share_W = share_W
    self._M = self._A.shape[0]
    self._N = self._A.shape[1]

    self._scale = 1.001 * np.linalg.norm(A, ord=2)**2
    self._theta = (self._lam / self._scale).astype(np.float32)

    self._B = (np.transpose(self._A) / self._scale).astype(np.float32)
    _W = np.eye(self._N, dtype=np.float32) - np.matmul(self._B, self._A)

    _tk = [1.0, 1.0]
    self._mk = []
    for i in range(self._T):
        _tk.append((1 + math.sqrt(1 + 4*_tk[-1]**2.0)) / 2)
        self._mk.append((_tk[-2] - 1) / _tk[-1])

    self._Wg = [
        tf.Variable(_W * (1 + self._mk[i]), trainable=True, name=name + "_Wg" + str(i + 1))
        for i in range(1, self._T)
    ]
    self._Wm = [
        tf.Variable(- self._mk[i] * _W, trainable=True, name=name + "_Wm" + str(i + 1))
        for i in range(1, self._T)
    ]
    self.theta = [
        tf.Variable(
            self._theta, trainable=True, name=name + "_theta" + str(i + 1))
        for i in range(self._T)
    ]

    self._We = [
        tf.Variable(self._B, trainable=True, name=name + "_We" + str(i + 1))
    ] * self._T

    if D is not None:
      self._D = D
      self._W_D_constant = tf.Variable(self._D, trainable=False, name=name + "_W_D_constant")
      self._W_D = tf.Variable(self._D, trainable=True, name=name + "_W_D")
    else:
      self._D = None
      self._W_D = None

  def create_cell(self, layer_id):
    if layer_id != 0:
      Wg = self._Wg[layer_id - 1]
      Wm = self._Wm[layer_id - 1]
    else:
      Wg = None
      Wm = None

    if self._D is None:
      w_d = self._W_D
      F = 0
    else:
      F = self._D.shape[0]
      if layer_id == self._T - 1:
        w_d = self._W_D
      else:
        w_d = self._W_D_constant

    We = self._We[layer_id]
    cell = LfistaCell(self._A,
                      Wg,
                      Wm,
                      We,
                      self.theta[layer_id],
                      layer_id,
                      "Lfista_layer" + str(layer_id + 1))

    self.add(cell)

