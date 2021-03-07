import numpy as np
from numpy import linalg as LA
import tensorflow.compat.v2 as tf
from tensorflow.compat.v2 import keras
from .utils import shrink_free


class StepListaCell(keras.layers.Layer):
  """Lista cell."""

  def __init__(self,
               A,
               lam,
               step_size,
               layer_id,
               name=None):
    super(StepListaCell, self).__init__(name=name)
    self._A = A
    self._lam = lam
    self.step_size = step_size
    self.layer_id = layer_id
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
    output = xk + self.step_size * tf.matmul(res, self._A, transpose_b=False)
    theta = self._lam * self.step_size
    output = shrink_free(output, theta)
    return tf.concat([inputs, output], 1)


class StepLista(keras.Sequential):
  """Lista model."""

  def __init__(self,
               A,
               T,
               lam,
               D=None,
               name="StepLista"):
    super(StepLista, self).__init__(name=name)
    self._A = tf.constant(A.astype(np.float32), name=name+'_A_const')
    self._T = int(T)
    self._lam = lam
    self._M = self._A.shape[0]
    self._N = self._A.shape[1]
    self._name = name

    self._scale = 1.001 * np.linalg.norm(A, ord=2)**2
    self.step_size = [
        tf.Variable(1.0 / self._scale, trainable=True, dtype=tf.float32,
                    name=name + "_step_size" + str(i + 1))
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
    # if self._D is None:
    #   w_d = self._W_D
    #   F = 0
    # else:
    #   F = self._D.shape[0]
    #   if layer_id == self._T - 1:
    #     w_d = self._W_D
    #   else:
    #     w_d = self._W_D_constant

    cell = StepListaCell(self._A,
                         self._lam,
                         self.step_size[layer_id],
                         layer_id,
                         self._name+"_layer"+str(layer_id + 1))

    self.add(cell)

