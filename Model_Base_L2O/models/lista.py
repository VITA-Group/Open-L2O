import numpy as np
from numpy import linalg as LA
import tensorflow.compat.v2 as tf
from tensorflow.compat.v2 import keras
from .utils import shrink_free


class ListaCell(keras.layers.Layer):
  """Lista cell."""

  def __init__(self,
               A,
               w_1,
               W,
               share_W,
               step_size,
               theta,
               layer_id,
               name=None):
    super(ListaCell, self).__init__(name=name)
    self._A = A.astype(np.float32)
    self.w_1 = w_1
    if layer_id != 0:
      self.w_2 = W
    self.share_W = share_W
    self.step_size = step_size
    self.theta = theta
    self.layer_id = layer_id
    self._M = self._A.shape[0]
    self._N = self._A.shape[1]

  def call(self, inputs):
    # output = B * y 
    output = tf.matmul(inputs[:, :self._M], self.w_1, transpose_b=True)
    # if the current layer is not the first layer, take the ouput of the
    # last layer as the input.
    if self.layer_id != 0:
      inputs_ = inputs[:, -self._N:]

    if self.layer_id != 0:
      if self.share_W:
        inputs_ = inputs_ * self.step_size[self.layer_id - 1]
      output = output + tf.matmul(inputs_, self.w_2, transpose_b=True)
    output = shrink_free(output, self.theta)
    return tf.concat([inputs, output], 1)


class Lista(keras.Sequential):
  """Lista model."""

  def __init__(self,
               A,
               T,
               lam,
               share_W=False,
               D=None,
               name="Lista"):
    super(Lista, self).__init__(name=name)
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
    if share_W:
      self._W = tf.Variable(_W, trainable=True, name=name + "_W")
    else:
      self._W = [
          tf.Variable(_W, trainable=True, name=name + "_W" + str(i + 1))
          for i in range(1, self._T)
      ]
    self.theta = [
        tf.Variable(
            self._theta, trainable=True, name=name + "_theta" + str(i + 1))
        for i in range(self._T)
    ]
    if share_W:
      self.step_size = [
          tf.Variable(1.0, trainable=True,
                      name=name + "_step_size" + str(i + 1)) for i in range(self._T)
      ]
    else:
      self.step_size = [1.0] * self._T

    self.w_1 = tf.Variable(self._B, trainable=True, name=name + "_B")
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
    elif layer_id != 0:
      w = self._W[layer_id - 1]
    else:
      w = None
    if self._D is None:
      w_d = self._W_D
      F = 0
    else:
      F = self._D.shape[0]
      if layer_id == self._T - 1:
        w_d = self._W_D
      else:
        w_d = self._W_D_constant

    w_1 = self.w_1
    cell = ListaCell(self._A, w_1, w, self.share_W,
                     self.step_size[layer_id],
                     self.theta[layer_id],
                     layer_id, "Lista_layer" + str(layer_id + 1))

    self.add(cell)

