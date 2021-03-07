import os
import numpy as np
import tensorflow.compat.v2 as tf


class MSE(tf.keras.losses.Loss):

  def __init__(self, N, F=None, **kwargs):
    super(MSE, self).__init__(**kwargs)
    self.F = F
    self.N = N

  def call(self, y_true, y_pred):
    # return tf.keras.losses.MSE(y_true, y_pred[:, -500:]) * 500
    if self.F is None:
      loss = tf.nn.l2_loss(y_pred[:, -self.N:] - y_true[:, -self.N:])
    else:  # Switched the notation of N and F here
      loss = tf.nn.l2_loss(y_pred[:, -self.N:] - y_true[:, -self.N:]) + 0.001 * tf.reduce_sum(
          tf.abs(y_pred[:, -self.F - self.N:-self.N]))
    return loss


class LassoLoss(tf.keras.losses.Loss):

  def __init__(self, A, lam, N, F=None, **kwargs):
    super(LassoLoss, self).__init__(**kwargs)
    # self._A = tf.Variable(A, trainable=False, name='A_const_loss')
    self._A = A
    self._lam = lam
    self.F = F
    self.N = N

  def call(self, y_true, y_pred):
    sparse_pred = y_pred[:, -self.N:]
    measure_pred = tf.matmul(y_pred[:, -self.N:], self._A, transpose_b=True)
    l2_loss = tf.nn.l2_loss(measure_pred - y_true[:, :-self.N])
    l1_loss = tf.reduce_sum(tf.abs(sparse_pred))
    return 0.5 * l2_loss + self._lam * l1_loss


class LassoObjective(tf.keras.metrics.Mean):

  def __init__(self, name, A, lam, M, N, layer_id=-1, dtype=None):
    super(LassoObjective, self).__init__(name=name, dtype=dtype)
    # self._A = tf.Variable(A, trainable=False, name='A_const_obj')
    self._A = A
    self._lam = lam
    self.M = M
    self.N = N
    self.layer_id = layer_id

  def update_state(self, y_true, y_pred, sample_weight=None):
    if self.layer_id == -1:
      sparse_pred = y_pred[:, -self.N:]
    elif self.layer_id >= 0:
        idx = self.M + self.layer_id * self.N
        sparse_pred = y_pred[:, idx:idx+self.N]
    else:
      raise ValueError('Invalid layer index {}'.format(self.layer_id))
    measure_pred = tf.matmul(sparse_pred, self._A, transpose_b=True)
    l2_loss = tf.reduce_sum(tf.square(measure_pred - y_true[:, :self.M]), axis=-1)
    l1_loss = tf.reduce_sum(tf.abs(sparse_pred), axis=-1)
    lasso_obj = 0.5 * l2_loss + self._lam * l1_loss
    return super(LassoObjective, self).update_state(lasso_obj, sample_weight=sample_weight)


class PSNR(tf.keras.metrics.Mean):

  def update_state(self, y_true, y_pred, sample_weight=None):
    max_val = 255.0
    mse = tf.reduce_mean(tf.square(y_true - tf.clip_by_value(y_pred, 0, 255)))
    psnr_val = 20 * tf.math.log(max_val / tf.sqrt(mse)) / tf.math.log(10.0)
    # psnr_val = 10 * tf.math.log(max_val**2 / mse) / tf.math.log(10.0)
    return super(PSNR, self).update_state([psnr_val], sample_weight=sample_weight)


class NMSE(tf.keras.metrics.Mean):

  def __init__(self, name, N, dtype=None):
    super(NMSE, self).__init__(name=name, dtype=dtype)
    self.N = N

  def update_state(self, y_true, y_pred, sample_weight=None):
    mse = tf.reduce_mean(tf.square(y_true[:, -self.N:] - y_pred[:, -self.N:]), axis=-1)+1e-10
    nmse_denom = tf.reduce_mean(tf.square(y_true[:, -self.N:]), axis=-1)+1e-10
    nmse = mse / nmse_denom
    nmse_val = 10.0 * tf.math.log(nmse) / tf.math.log(10.0)
    return super(NMSE, self).update_state(nmse_val, sample_weight=sample_weight)


class EvalNMSE(tf.keras.metrics.Mean):

  def __init__(self, name, M, N, interval, layer_id=-1, dtype=None):
    super(EvalNMSE, self).__init__(name=name, dtype=dtype)
    self.M = M
    self.N = N
    self.interval = interval
    self.layer_id = layer_id

  def update_state(self, y_true, y_pred, sample_weight=None):
    if self.layer_id == -1:
      sparse_pred = y_pred[:, -self.N:]
    elif self.layer_id >= 0:
      idx_end = self.M + (self.layer_id + 1) * self.interval
      sparse_pred = y_pred[:, idx_end - self.N:idx_end]
    else:
      raise ValueError('Invalid layer index {}'.format(self.layer_id))
    mse = tf.reduce_mean(tf.square(y_true[:, -self.N:] - sparse_pred), axis=-1)+1e-10
    nmse_denom = tf.reduce_mean(tf.square(y_true[:, -self.N:]), axis=-1)+1e-10
    nmse = mse / nmse_denom
    nmse_val = 10.0 * tf.math.log(nmse) / tf.math.log(10.0)
    return super(EvalNMSE, self).update_state(nmse_val, sample_weight=sample_weight)


class Adam(tf.keras.optimizers.Adam):

  def __init__(self, var_list, freeze_layer=False, **kwargs):
    self.var_list = var_list
    self.freeze_layer = freeze_layer
    super(Adam, self).__init__(**kwargs)

  def apply_gradients(self, grads_and_vars, name=None,
                      all_reduce_sum_gradients=True):
    grads_and_vars_multiplied = []
    for g, v in grads_and_vars:
      if g is None:
        continue
      if self.freeze_layer:
        if self.var_list[v.name] == 0:
          grads_and_vars_multiplied.append((g, v))
      else:
        g_mul = g * 0.3**self.var_list[v.name]
        grads_and_vars_multiplied.append((g_mul, v))
    super(Adam, self).apply_gradients(grads_and_vars_multiplied, name,
                                      all_reduce_sum_gradients)


# def save_partial(checkpoint_dir, arch):
#   arch_str = []
#   for i in range(len(arch)):
#     arch_str.append(str(arch[i][0])+arch[i][1])
#   arch_str = '/'.join(arch_str)
#   model_dir = os.path.join(checkpoint_dir, arch_str)
#   return os.path.join(model_dir, 'model')


def save_partial(checkpoint_dir, layer):
  model_dir = os.path.join(checkpoint_dir, 'layer_' + str(layer + 1))
  return os.path.join(model_dir, 'model')


# def check_and_load_partial(checkpoint_dir, arch):
#   prev_model = None
#   model_dir = checkpoint_dir
#   for i in range(len(arch)):
#     model_dir = os.path.join(model_dir, str(arch[i][0])+arch[i][1])
#     model = tf.train.latest_checkpoint(model_dir)
#     if not model:
#       return prev_model, i
#     prev_model = model
#   return prev_model, len(arch)


def check_and_load_partial(checkpoint_dir, num_layers):
  prev_model = None
  model_dir = checkpoint_dir
  for i in range(num_layers):
    model_dir = os.path.join(checkpoint_dir, 'layer_' + str(i + 1))
    model = tf.train.latest_checkpoint(model_dir)
    if not model:
      return prev_model, i
    prev_model = model
  return prev_model, num_layers


def im2cols(im):
  h, w = im.shape
  cols = []
  for i in range(0, h, 16):
    for j in range(0, w, 16):
      cols.append(im[i:i+16, j:j+16])
  cols = np.array(cols).reshape(-1, 256)
  return cols


def col2im(cols):
  hw = cols.shape[0]
  h = w = int(np.sqrt(hw)) * 16
  im_rec = np.zeros((h, w))
  k = 0
  for i in range(0, h, 16):
    for j in range(0, w, 16):
      im_rec[i:i+16, j:j+16] += cols[k].reshape(16, 16)
      k += 1
  return im_rec

