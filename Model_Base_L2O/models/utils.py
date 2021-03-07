import tensorflow.compat.v2 as tf
from tensorflow.compat.v2 import keras
import tensorflow_probability as tfp


def shrink(data, theta):
  theta = keras.layers.ReLU()(theta)
  return tf.sign(data) * keras.layers.ReLU()(tf.abs(data) - theta)


def shrink_free(data, theta):
  return tf.sign(data) * keras.layers.ReLU()(tf.abs(data) - theta)


def shrink_lamp(r_, rvar_, lam_):
    """
    Implementation of thresholding neuron in Learned AMP model.
    """
    theta_ = tf.maximum(tf.sqrt(rvar_) * lam_, 0.0)
    xh_    = tf.sign(r_) * tf.maximum(tf.abs(r_) - theta_, 0.0)
    return xh_


def shrink_ss(inputs_, theta_, q, return_index=False):
    """
    Special shrink that does not apply soft shrinkage to entries of top q%
    magnitudes.
    :inputs_: TODO
    :thres_: TODO
    :q: TODO
    :returns: TODO
    """
    abs_ = tf.abs(inputs_)
    thres_ = tfp.stats.percentile(abs_, 100.0-q, axis=1, keepdims=True)

    """
    Entries that are greater than thresholds and in the top q% simultnaneously
    will be selected into the support, and thus will not be sent to the
    shrinkage function.
    """
    index_ = tf.logical_and(abs_ > theta_, abs_ > thres_)
    index_ = tf.cast(index_, tf.float32)
    """Stop gradient at index_, considering it as constant."""
    index_ = tf.stop_gradient(index_)
    cindex_ = 1.0 - index_ # complementary index

    output = (tf.multiply(index_, inputs_) +
              shrink_free(tf.multiply(cindex_, inputs_), theta_ ))
    if return_index:
        return output, cindex_
    else:
        return output

