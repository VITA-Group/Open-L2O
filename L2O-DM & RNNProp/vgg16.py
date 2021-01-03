import tensorflow as tf 
def conv_layer(x, num_filters, name, filter_height = 3, filter_width = 3, stride = 1, padding = 'SAME'):

  input_channels = int(x.get_shape()[-1])

  with tf.variable_scope(name) as scope:
    W = tf.get_variable('weights', shape = [filter_height, filter_width, input_channels, num_filters], 
      initializer = tf.random_normal_initializer(mean = 0, stddev = 0.01))
    b = tf.get_variable('biases', shape = [num_filters], 
      initializer = tf.constant_initializer(0.0))
  
  conv = tf.nn.conv2d(x, W, strides = [1, stride, stride, 1], padding = padding)
  
  z = tf.nn.bias_add(conv, b)

  return tf.nn.relu(z, name = scope.name)

def fc_layer(x, input_size, output_size, name, activation = 'relu'):

  with tf.variable_scope(name) as scope:
    W = tf.get_variable('weights', shape = [input_size, output_size], 
      initializer = tf.random_normal_initializer(mean = 0, stddev = 0.01))
    b = tf.get_variable('biases', shape = [output_size], 
      initializer = tf.constant_initializer(1.0))

  z = tf.nn.bias_add(tf.matmul(x, W), b, name = scope.name)
  
  if activation == 'relu':
    # Apply ReLu non linearity.
    return tf.nn.relu(z, name = scope.name)
  elif activation == 'softmax':
    return tf.nn.softmax(z, name = scope.name)
  else:
    return z

def max_pool(x, name, filter_height = 2, filter_width = 2, stride = 2, padding = 'SAME'):
  return tf.nn.max_pool(x, ksize = [1, filter_height, filter_width, 1],
            strides = [1, stride, stride, 1], padding = padding,
            name = name)

def dropout(x, keep_prob):
  return tf.nn.dropout(x, keep_prob = keep_prob)

class VGG16(object):

  def __init__(self, keep_prob, num_classes):

    self.KEEP_PROB = keep_prob
    self.NUM_CLASSES = num_classes

  def _build_model(self, inputs):

    # Block 1
    block1_conv1 = conv_layer(inputs, 64, name='block1_conv1')
    block1_conv2 = conv_layer(block1_conv1, 64, name='block1_conv2')
    block1_pool = max_pool(block1_conv2, name='block1_pool')

    # Block 2
    block2_conv1 = conv_layer(block1_pool, 128, name='block2_conv1')
    block2_conv2 = conv_layer(block2_conv1, 128, name='block2_conv2')
    block2_pool = max_pool(block2_conv2, name='block2_pool')

    # Block 3
    block3_conv1 = conv_layer(block2_pool, 256, name='block3_conv1')
    block3_conv2 = conv_layer(block3_conv1, 256, name='block3_conv2')
    block3_conv3 = conv_layer(block3_conv2, 256, name='block3_conv3')
    block3_pool = max_pool(block3_conv3, name='block3_pool')

    # Block 4
    block4_conv1 = conv_layer(block3_pool, 512, name='block4_conv1')
    block4_conv2 = conv_layer(block4_conv1, 512, name='block4_conv2')
    block4_conv3 = conv_layer(block4_conv2, 512, name='block4_conv3')
    block4_pool = max_pool(block4_conv3, name = 'block4_pool')

    # Block 5
    block5_conv1 = conv_layer(block4_pool, 512, name='block5_conv1')
    block5_conv2 = conv_layer(block5_conv1, 512, name='block5_conv2')
    block5_conv3 = conv_layer(block5_conv2, 512, name='block5_conv3')
    block5_pool = max_pool(block5_conv3, name='block5_pool')

    # Full connection layers

    # In the original paper implementaion this will be:
    #flattened = tf.reshape(block5_pool, [-1, 7*7*512]) 
    #fc1 = fc_layer(flattened, 7*7*512, 7*7*512, name = 'fc1')
    flattened = tf.reshape(block5_pool, [-1, 1*1*512]) 
    # fc1 = fc_layer(flattened, 1*1*512, 1*1*512, name = 'fc1', activation = 'relu')
    # dropout1 = dropout(fc1, self.KEEP_PROB)

    # In the original paper implementaion this will be:
    #fc2 = fc_layer(dropout1, 7*7*512, 7*7*512, name = 'fc1')
    # fc2 = fc_layer(dropout1, 1*1*512, 1*1*512, name = 'fc2', activation = 'relu')
    # dropout2 = dropout(fc2, self.KEEP_PROB)

    # In the original paper implementaion this will be:
    #self.fc3 = fc_layer(dropout2, 7*7*512, self.NUM_CLASSES, name = 'fc3', relu = False)
    fc = fc_layer(flattened, 1*1*512, self.NUM_CLASSES, name = 'fc', activation = 'softmax')
    return fc