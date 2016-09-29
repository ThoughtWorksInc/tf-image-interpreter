import tensorflow as tf
import numpy as np


class ConvNetMixin(object):
  def _conv_layer(self, bottom, filter_size, filter_num, scope_name, bottom_channel=None):
    if not bottom_channel:
      _, _, _, bottom_channel = bottom.get_shape().as_list()
    with tf.variable_scope(scope_name):
      kernel = tf.Variable(
        tf.truncated_normal([*filter_size, bottom_channel, filter_num], dtype=tf.float32, stddev=1e-1),
        trainable=False,
        name='weights'
      )

      conv = tf.nn.conv2d(bottom, kernel, [1, 1, 1, 1], padding='SAME')
      biases = tf.Variable(
        tf.constant(0.0, shape=[filter_num], dtype=tf.float32),
        trainable=True,
        name='bias'
      )
      out = tf.nn.bias_add(conv, biases)

      relu = tf.nn.relu(out, name=scope_name)
      return relu

  def _max_pool(self, bottom, name):
    return tf.nn.max_pool(
      bottom,
      ksize=[1, 2, 2, 1],
      strides=[1, 2, 2, 1],
      padding='SAME',
      name=name
    )


def load_params(param_path, sess):
  params = np.load(param_path)

  for v in tf.get_collection(tf.GraphKeys.VARIABLES, scope='vgg16'):
    print(v.name)
    mapping = {'weights:0': 'W', 'bias:0': 'b'}
    parts = v.name.split('/')
    param_file_name = parts[-2] + '_' + mapping[parts[-1]]
    sess.run(v.assign(params[param_file_name]))


class Vgg16(ConvNetMixin):
  def __init__(self):
    super().__init__()

  def build(self, image, scope_name='vgg16'):
    with tf.variable_scope(scope_name):
      self._build_graph(image)

  def _build_graph(self, image):
    self.conv1_1 = self._conv_layer(image, (3, 3), 64, "conv1_1", bottom_channel=3)
    self.conv1_2 = self._conv_layer(self.conv1_1, (3, 3), 64, "conv1_2")
    self.pool1 = self._max_pool(self.conv1_2, 'pool1')

    self.conv2_1 = self._conv_layer(self.pool1, (3, 3), 128, "conv2_1")
    self.conv2_2 = self._conv_layer(self.conv2_1, (3, 3), 128, "conv2_2")
    self.pool2 = self._max_pool(self.conv2_2, 'pool2')

    self.conv3_1 = self._conv_layer(self.pool2, (3, 3), 256, "conv3_1")
    self.conv3_2 = self._conv_layer(self.conv3_1, (3, 3), 256, "conv3_2")
    self.conv3_3 = self._conv_layer(self.conv3_2, (3, 3), 256, "conv3_3")
    self.pool3 = self._max_pool(self.conv3_3, 'pool3')

    self.conv4_1 = self._conv_layer(self.pool3, (3, 3), 512, "conv4_1")
    self.conv4_2 = self._conv_layer(self.conv4_1, (3, 3), 512, "conv4_2")
    self.conv4_3 = self._conv_layer(self.conv4_2, (3, 3), 512, "conv4_3")
    self.pool4 = self._max_pool(self.conv4_3, 'pool4')

    self.conv5_1 = self._conv_layer(self.pool4, (3, 3), 512, "conv5_1")
    self.conv5_2 = self._conv_layer(self.conv5_1, (3, 3), 512, "conv5_2")
    self.conv5_3 = self._conv_layer(self.conv5_2, (3, 3), 512, "conv5_3")
