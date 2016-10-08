import numpy as np
import tensorflow as tf
from image_interpreter.layers.common import ConvNetMixin


def load_feature_layer_params(param_path, sess):
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
    self.conv1_1 = self._conv_layer_with_relu(image, (3, 3), 64, "conv1_1", bottom_channel=3)
    self.conv1_2 = self._conv_layer_with_relu(self.conv1_1, (3, 3), 64, "conv1_2")
    self.pool1 = self._max_pool(self.conv1_2, 'pool1')

    self.conv2_1 = self._conv_layer_with_relu(self.pool1, (3, 3), 128, "conv2_1")
    self.conv2_2 = self._conv_layer_with_relu(self.conv2_1, (3, 3), 128, "conv2_2")
    self.pool2 = self._max_pool(self.conv2_2, 'pool2')

    self.conv3_1 = self._conv_layer_with_relu(self.pool2, (3, 3), 256, "conv3_1")
    self.conv3_2 = self._conv_layer_with_relu(self.conv3_1, (3, 3), 256, "conv3_2")
    self.conv3_3 = self._conv_layer_with_relu(self.conv3_2, (3, 3), 256, "conv3_3")
    self.pool3 = self._max_pool(self.conv3_3, 'pool3')

    self.conv4_1 = self._conv_layer_with_relu(self.pool3, (3, 3), 512, "conv4_1")
    self.conv4_2 = self._conv_layer_with_relu(self.conv4_1, (3, 3), 512, "conv4_2")
    self.conv4_3 = self._conv_layer_with_relu(self.conv4_2, (3, 3), 512, "conv4_3")
    self.pool4 = self._max_pool(self.conv4_3, 'pool4')

    self.conv5_1 = self._conv_layer_with_relu(self.pool4, (3, 3), 512, "conv5_1")
    self.conv5_2 = self._conv_layer_with_relu(self.conv5_1, (3, 3), 512, "conv5_2")
    self.conv5_3 = self._conv_layer_with_relu(self.conv5_2, (3, 3), 512, "conv5_3")
