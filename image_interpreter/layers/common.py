import tensorflow as tf
import numpy as np


class ConvNetMixin(object):
  def _conv_layer_with_relu(self, bottom, filter_size, filter_num, scope_name, bottom_channel=None):
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

      return out

  def _max_pool(self, bottom, name):
    return tf.nn.max_pool(
      bottom,
      ksize=[1, 2, 2, 1],
      strides=[1, 2, 2, 1],
      padding='SAME',
      name=name
    )


class AnchorTargetMixin(object):
  def generate_anchors(self, scales):
    base_size = 16
    ratios = np.array([0.5, 1, 2])
    base_anchor = np.array([1, 1, base_size, base_size]) - 1
    w, h, x_ctr, y_ctr = self._whctrs(base_anchor)
    w_ratios = np.round(np.sqrt(w * h / ratios))
    h_ratios = np.round(np.sqrt(ratios * w * h))
    anchors = np.vstack([self._mkanchors(w_ratios * scale, h_ratios * scale, x_ctr, y_ctr) for scale in scales])
    return anchors

  def _whctrs(self, anchor):
    w = anchor[2] - anchor[0] + 1
    h = anchor[3] - anchor[1] + 1
    x_ctr = anchor[0] + 0.5 * (w - 1)
    y_ctr = anchor[1] + 0.5 * (h - 1)
    return w, h, x_ctr, y_ctr

  def _mkanchors(self, w_ratios, h_ratios, x_ctr, y_ctr):
    ws = w_ratios[:, np.newaxis]
    hs = h_ratios[:, np.newaxis]
    anchors = np.hstack((x_ctr - 0.5 * (ws - 1),
                         y_ctr - 0.5 * (hs - 1),
                         x_ctr + 0.5 * (ws - 1),
                         y_ctr + 0.5 * (hs - 1)))
    return anchors


if __name__ == '__main__':
  anchors = AnchorTargetMixin().generate_anchors(np.array([8, 16, 32]))
  """
  [[ -84.  -36.   99.   51.]
   [ -56.  -56.   71.   71.]
   [ -36.  -84.   51.   99.]
   [-176.  -80.  191.   95.]
   [-120. -120.  135.  135.]
   [ -80. -176.   95.  191.]
   [-360. -168.  375.  183.]
   [-248. -248.  263.  263.]
   [-168. -360.  183.  375.]]
  """
  print(anchors)
