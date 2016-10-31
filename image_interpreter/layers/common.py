import tensorflow as tf
import numpy as np


class ConvNetMixin(object):
  def _conv_layer_with_relu(self, bottom, filter_size, filter_num, scope_name, bottom_channel=None, padding='SAME'):
    out = self._conv_layer(bottom, filter_size, filter_num, scope_name, bottom_channel, padding)
    with tf.variable_scope(scope_name):
      relu = tf.nn.relu(out, name=scope_name)
      return relu

  def _conv_layer(self, bottom, filter_size, filter_num, scope_name, bottom_channel=None, padding='SAME'):
    if not bottom_channel:
      _, _, _, bottom_channel = bottom.get_shape().as_list()
    with tf.variable_scope(scope_name):
      kernel = tf.Variable(
        tf.truncated_normal([*filter_size, bottom_channel, filter_num], dtype=tf.float32, stddev=1e-1),
        trainable=False,
        name='weights'
      )

      conv = tf.nn.conv2d(bottom, kernel, [1, 1, 1, 1], padding=padding)
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
  def __init__(self, debug):
    self._anchors = self.generate_anchors(scales=(8, 16, 32))
    self._debug = debug
    self._feat_stride = 16

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

  def _generate_shifts(self, width, height):
    shift_x = tf.range(0, height) * self._feat_stride
    shift_y = tf.range(0, width) * self._feat_stride
    shift_x, shift_y = tf.meshgrid(shift_x, shift_y, indexing='ij')
    shifts = tf.transpose(tf.pack(
      [tf.reshape(shift_x, (-1,)),
       tf.reshape(shift_y, (-1,)),
       tf.reshape(shift_x, (-1,)),
       tf.reshape(shift_y, (-1,))],
      axis=0
    ))
    return shifts

  def _generate_all_anchors(self, shifts):
    num_anchors = self._anchors.shape[0]
    num_shifts = tf.shape(shifts)[0]
    all_anchors = (self._anchors.reshape(1, num_anchors, 4) +
                   tf.transpose(tf.reshape(shifts, (1, num_shifts, 4)), perm=(1, 0, 2)))
    all_anchors = tf.reshape(all_anchors, (num_shifts * num_anchors, 4))

    if self._debug:
      num_all_anchors = num_shifts * num_anchors
      tf.Print(num_all_anchors, [num_all_anchors], message='all anchor: ')
    return all_anchors

  def _calculate_overlaps(self, anchors, bboxes):
    if self._debug:
      bboxes = tf.Print(bboxes, [tf.shape(bboxes), bboxes], message='calculating overlaps, bbox: ', summarize=10)

    num_anchors = tf.shape(anchors)[0]
    num_bboxes = tf.shape(bboxes)[0]

    if self._debug:
      num_anchors = tf.Print(num_anchors, [num_anchors, num_bboxes], message='num_anchors')
      num_bboxes = tf.Print(num_bboxes, [num_bboxes], message='num_bboxes')

    ia = tf.constant(0)

    def cond_outer(i, overlaps, num_bboxes):
      return tf.less(i, num_anchors)

    def body_outer(i, overlaps, num_bboxes):
      # TODO: 看起来while loop内部引用外部变量，不能识别他们的依赖关系
      anchor = anchors[i]
      anchor = tf.reshape(anchor, (4,))
      if self._debug:
        anchor = tf.Print(anchor, [tf.shape(anchor), anchor], message='anchor')
      anchor_area = (anchor[2] - anchor[0] + 1) * (anchor[3] - anchor[1] + 1)
      ib = tf.constant(0)

      def cond_inner(j, row, anchor, anchor_area):
        return tf.less(j, num_bboxes)

      def body_inner(j, row, anchor, anchor_area):
        bbox = bboxes[j]
        uw = tf.minimum(anchor[2], bbox[2]) - tf.maximum(anchor[0], bbox[0])
        uh = tf.minimum(anchor[3], bbox[3]) - tf.maximum(anchor[1], bbox[1])

        if self._debug:
          uw = tf.Print(uw, [i, j, uw], message='union width')
          uh = tf.Print(uh, [i, j, uh], message='union height')

        result = tf.cond(
          tf.logical_and(tf.less(0., uw),
                         tf.less(0., uh)),
          lambda: tf.cast(uw * uh / (anchor_area + (bbox[2] - anchor[0] + 1) * (bbox[3] - bbox[1] + 1) - uw * uh),
                          tf.float32),
          lambda: tf.constant(0, dtype=tf.float32)
        )

        if self._debug:
          j = tf.Print(j, [j], message='j')
        return j + 1, tf.concat(0, [row, [result]]), anchor, anchor_area

      _, generated_row, _, _ = tf.while_loop(
        cond_inner,
        body_inner,
        loop_vars=[ib, tf.constant([]), anchor, anchor_area],
        shape_invariants=[ib.get_shape(), tf.TensorShape([None]), anchor.get_shape(), anchor_area.get_shape()]
      )
      if self._debug:
        i = tf.Print(i, [i], message='i')
      return i + 1, tf.concat(0, [overlaps, [generated_row]]), num_bboxes

    _, generated_overlaps, _ = tf.while_loop(
      cond_outer,
      body_outer,
      loop_vars=[ia, tf.reshape(tf.constant([]), (0, num_bboxes)), num_bboxes],
      shape_invariants=[ia.get_shape(), tf.TensorShape([None, None]), num_bboxes.get_shape()]
    )
    return generated_overlaps


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


def arg_sort_op(arr):
  args = arr.argsort()[::-1]
  return args
