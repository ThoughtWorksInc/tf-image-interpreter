import numpy as np
import tensorflow as tf
from image_interpreter.layers.common import AnchorTargetMixin


class RpnData(AnchorTargetMixin):
  def __init__(self, debug=False):
    self._anchors = self.generate_anchors(np.array([8, 16, 32]))
    self._feat_stride = 16
    self._debug = debug

  def generate(self, image, scale, bboxes):
    shape = tf.shape(image)
    # TODO: NotImplementedError: Negative start indices are not currently supported
    # height, width = shape[-2:]
    # height, width = shape[-2:]
    height = shape[2]
    width = shape[3]

    if self._debug:
      height = tf.Print(height, [height], message='image height: ')
      width = tf.Print(width, [width], message='image width: ')

    anchors = self._generate_valid_anchors(width, height)
    overlaps = self._calculate_overlaps(anchors, bboxes)

    labels = self._generate_labels(overlaps)

    return labels

  def _generate_labels(self, overlaps):
    labels = tf.Variable(tf.ones(shape=(tf.shape(overlaps)[0],), dtype=tf.float32) * -1, trainable=False,
                         validate_shape=False)
    gt_max_overlaps = tf.arg_max(overlaps, dimension=0)
    anchor_max_overlaps = tf.arg_max(overlaps, dimension=1)
    mask = tf.one_hot(anchor_max_overlaps, tf.shape(overlaps)[1], on_value=True, off_value=False)
    max_overlaps = tf.boolean_mask(overlaps, mask)
    if self._debug:
      max_overlaps = tf.Print(max_overlaps, [max_overlaps])
    labels = tf.scatter_update(labels, gt_max_overlaps, tf.ones((tf.shape(gt_max_overlaps)[0],)))
    # TODO: extract config object
    over_threshold_mask = tf.reshape(tf.where(max_overlaps > 0.5), (-1,))
    if self._debug:
      over_threshold_mask = tf.Print(over_threshold_mask, [over_threshold_mask], message='over threshold index : ')
    labels = tf.scatter_update(labels, over_threshold_mask, tf.ones((tf.shape(over_threshold_mask)[0],)))
    # TODO: support clobber positive in the origin implement
    below_threshold_mask = tf.reshape(tf.where(max_overlaps < 0.3), (-1,))
    if self._debug:
      below_threshold_mask = tf.Print(below_threshold_mask, [below_threshold_mask], message='below threshold index : ')
    labels = tf.scatter_update(labels, below_threshold_mask, tf.zeros((tf.shape(below_threshold_mask)[0],)))
    return labels

  def _generate_valid_anchors(self, width, height):
    shifts = self._generate_shifts(width, height)
    all_anchors = self._generate_all_anchors(shifts)
    anchors = self._filter_inside_anchors(all_anchors, height, width)
    return anchors

  def _filter_inside_anchors(self, all_anchors, height, width):
    # filter anchors
    inds_inside = tf.where(
      (all_anchors[:, 0] > 0) &
      (all_anchors[:, 1] > 0) &
      (all_anchors[:, 2] < width) &
      (all_anchors[:, 3] < height)
    )
    if self._debug:
      inds_inside = tf.Print(inds_inside, [tf.shape(inds_inside)], message='inside anchors: ')
    anchors = tf.gather(all_anchors, inds_inside)
    return anchors

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

  def _calculate_overlaps(self, anchors, bboxes):
    if self._debug:
      bboxes = tf.Print(bboxes, [tf.shape(bboxes), bboxes], message='calculating overlaps, bbox: ', summarize=10)

    num_anchors = tf.shape(anchors)[0]
    num_bboxes = tf.shape(bboxes)[0]

    if self._debug:
      num_anchors = tf.Print(num_anchors, [num_anchors, num_bboxes])
      num_bboxes = tf.Print(num_bboxes, [num_bboxes])

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
          tf.logical_and(tf.less(0, uw),
                         tf.less(0, uh)),
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
      loop_vars=[ia, tf.reshape(tf.constant([]), (0, 6)), num_bboxes],
      shape_invariants=[ia.get_shape(), tf.TensorShape([None, 6]), num_bboxes.get_shape()]
    )
    return generated_overlaps


if __name__ == '__main__':
  with tf.Session() as sess:
    rpn_data = RpnData(debug=True)
    test_image = tf.reshape(tf.constant(np.ones((600, 400))), (1, 1, 600, 400))
    fake_bboxes = tf.constant([
      [10, 10, 150, 150],
      [70, 10, 150, 50],
      [10, 70, 50, 150],
      [150, 10, 70, 50],
      [10, 150, 50, 70],
      [10, 10, 390, 590],
    ], dtype=tf.int32)
    test_overlaps = rpn_data.generate(test_image, None, bboxes=fake_bboxes)
    sess.run(tf.initialize_all_variables())
    print_op = tf.Print(test_overlaps, [tf.shape(test_overlaps), tf.where(test_overlaps > 0), test_overlaps],
                        summarize=10)
    sess.run(print_op)
