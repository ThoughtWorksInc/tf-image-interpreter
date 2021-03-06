import numpy as np
import tensorflow as tf
from image_interpreter.layers.common import AnchorTargetMixin


class RpnData(AnchorTargetMixin):
  def __init__(self, debug=False):
    super().__init__(debug=debug)

  def generate(self, image, scale, bboxes):
    shape = tf.shape(image)
    # TODO: NotImplementedError: Negative start indices are not currently supported
    # height, width = shape[-2:]
    # height, width = shape[-2:]
    height = shape[1]
    width = shape[2]

    if self._debug:
      height = tf.Print(height, [height], message='image height: ')
      width = tf.Print(width, [width], message='image width: ')

    anchors = self._generate_valid_anchors(width, height)
    overlaps = self._calculate_overlaps(tf.cast(anchors, dtype=tf.float32), tf.cast(bboxes, dtype=tf.float32))

    labels = self._generate_labels(overlaps)

    labels = self._subsample_positive(labels)
    labels = self._subsample_negative(labels)

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

  def _subsample_positive(self, labels):
    # TODO: not implemented
    return labels

  def _subsample_negative(self, labels):
    # TODO: not implemented
    return labels


if __name__ == '__main__':
  with tf.Session() as sess:
    rpn_data = RpnData(debug=True)
    test_image = tf.reshape(tf.constant(np.ones((600, 400))), (1, 600, 400, 1))
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
