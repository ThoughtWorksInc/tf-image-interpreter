import tensorflow as tf
from image_interpreter.layers.common import AnchorTargetMixin, arg_sort_op


class Proposal(AnchorTargetMixin):
  def __init__(self, debug):
    super().__init__(debug=debug)
    self._num_anchors = tf.shape(self._anchors)[0]
    self._pre_nms_top_n = 12000
    self._post_nms_top_n = 2000
    self._nms_threshold = 0.7
    self._rpn_min_size = 16

  # TODO: 这个im_info还是需要的，因为需要的是原图的尺寸，image在这里是经过多次pooling的结果，shit，忘记了！
  def build(self, image, bbox_pred, cls_prob, im_info):
    # the first set of _num_anchors channels are bg probs
    # the second set are the fg probs, which we want
    scores = cls_prob[0, :, :, self._num_anchors:]
    height = tf.shape(scores)[0]
    width = tf.shape(scores)[1]

    if self._debug:
      height = tf.Print(height, [height], message='scores height: ')
      width = tf.Print(width, [width], message='scores width: ')

    shifts = self._generate_shifts(width, height)
    all_anchors = tf.cast(self._generate_all_anchors(shifts), dtype=tf.float32)

    if self._debug:
      all_anchors = tf.Print(all_anchors, [tf.shape(all_anchors)], message='anchor size: ')

    bbox_deltas = tf.reshape(bbox_pred, (-1, 4))
    scores = tf.reshape(scores, (-1, 1))

    proposals = self._combine_box_and_delta(all_anchors, bbox_deltas)
    proposals = self._clip_boxes(proposals, image)

    ws = proposals[:, 2] - proposals[:, 0] + 1
    hs = proposals[:, 3] - proposals[:, 1] + 1
    keep = tf.where(
      (ws >= tf.cast(self._rpn_min_size * im_info[2], tf.float32)) &
      (hs >= tf.cast(self._rpn_min_size * im_info[2], tf.float32))
    )[:, 0]
    if self._debug:
      keep = tf.Print(keep, [tf.shape(keep), keep], message='min size filter keep: ', summarize=10)
    proposals = tf.gather(proposals, keep)
    scores = tf.gather(scores, keep)

    proposals, scores = self._filter_top_n(proposals, scores)

    keep = tf.image.non_max_suppression(proposals, tf.reshape(scores, (-1,)), self._post_nms_top_n, iou_threshold=self._nms_threshold)
    if self._debug:
      keep = tf.Print(keep, [tf.shape(keep), keep], message='nms keep: ', summarize=10)
    proposals = tf.gather(proposals, keep)
    scores = tf.gather(scores, keep)
    return proposals, scores

  def _filter_top_n(self, proposals, scores):
    order = tf.py_func(arg_sort_op, [tf.reshape(scores, (-1,))], [tf.int64])[0]
    if self._debug:
      order = tf.Print(order, [tf.shape(order), tf.shape(proposals), tf.shape(scores)], message='order shape: ')
    if self._pre_nms_top_n > 0:
      order = tf.gather(order, tf.range(0, self._pre_nms_top_n))
    proposals = tf.gather(proposals, order)
    scores = tf.gather(scores, order)
    return proposals, scores

  def _clip_boxes(self, boxes, image):
    height = tf.shape(image)[1]
    width = tf.shape(image)[2]
    # TODO: what TF will do with tensors that will not be used anymore?
    x1_over_0 = tf.reshape(tf.maximum(tf.minimum(boxes[:, 0::4], tf.cast(width - 1, tf.float32)), 0), (-1,))
    y1_over_0 = tf.reshape(tf.maximum(tf.minimum(boxes[:, 1::4], tf.cast(height - 1, tf.float32)), 0), (-1,))
    x2_below_width = tf.reshape(tf.maximum(tf.minimum(boxes[:, 2::4], tf.cast(width - 1, tf.float32)), 0), (-1,))
    y2_below_height = tf.reshape(tf.maximum(tf.minimum(boxes[:, 3::4], tf.cast(height - 1, tf.float32)), 0), (-1,))
    boxes = tf.pack(
      [x1_over_0,  # x1 >= 0
       y1_over_0,  # y1 >= 0
       x2_below_width,  # x2 < im_shape[1]
       y2_below_height],  # y2 < im_shape[0]
      axis=1
    )
    return boxes

  # bbox_transform_inv
  def _combine_box_and_delta(self, bboxes, deltas):
    widths = bboxes[:, 2] - bboxes[:, 0] + 1.0
    heights = bboxes[:, 3] - bboxes[:, 1] + 1.0
    ctr_x = bboxes[:, 0] + 0.5 * widths
    ctr_y = bboxes[:, 1] + 0.5 * heights

    # use 0::4 to make it a [-1, 1] matrix, while the columns are 4
    dx = deltas[:, 0::4]
    dy = deltas[:, 1::4]
    dw = deltas[:, 2::4]
    dh = deltas[:, 3::4]

    # do not understand the transformation
    # TF 在这里需要reshape成向量，不然会多出一个没用的维度
    pred_ctr_x = tf.reshape(dx * widths[:, tf.newaxis] + ctr_x[:, tf.newaxis], (-1,))
    pred_ctr_y = tf.reshape(dy * heights[:, tf.newaxis] + ctr_y[:, tf.newaxis], (-1,))
    pred_w = tf.reshape(tf.exp(dw) * widths[:, tf.newaxis], (-1,))
    pred_h = tf.reshape(tf.exp(dh) * heights[:, tf.newaxis], (-1,))

    pred_boxes = tf.pack(
      [pred_ctr_x - 0.5 * pred_w,
       pred_ctr_y - 0.5 * pred_h,
       pred_ctr_x + 0.5 * pred_w,
       pred_ctr_y + 0.5 * pred_h],
      axis=1
    )

    return pred_boxes


if __name__ == '__main__':
  test_image = tf.reshape(tf.ones((600, 600), dtype=tf.float32), (1, 600, 600, 1))
  test_preds = tf.reshape(tf.ones((100, 100, 36), dtype=tf.float32), (1, 100, 100, 36))
  test_cls_probs = tf.reshape(tf.truncated_normal((100, 100, 18)), (1, 100, 100, 18))
  proposal = Proposal(debug=True)
  proposals, scores = proposal.build(test_image, test_preds, test_cls_probs, im_info=tf.constant([600, 600, 1.5]))

  with tf.Session() as sess:
    props, scs = sess.run([proposals, scores])
    print(props.shape, scs.shape)
    print(props, scs)
