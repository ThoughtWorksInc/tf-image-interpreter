import tensorflow as tf
from image_interpreter.layers.common import ConvNetMixin, AnchorTargetMixin


class RoiPooling(ConvNetMixin):
  def __init__(self):
    super().__init__()

  def build(self, feature_map, rpn_cls_score_reshape):
    # layer
    # {
    #   name: "rpn_bbox_pred"
    #   type: "Convolution"
    #   bottom: "rpn/output"
    #   top: "rpn_bbox_pred"
    #   param {lr_mult: 1.0}
    #   param {lr_mult: 2.0}
    #   convolution_param {
    #     num_output: 36  # 4 * 9(anchors)
    #     kernel_size: 1 pad: 0 stride: 1
    #     weight_filler {type: "gaussian" std: 0.01}
    #     bias_filler {type: "constant" value: 0}
    #   }
    # }
    self.rpn_data_pred = self._conv_layer(feature_map, (1, 1), 36, padding='VALID', scope_name='rpn_data_pred')

    # layer
    # {
    #   name: "rpn_cls_prob"
    #   type: "Softmax"
    #   bottom: "rpn_cls_score_reshape"
    #   top: "rpn_cls_prob"
    # }
    self.rpn_cls_prob = tf.nn.softmax(rpn_cls_score_reshape)

    # layer
    # {
    #   name: 'rpn_cls_prob_reshape'
    #   type: 'Reshape'
    #   bottom: 'rpn_cls_prob'
    #   top: 'rpn_cls_prob_reshape'
    #   reshape_param {shape {dim: 0 dim: 18 dim: -1 dim: 0}}
    # }
    cls_prob_shape = tf.shape(self.rpn_cls_prob)
    self.rpn_cls_prob_reshape = tf.reshape(self.rpn_cls_prob, (cls_prob_shape[0], cls_prob_shape[1], -1, 18))

