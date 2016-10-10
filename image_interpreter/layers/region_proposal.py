import tensorflow as tf

from image_interpreter.layers.common import ConvNetMixin


class RpnNet(ConvNetMixin):
  def build(self, feature_map, rpn_data):
    # layer {
    #   name: "rpn_conv/3x3"
    #   type: "Convolution"
    #   bottom: "conv5_3"
    #   top: "rpn/output"
    #   param { lr_mult: 1.0 }
    #   param { lr_mult: 2.0 }
    #   convolution_param {
    #     num_output: 512
    #     kernel_size: 3 pad: 1 stride: 1
    #     weight_filler { type: "gaussian" std: 0.01 }
    #     bias_filler { type: "constant" value: 0 }
    #   }
    # }
    self.rpn_output = self._conv_layer_with_relu(feature_map, (3, 3), 512, 'rpn_conv')

    # layer
    # {
    #   name: "rpn_cls_score"
    #   type: "Convolution"
    #   bottom: "rpn/output"
    #   top: "rpn_cls_score"
    #   param {lr_mult: 1.0}
    #   param {lr_mult: 2.0}
    #   convolution_param {
    #     num_output: 18  # 2(bg/fg) * 9(anchors)
    #     kernel_size: 1 pad: 0 stride: 1
    #     weight_filler {type: "gaussian" std: 0.01}
    #     bias_filler {type: "constant" value: 0}
    #   }
    # }
    self.rpn_cls_score = self._conv_layer(self.rpn_output, (1, 1), 18, 'rpn_cls_score')

    # layer {
    #    bottom: "rpn_cls_score"
    #    top: "rpn_cls_score_reshape"
    #    name: "rpn_cls_score_reshape"
    #    type: "Reshape"
    #    reshape_param { shape { dim: 0 dim: 2 dim: -1 dim: 0 } }
    # }
    shape = tf.shape(self.rpn_cls_score)
    self.rpn_cls_score_reshape = tf.reshape(self.rpn_cls_score, (shape[0], -1, shape[2], 2), 'rpn_cls_score_reshape')

    # layer
    # {
    #   name: "rpn_loss_cls"
    #   type: "SoftmaxWithLoss"
    #   bottom: "rpn_cls_score_reshape"
    #   bottom: "rpn_labels"
    #   propagate_down: 1
    #   propagate_down: 0
    #   top: "rpn_cls_loss"
    #   loss_weight: 1
    #   loss_param {
    #     ignore_label: -1
    #     normalize: true
    #   }
    # }
    # self.rpn_loss_cls = tf.nn.softmax_cross_entropy_with_logits(self.rpn_cls_score_reshape, rpn_data.rpn_cls_labels)
