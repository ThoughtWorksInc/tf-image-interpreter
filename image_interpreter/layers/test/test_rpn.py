import tensorflow as tf
from datasets.roidb import RoiDb

from image_interpreter.layers.feature import Vgg16, load_feature_layer_params
from image_interpreter.layers.region_proposal import RpnNet
from image_interpreter.minibatch import BatchGenerator


def test_rpn():
  vgg = Vgg16()
  rpn = RpnNet()
  image_tensor = tf.placeholder(tf.float32)
  with tf.Session() as sess:
    vgg.build(image_tensor)
    rpn.build(vgg.conv5_3)
    init = tf.initialize_all_variables()
    sess.run(init)

    load_feature_layer_params('/Users/dtong/code/data/tf-image-interpreter/pretrain/vgg16_weights.npz', sess)

    roidb = RoiDb('val.txt', 2007)
    batch_gen = BatchGenerator(roidb)

    for i in range(10):
      image, scale, bboxes = batch_gen.next_batch()
      feature_shape = tf.shape(rpn.rpn_cls_score)
      print_feat_shape = tf.Print(feature_shape, [feature_shape], summarize=5)
      sess.run(print_feat_shape, feed_dict={image_tensor: image})

      # print(sess.run(vgg.conv5_3, feed_dict={image_tensor: image}))


if __name__ == '__main__':
  test_rpn()
