import tensorflow as tf
from datasets.roidb import RoiDb

from image_interpreter.layers.feature import Vgg16, load_params
from image_interpreter.minibatch import BatchGenerator


def test_vgg():
  vgg = Vgg16()
  image_tensor = tf.placeholder(tf.float32)
  with tf.Session() as sess:
    vgg.build(image_tensor)
    init = tf.initialize_all_variables()
    sess.run(init)

    load_params('/Users/dtong/code/data/tf-image-interpreter/pretrain/vgg16_weights.npz', sess)

    for v in tf.get_collection(tf.GraphKeys.VARIABLES):
      print_op = tf.Print(v, [v], message=v.name, first_n=10)
      sess.run(print_op)

    roidb = RoiDb('val.txt', 2007)
    batch_gen = BatchGenerator(roidb)

    for i in range(10):
      image, scale, bboxes = batch_gen.next_batch()

      print(sess.run(vgg.conv5_3, feed_dict={image_tensor: image}))

if __name__ == '__main__':
    test_vgg()
