import tensorflow as tf
from datasets.roidb import RoiDb
from image_interpreter.minibatch import BatchGenerator
from tensorflow.python.training import queue_runner


def main():
  roidb = RoiDb('val.txt', 2007)
  batch_gen = BatchGenerator(roidb)

  image_tensor = tf.placeholder(dtype=tf.float32)
  scale_tensor = tf.placeholder(dtype=tf.float32)
  bboxes_tensor = tf.placeholder(dtype=tf.float32)
  p_op = tf.Print(image_tensor, [tf.shape(image_tensor), scale_tensor, bboxes_tensor])

  sess = tf.Session()
  init = tf.initialize_all_variables()
  sess.run(init)

  coord = tf.train.Coordinator()
  queue_threads = queue_runner.start_queue_runners(sess, coord=coord)

  for i in range(10):
    if coord.should_stop():
      break
    image, scale, bboxes = batch_gen.next_batch()

    sess.run([p_op], feed_dict={image_tensor: image, scale_tensor: scale, bboxes_tensor:bboxes})

  coord.request_stop()
  coord.join(queue_threads)


if __name__ == '__main__':
  main()
