import tensorflow as tf
import numpy as np

# feature_map = feature_inference(image_tensor)
# rpn_loss_bbox_tensor = rpn_loss_bbox(feature_map, im_info_tensor, boxes_tensor)
# rpn_cls_loss()
# rpn_rois_tensor = rpn_rois()
# roi_pool_tensor = roi_pool(rpn_rois_tensor)

x = tf.placeholder(dtype=tf.float32)

p_op = tf.Print(x, [x])

init_op = tf.initialize_all_variables()

sess = tf.Session()

sess.run(init_op)

for i in range(10):
  value = np.ones((i, i))
  sess.run(p_op, feed_dict={x: value})
