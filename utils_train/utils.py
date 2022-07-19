import tensorflow as tf

@tf.function()
def CalculateIOA(boxes1, boxes2):
    boxes1_corners = boxes1
    boxes2_corners = boxes2

    lu = tf.maximum(boxes1_corners[:, None, :2], boxes2_corners[:, :2]) #[BOX1 BOX2 2]
    rd = tf.minimum(boxes1_corners[:, None, 2:], boxes2_corners[:, 2:]) #[BOX1 BOX2 2]

    intersection = tf.maximum(0.0, rd - lu)
    intersection_area = intersection[:, :, 0] * intersection[:, :, 1]

    boxes1_area = tf.reduce_prod(boxes1_corners[:, 2:] - boxes1_corners[:, :2], -1)
    boxes2_area = tf.reduce_prod(boxes2_corners[:, 2:] - boxes2_corners[:, :2], -1)

    return tf.math.divide_no_nan(intersection_area, boxes2_area)

################################## from TFOD
@tf.function()
def scale(boxlist, y_scale, x_scale):
    y_scale = tf.cast(y_scale, tf.float32)
    x_scale = tf.cast(x_scale, tf.float32)
    y_min, x_min, y_max, x_max = tf.split(value=boxlist, num_or_size_splits=4, axis=1)
    y_min = y_scale * y_min
    y_max = y_scale * y_max
    x_min = x_scale * x_min
    x_max = x_scale * x_max
    scaled_boxlist = tf.concat([y_min, x_min, y_max, x_max], 1)
    return scaled_boxlist