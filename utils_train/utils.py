import tensorflow as tf
import matplotlib.pylab as plt
import numpy as np

_policy = tf.keras.mixed_precision.global_policy()

def CalculateIOU(b1, b2):
    '''
        input_format y1 x1 y2 x2
    '''
    inter_lu = tf.maximum(b1[..., :2], b2[..., :2])
    inter_rd = tf.minimum(b1[..., 2:], b2[..., 2:])
    
    outer_lu = tf.minimum(b1[..., :2], b2[..., :2])
    outer_rd = tf.maximum(b1[..., 2:], b2[..., 2:])

    w_gt = b1[..., 2] - b1[..., 0]
    h_gt = b1[..., 3] - b1[..., 1]
    w_pred = b2[..., 2] - b2[..., 0]
    h_pred = b2[..., 3] - b2[..., 1]

    center_gt = (b1[..., 2:] + b1[..., :2]) / 2
    center_pred = (b2[..., 2:] + b2[..., :2]) / 2

    area_gt = tf.reduce_prod(b1[..., 2: ] - b1[..., :2], -1)
    area_pred = tf.reduce_prod(b2[..., 2: ] - b2[..., :2], -1)

    inter_intersection = tf.maximum(0.0, inter_rd - inter_lu)
    outer_intersection = tf.maximum(0.0, outer_rd - outer_lu)

    inter_intersection_area = inter_intersection[..., 0] * inter_intersection[..., 1]
    union_area = area_gt + area_pred - inter_intersection_area

    iou = tf.math.divide_no_nan(inter_intersection_area, union_area)
    
    center_distance = tf.reduce_sum(tf.math.square(center_gt - center_pred), axis = -1)
    diagonal_distance =  tf.square(tf.linalg.norm(outer_intersection, axis = -1))
    u = tf.math.divide_no_nan(center_distance, diagonal_distance)

    arctanTerm = tf.math.atan(w_gt / (h_gt+1e-8)) - tf.math.atan(w_pred / (h_pred+1e-8))
    v = 4 / (np.pi ** 2) * tf.pow(arctanTerm, 2)
    ar = 8 / (np.pi ** 2) * arctanTerm * ((w_pred - 2 * w_pred) * h_pred)

    S = 1 - iou
    alpha = v/(S + v + 1e-8)

    cious = tf.clip_by_value(iou - (u + alpha * ar), -1.0, 1.0)
    diou = iou - u
    return 1.0 - diou

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