
import tensorflow as tf
import numpy as np
from utils_train.utils import CalculateIOU

_policy = tf.keras.mixed_precision.global_policy()

class IOU(tf.losses.Loss):
    def __init__(self, mode):
        super().__init__(reduction="none", name="IOU")
        self.mode = mode

    def call(self, box_true_with_W, box_pred):
        offset_pred = box_pred[..., :4]
        bbox_gt = box_true_with_W[..., :4]
        wh_weight = box_true_with_W[..., 4]
        pos_mask = wh_weight>0 # b h w

        b, h, w = tf.unstack(tf.shape(wh_weight))
        
        x_grid, y_gird = tf.meshgrid(tf.range(h, dtype=tf.float32), tf.range(w, dtype=tf.float32))
        grid = tf.stack([y_gird, x_grid], -1)
        ################################
        #h = tf.cast(h, tf.float32)##
        #w = tf.cast(w, tf.float32)##
        #x_grid, y_gird = tf.meshgrid(tf.range(0.0, 1.0, 1.0/h, dtype=tf.float32), tf.range(0.0, 1.0, 1.0/w, dtype=tf.float32))
        #grid = tf.stack([y_gird, x_grid], -1)
        ################################
        
        yx_min = grid-offset_pred[..., :2]
        yx_max = grid+offset_pred[..., 2:]

        bbox_pred = tf.concat((yx_min, yx_max), axis=-1)
        bbox_pred = tf.where(pos_mask[..., tf.newaxis], bbox_pred, 0.0)
        bbox_gt = tf.where(pos_mask[..., tf.newaxis], bbox_gt, 0.0)
        wh_weight = tf.where(pos_mask, wh_weight, 0.0)

        loss = 1 - CalculateIOU(bbox_gt, bbox_pred, mode="diou")
        loss = loss*wh_weight

        return tf.reduce_sum(loss)/tf.reduce_sum(wh_weight)

class HeatmapFocal(tf.losses.Loss):
    def __init__(self, alpha = 2, gamma = 4):
        super().__init__(reduction="none", name="HeatmapFocal")
        self._alpha = alpha
        self._gamma = gamma

    def call(self, hm_true, hm_pred):        
        hm_pred = tf.clip_by_value(tf.nn.sigmoid(hm_pred), 1e-4, 1.0-1e-4)
        
        pos_mask = tf.abs(hm_true - 1.0) < 1e-4
        loss = -tf.where(pos_mask, \
            tf.math.pow(1.0 - hm_pred, self._alpha)*tf.math.log(hm_pred), \
            tf.math.pow(hm_pred, self._alpha)*tf.math.log(1.0 - hm_pred)*tf.math.pow(1.0 - hm_true, self._gamma)
            )
        loss = tf.reduce_sum(loss)
        normalizer = tf.maximum(tf.reduce_sum(tf.cast(pos_mask, tf.float32)), 1.0)
        loss = loss/normalizer

        '''pos_loss = -tf.math.pow(1.0 - hm_pred, self._alpha)*tf.math.log(hm_pred)
        neg_loss = -tf.math.pow(hm_pred, self._alpha)*tf.math.log(1.0 - hm_pred)*tf.math.pow(1.0 - hm_true, self._gamma)

        pos_loss = tf.where(pos_mask, pos_loss, 0.0)
        neg_loss = tf.where(pos_mask, 0.0, neg_loss)

        pos_loss = tf.reduce_sum(pos_loss)
        neg_loss = tf.reduce_sum(neg_loss)
        num_pos = tf.maximum(tf.reduce_sum(tf.cast(pos_mask, tf.float32)), 1.0)

        loss = tf.cond(tf.greater(num_pos, 0), lambda: (pos_loss + neg_loss) / num_pos, lambda: neg_loss)'''
        return loss

class CenterNetLoss(tf.losses.Loss):
    def __init__(self, config):
        super().__init__(reduction="none", name="CenterNetLoss")
        self._heatmap_loss = HeatmapFocal(alpha = config['training_config']["HeatLoss"]["Alpha"], 
                                        gamma=config['training_config']["HeatLoss"]["Gamma"])
        self._box_loss = IOU(mode = config['training_config']["BoxLoss"]["LossFunction"])

        self._num_classes = config['training_config']["num_classes"]
        self._heat_loss_weight = config['training_config']["HeatLoss"]["Weight"]
        self._loc_loss_weight = config['training_config']["BoxLoss"]["Weight"] # 5.0

    def call(self, y_true, y_pred):
        b = tf.shape(y_true)[0]
        heatmap_true = y_true[..., :80]
        heatmap_pred = y_pred[..., :80]
        
        box_true = y_true[..., 80:]
        box_pred = y_pred[..., 80:]

        heat_loss = self._heatmap_loss(heatmap_true, heatmap_pred)
        box_loss = self._box_loss(box_true, box_pred)

        #heat_loss = tf.where(tf.math.is_nan(heat_loss), 0.0, heat_loss)
        #box_loss = tf.where(tf.math.is_nan(box_loss), 0.0, box_loss)
        return heat_loss*self._heat_loss_weight, box_loss*self._loc_loss_weight