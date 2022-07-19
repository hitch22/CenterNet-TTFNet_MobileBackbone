
import tensorflow as tf
import numpy as np
from utils_train.utils import CalculateIOU

_policy = tf.keras.mixed_precision.global_policy()

class L1(tf.losses.Loss):
    def __init__(self, config):
        super().__init__(reduction="none", name="L1loss")

    def call(y_true, y_pred, indices, mask):
        """
        This function was taken from:
            https://github.com/MioChiu/TF_CenterNet/blob/master/loss.py
        :param y_true: (batch, max_objects, 2)
        :param y_pred: (batch, heatmap_height, heatmap_width, max_objects)
        :param indices: (batch, max_objects)
        :param mask: (batch, max_objects)
        :return: l1 loss (single float value) for given predictions and ground truth
        """
        batch_dim = tf.shape(y_pred)[0]
        channel_dim = tf.shape(y_pred)[-1]
        y_pred = tf.reshape(y_pred, (batch_dim, -1, channel_dim))
        indices = tf.cast(indices, tf.int32)
        y_pred = tf.gather(y_pred, indices, batch_dims=1)
        mask = tf.tile(tf.expand_dims(mask, axis=-1), (1, 1, 2))
        total_loss = tf.reduce_sum(tf.abs(y_true * mask - y_pred * mask))
        loss = total_loss / (tf.reduce_sum(mask) + 1e-5)
        return loss

class IOU(tf.losses.Loss):
    def __init__(self, config):
        super().__init__(reduction="none", name="IOULoss")
        #self.mode = mode
        h = w = config['model_config']['target_size']//4
        x_grid, y_gird = tf.meshgrid(tf.range(h, dtype=_policy.compute_dtype), tf.range(w, dtype= _policy.compute_dtype))
        #x_grid, y_gird = tf.meshgrid(tf.range(0.0, 1.0, 1.0/h, dtype=tf.float32), tf.range(0.0, 1.0, 1.0/w, dtype=tf.float32))
        self.grid = tf.stack([y_gird, x_grid], -1)

    def call(self, box_true_with_W, box_pred):
        offset_pred = box_pred[..., :4]
        bbox_gt = box_true_with_W[..., :4]
        wh_weight = box_true_with_W[..., 4]
        pos_mask = wh_weight>0 # b h w       
        
        yx_min = self.grid-offset_pred[..., :2]
        yx_max = self.grid+offset_pred[..., 2:]

        bbox_pred = tf.concat((yx_min, yx_max), axis=-1)
        bbox_pred = tf.where(pos_mask[..., tf.newaxis], bbox_pred, 0.0)

        #bbox_gt = tf.where(pos_mask[..., tf.newaxis], bbox_gt, 0.0)

        loss = 1 - CalculateIOU(bbox_gt, bbox_pred, mode="diou")
        loss = loss*wh_weight
        loss = tf.reduce_sum(loss, [1,2])

        normalizer = tf.reduce_sum(tf.cast(pos_mask, tf.float32), [1,2])
        return tf.math.divide_no_nan(loss, normalizer)

class HeatmapFocal(tf.losses.Loss):
    def __init__(self, alpha = 2, gamma = 4):
        super().__init__(reduction="none", name="HeatmapFocalLoss")
        self._alpha = alpha
        self._gamma = gamma

    def call(self, hm_true, hm_pred):        
        hm_pred = tf.clip_by_value(tf.nn.sigmoid(hm_pred), 1e-4, 1.0-1e-4)
        
        pos_mask = tf.math.equal(hm_true, 1.0)

        loss = -tf.where(pos_mask, \
            tf.math.pow(1.0 - hm_pred, self._alpha)*tf.math.log(hm_pred), \
            tf.math.pow(hm_pred, self._alpha)*tf.math.log(1.0 - hm_pred)*tf.math.pow(1.0 - hm_true, self._gamma)
            )
        loss = tf.reduce_sum(loss, [1,2,3])

        normalizer = tf.reduce_sum(tf.cast(pos_mask, tf.float32), [1,2,3])
        return tf.math.divide_no_nan(loss, normalizer)

class TTFNetLoss(tf.losses.Loss):
    def __init__(self, config):
        super().__init__(reduction="none", name="TTFNetLoss")
        self._heatmap_loss = HeatmapFocal(alpha = config['training_config']["HeatLoss"]["Alpha"], 
                                        gamma=config['training_config']["HeatLoss"]["Gamma"])
        self._box_loss = IOU(config)#(mode = config['training_config']["BoxLoss"]["LossFunction"])

        self._num_classes = config['training_config']["num_classes"]
        self._heat_loss_weight = config['training_config']["HeatLoss"]["Weight"]
        self._loc_loss_weight = config['training_config']["BoxLoss"]["Weight"]

    def call(self, y_true, y_pred): #32 16
        heatmap_true = y_true[..., :self._num_classes]
        heatmap_pred = y_pred[..., :self._num_classes]
        
        box_true = y_true[..., self._num_classes:]
        box_pred = y_pred[..., self._num_classes:]

        heat_loss = self._heatmap_loss(heatmap_true, heatmap_pred)
        box_loss = self._box_loss(box_true, box_pred)
        return heat_loss*self._heat_loss_weight, box_loss*self._loc_loss_weight

class CenterNetLoss(tf.losses.Loss):
    def __init__(self, config):
        super().__init__(reduction="none", name="CenterNetLoss")
        self._heatmap_loss = HeatmapFocal(alpha = config['training_config']["HeatLoss"]["Alpha"], 
                                        gamma=config['training_config']["HeatLoss"]["Gamma"])
        self._box_loss = L1(config)

        self._num_classes = config['training_config']["num_classes"]
        self._heat_loss_weight = config['training_config']["HeatLoss"]["Weight"]
        self._loc_loss_weight = config['training_config']["BoxLoss"]["Weight"]

    def call(self, y_true, y_pred):
        heatmap_true = y_true[..., :self._num_classes]
        heatmap_pred = y_pred[..., :self._num_classes]
        
        size_true = y_true[..., self._num_classes:self._num_classes+2]
        size_pred = y_pred[..., self._num_classes:self._num_classes+2]
        
        offset_true = y_true[..., -2:]
        offset_pred = y_pred[..., -2:]

        heat_loss = self._heatmap_loss(heatmap_true, heatmap_pred)
        size_loss = self._box_loss(size_true, size_pred)
        offset_loss = self._box_loss(offset_true, offset_pred)
        return heat_loss*self._heat_loss_weight, size_loss*0.1 + offset_loss*self._loc_loss_weight