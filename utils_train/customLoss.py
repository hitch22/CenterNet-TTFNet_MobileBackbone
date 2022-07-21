import tensorflow as tf
import numpy as np
from absl import app, logging, flags

_policy = tf.keras.mixed_precision.global_policy()

class L1(tf.losses.Loss):
    def __init__(self):
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

        pos_mask = 'asd'#wh_weight>0 # b h w   
        y_pred = tf.where(pos_mask[..., tf.newaxis], y_pred, 0.0)
        y_true= tf.where(pos_mask[..., tf.newaxis], y_true, 0.0)

        total_loss = tf.reduce_sum(tf.abs(y_true - y_pred))
        loss /= mask
        normalizer = tf.reduce_sum(mask, [1,2])
        return tf.math.divide_no_nan(loss, normalizer)

class IOU(tf.losses.Loss):
    def __init__(self, config):
        super().__init__(reduction="none", name="IOULoss")
        self.mode = config['training_config']['BoxLoss']["LossFunction"].upper()
        logging.warning('IOULoss: {}'.format(self.mode))

        h = w = config['model_config']['target_size']//4
        #x_grid, y_gird = tf.meshgrid(tf.range(h, dtype=_policy.compute_dtype), tf.range(w, dtype= _policy.compute_dtype))
        x_grid, y_gird = tf.meshgrid(tf.range(0.0, 1.0, 1.0/h, dtype=tf.float32), tf.range(0.0, 1.0, 1.0/w, dtype=tf.float32))
        self.grid = tf.stack([y_gird, x_grid], -1)

    def call(self, box_true_with_W, box_pred):
        Size_pred = box_pred[..., :4]
        bbox_gt = box_true_with_W[..., :4]
        wh_weight = box_true_with_W[..., 4]
        
        yx_min = self.grid-Size_pred[..., :2]
        yx_max = self.grid+Size_pred[..., 2:]

        bbox_pred = tf.concat((yx_min, yx_max), axis=-1)

        loss = self._calculateIOULoss(bbox_gt, bbox_pred)
        loss = tf.where(wh_weight>0, loss, 0.0)
        loss = loss*wh_weight

        loss = tf.reduce_sum(loss, [1,2])
        normalizer = tf.reduce_sum(wh_weight, [1,2])
        return tf.math.divide_no_nan(loss, normalizer)

    def _calculateIOULoss(self, b1, b2):
        '''
            input_format y1 x1 y2 x2
        '''
        inner_lu = tf.maximum(b1[..., :2], b2[..., :2])
        inner_rd = tf.minimum(b1[..., 2:], b2[..., 2:])
        
        enclose_lu = tf.minimum(b1[..., :2], b2[..., :2])
        enclose_rd = tf.maximum(b1[..., 2:], b2[..., 2:])

        center_gt = (b1[..., 2:] + b1[..., :2]) / 2
        center_pred = (b2[..., 2:] + b2[..., :2]) / 2

        area_gt = tf.reduce_prod(b1[..., 2: ] - b1[..., :2], -1)
        area_pred = tf.reduce_prod(b2[..., 2: ] - b2[..., :2], -1)

        inner_intersection = tf.maximum(0.0, inner_rd - inner_lu)
        enclose_intersection = tf.maximum(0.0, enclose_rd - enclose_lu)

        inner_intersection_area = inner_intersection[..., 0] * inner_intersection[..., 1]
        union_area = area_gt + area_pred - inner_intersection_area

        iou = tf.math.divide_no_nan(inner_intersection_area, union_area)

        if self.mode == "IOU":
            iou_loss = tf.clip_by_value(iou, 1e-4, 1.0)
            return -tf.math.log(iou_loss)

        elif self.mode =="GIOU":
            enclose_area = enclose_intersection[..., 0] * enclose_intersection[..., 1]
            gterm = tf.math.divide_no_nan((enclose_area - union_area), enclose_area)
            return 1-tf.clip_by_value(iou-gterm, -1.0, 1.0)

        else:
            center_distance_square = tf.reduce_sum(tf.math.square(center_gt - center_pred), axis = -1)
            diagonal_distance_square = tf.reduce_sum(tf.math.square(enclose_intersection), axis = -1)
            u = tf.math.divide_no_nan(center_distance_square, diagonal_distance_square)

            if self.mode == "DIOU":
                return 1-tf.clip_by_value(iou-u, -1.0, 1.0)

            elif self.mode == "CIOU":
                w_gt = b1[..., 2] - b1[..., 0]
                h_gt = b1[..., 3] - b1[..., 1]
                w_pred = b2[..., 2] - b2[..., 0]
                h_pred = b2[..., 3] - b2[..., 1]

                arctanTerm = tf.math.atan(w_gt / (h_gt+1e-8)) - tf.math.atan(w_pred / (h_pred+1e-8))
                v = 4 / (np.pi ** 2) * tf.pow(arctanTerm, 2)
                ar = 8 / (np.pi ** 2) * arctanTerm * ((w_pred - 2 * w_pred) * h_pred)

                S = 1 - iou
                alpha = tf.math.divide_no_nan(v, S+v)
                return 1-tf.clip_by_value(iou - (u + alpha*ar), -1.0, 1.0)

            else:
                raise ValueError("{} is not implemented".format(self.mode))

class HeatmapFocal(tf.losses.Loss):
    def __init__(self, alpha = 2, gamma = 4):
        super().__init__(reduction="none", name="HeatmapFocalLoss")
        self._alpha = alpha
        self._gamma = gamma

    def call(self, hm_true, hm_pred):        
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
        self._box_loss = IOU(config)

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
        self._box_loss = L1()

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