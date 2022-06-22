import tensorflow as tf

class LabelEncoder():
    def __init__(self, config):
        self.FeatureMapResolution=config["model_config"]["feature_map_shapes"]
        self.targetSize=config["model_config"]["target_size"]
        self.classNum=config["training_config"]["num_classes"]
        self.downSampleRate=self.targetSize/self.FeatureMapResolution

    def _make_box_target(self, hm, rescaled_boxes):
        bbox_area = tf.math.log(tf.reduce_prod(rescaled_boxes[..., 2:]-rescaled_boxes[..., :2]+1, axis=-1))
        ids = tf.argsort(tf.cast(-bbox_area, tf.float64))

        sorted_area = tf.gather(bbox_area, ids, axis = -1)
        sorted_rescaled_boxes = tf.gather(rescaled_boxes, ids)
        sorted_hm = tf.gather(hm, ids, axis=-1)
        sorted_mask =  tf.where(sorted_hm > 0, 1, 0)

        sorted_mask = sorted_mask*(tf.range(tf.shape(rescaled_boxes)[0])+1)
        priority_mask = tf.logical_and(tf.equal(tf.reduce_max(sorted_mask, [-1], keepdims=True), sorted_mask), sorted_mask > 0)

        temp_target = tf.tile(sorted_rescaled_boxes[tf.newaxis, tf.newaxis, ...], [self.FeatureMapResolution,self.FeatureMapResolution,1,1])
        box_target = tf.reduce_max(tf.where(priority_mask[..., tf.newaxis], temp_target, 1.0), -2)

        normalizer = tf.reduce_sum(sorted_hm, [0, 1])
        normalizer = tf.floor(normalizer)
        temp_regW_target = tf.math.divide_no_nan(sorted_hm*sorted_area, normalizer)
        regW_target = tf.reduce_max(tf.where(priority_mask, temp_regW_target, 0.0), -1, keepdims=True)
        return box_target, regW_target

    def _make_hm_target(self, hm, cls_ids):
        unique_class = tf.unique_with_counts(cls_ids)
        counts_cum=tf.math.cumsum(unique_class.count)
        y=unique_class.y
        
        ################
        hm2 = tf.zeros([self.FeatureMapResolution, self.FeatureMapResolution, 0])
        hm2 = tf.concat([hm2, tf.zeros([self.FeatureMapResolution, self.FeatureMapResolution, y[0] if y[0] > 0 else 0])], -1)
        for i in range(len(y)):
            if i == 0:
                hm2 = tf.concat([hm2, tf.reduce_max(hm[..., :counts_cum[i]], axis=-1, keepdims=True)], -1)
            else:
                hm2 = tf.concat([hm2, tf.reduce_max(hm[..., counts_cum[i-1]:counts_cum[i]], axis=-1, keepdims=True)], -1)
            if i+1 < tf.shape(y):
                hm2 = tf.concat([hm2, tf.zeros([self.FeatureMapResolution, self.FeatureMapResolution, y[i+1]-y[i]-1])], -1)
            else:
                hm2 = tf.concat([hm2, tf.zeros([self.FeatureMapResolution, self.FeatureMapResolution, self.classNum-y[-1]-1])], -1)
        ################
        return hm2

    def _make_hm_target2(self, hm, cls_ids):
        channel_onehot = tf.one_hot(cls_ids, 80, axis=-1) #[n 80]

        reshaped_gaussian_map = tf.expand_dims(hm, axis=-1)
        reshaped_channel_onehot = channel_onehot[tf.newaxis, tf.newaxis, ...]

        gaussian_per_box_per_class_map = reshaped_gaussian_map * reshaped_channel_onehot
        return tf.reduce_max(gaussian_per_box_per_class_map, axis=2)

    def _encode_sample(self, gt_boxes, cls_ids):
        # input gt_boxes format [y1 x1 y2 x2]
        if len(gt_boxes) <= 0:
            return tf.zeros([self.FeatureMapResolution, self.FeatureMapResolution, self.classNum+4+1])

        rescaled_boxes = gt_boxes*self.FeatureMapResolution
        ################radius_ttf
        alpha = 0.54
        hw = rescaled_boxes[..., 2:] - rescaled_boxes[..., :2]
        radius = tf.floor(hw/ 2.0 * alpha)
        center = tf.floor((rescaled_boxes[..., 2:]+rescaled_boxes[..., :2])/2.0)
        
        #hw = rescaled_boxes[..., 2:] - rescaled_boxes[..., :2]
        #radius = hw/2.0*alpha
        #center = (rescaled_boxes[..., 2:]+rescaled_boxes[..., :2])/2.0
        ################
        sigma=(2*radius+1)/6

        x_grid, y_gird = tf.meshgrid(tf.range(self.FeatureMapResolution, dtype=tf.float32), tf.range(self.FeatureMapResolution, dtype=tf.float32))
        #x_grid, y_gird = tf.meshgrid(tf.range(0.0, 1.0, 1.0/self.FeatureMapResolution, dtype=tf.float32), tf.range(0.0, 1.0, 1.0/self.FeatureMapResolution, dtype=tf.float32))
        grid = tf.stack([y_gird, x_grid], -1)
        dist_from_center = tf.expand_dims(grid, -2)-center
            
        mask = tf.reduce_all(tf.abs(dist_from_center) <= tf.minimum(radius, hw//2.0), axis=-1)
        hm = tf.math.exp(-tf.reduce_sum(0.5*tf.square(dist_from_center/sigma), -1))
        hm = tf.where(mask, hm, 0.0)
        hm = tf.where(hm>1e-4, hm, 0.0)

        hm_target = self._make_hm_target(hm, cls_ids)
        box_target, regW_target = self._make_box_target(hm, rescaled_boxes)

        return tf.concat([hm_target, box_target, regW_target], -1)