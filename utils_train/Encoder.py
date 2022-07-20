import tensorflow as tf

class LabelEncoder():
    def __init__(self, config):
        self.FeatureMapResolution=config['model_config']['target_size']//4
        self.targetSize=config["model_config"]["target_size"]
        self.classNum=config["training_config"]["num_classes"]
        self.downSampleRate=self.targetSize/self.FeatureMapResolution

        x_grid, y_gird = tf.meshgrid(tf.range(self.FeatureMapResolution, dtype=tf.float32), tf.range(self.FeatureMapResolution, dtype=tf.float32))
        #x_grid, y_gird = tf.meshgrid(tf.range(0.0, 1.0, 1.0/self.FeatureMapResolution, dtype=tf.float32), tf.range(0.0, 1.0, 1.0/self.FeatureMapResolution, dtype=tf.float32))
        grid = tf.stack([y_gird, x_grid], -1)
        self.grid = tf.expand_dims(grid, -2)
        self.mode = config["model_config"]["head"]["name"].upper()

    def _make_box_target(self, hm, rescaled_boxes):
        gt_boxes = rescaled_boxes/self.FeatureMapResolution
        bbox_area = tf.math.log(tf.reduce_prod(rescaled_boxes[..., 2:]-rescaled_boxes[..., :2]+1, axis=-1))
        ids = tf.argsort(-bbox_area)
        #ids = tf.argsort(tf.cast(-bbox_area, tf.float64))

        sorted_area = tf.gather(bbox_area, ids, axis = -1) #N
        sorted_boxes = tf.gather(gt_boxes, ids)
        sorted_hm = tf.gather(hm, ids, axis=-1) #80 80 N
        
        sorted_mask = tf.where(sorted_hm > 0, 1, 0)
        sorted_mask = sorted_mask*(tf.range(tf.shape(rescaled_boxes)[0])+1)
        priority_mask = tf.logical_and(tf.equal(tf.reduce_max(sorted_mask, axis=-1, keepdims=True), sorted_mask), sorted_mask > 0)
        #priority_mask = tf.equal(tf.reduce_max(sorted_mask, axis=-1, keepdims=True), sorted_mask)

        normalizer = tf.reduce_sum(sorted_hm, [0, 1])

        box_target = tf.tile(sorted_boxes[tf.newaxis, tf.newaxis, ...], [self.FeatureMapResolution,self.FeatureMapResolution,1,1])
        box_target = tf.reduce_max(tf.where(priority_mask[..., tf.newaxis], box_target, 0.0), -2)

        regW_target = tf.math.divide_no_nan(sorted_hm*sorted_area, normalizer)
        regW_target = tf.reduce_max(tf.where(priority_mask, regW_target, 0.0), -1, keepdims=True)
        return box_target, regW_target

    def _make_hm_target(self, hm, cls_ids):
        channel_onehot = tf.one_hot(cls_ids, self.classNum, axis=-1) #[n 80]

        reshaped_gaussian_map = tf.expand_dims(hm, axis=-1)
        reshaped_channel_onehot = channel_onehot[tf.newaxis, tf.newaxis, ...]

        gaussian_per_box_per_class_map = reshaped_gaussian_map * reshaped_channel_onehot
        return tf.reduce_max(gaussian_per_box_per_class_map, axis=2)
    
    def _calculate_hm(self, rescaled_boxes):
        hw = rescaled_boxes[..., 2:] - rescaled_boxes[..., :2]
        center = tf.floor((rescaled_boxes[..., 2:]+rescaled_boxes[..., :2])/2.0)
        dist_from_center = self.grid-center

        if self.mode == "TTFNET":
            alpha = 0.54
            radius = tf.floor(hw/ 2.0 * alpha)
            sigma=(2*radius+1)/6
            mask = tf.reduce_all(tf.abs(dist_from_center) <= tf.minimum(radius, hw//2.0), axis=-1)

        elif self.mode == "CENTERNET":
            min_iou = 0.7
            h_gt = hw[..., 0]
            w_gt = hw[..., 1]

            ####distance_detection_offset
            a=1
            b=-(h_gt+w_gt)
            c=w_gt * h_gt * ((1 - min_iou) / (1 + min_iou))

            discriminant = tf.sqrt(b ** 2 - 4 * a * c)
            distance_detection_offset = (-b + discriminant) / (2.0)
            ####distance_detection_offset

            ####distance_detection_in_gt
            a = 4
            b = -2 * (h_gt + w_gt)
            c = (1 - min_iou) * w_gt * h_gt

            discriminant = tf.sqrt(b ** 2 - 4 * a * c)
            distance_detection_in_gt = (-b + discriminant) / (2.0)
            ####distance_detection_in_gt

            ####distance_gt_in_detection
            a = 4 * min_iou
            b = (2 * min_iou) * (w_gt + h_gt)
            c = (min_iou - 1) * w_gt * h_gt

            discriminant = tf.sqrt(b ** 2 - 4 * a * c)
            distance_gt_in_detection = (-b + discriminant) / (2.0)
            ####distance_gt_in_detection

            radius=tf.reduce_min([distance_detection_offset,
                                distance_gt_in_detection,
                                distance_detection_in_gt], axis=0)
            
            
            radius = radius[tf.newaxis, tf.newaxis, ..., tf.newaxis]
            sigma=(2*radius+1)/6
            mask = tf.reduce_all(tf.abs(dist_from_center) <= radius, axis=-1)
        
        hm = tf.math.exp(-tf.reduce_sum(0.5*tf.square(dist_from_center/sigma), -1))
        hm = tf.where(mask, hm, 0.0)
        hm = tf.where(hm>1e-4, hm, 0.0)

        return hm

    def _encode_sample(self, gt_boxes, cls_ids):
        '''
            input gt_boxes format [y1 x1 y2 x2]
        '''

        if len(gt_boxes) <= 0:
            return tf.zeros([self.FeatureMapResolution, self.FeatureMapResolution, self.classNum+4+1])

        rescaled_boxes = gt_boxes*self.FeatureMapResolution
        hm=self._calculate_hm(rescaled_boxes)
        
        hm_target = self._make_hm_target(hm, cls_ids)
        box_target, regW_target = self._make_box_target(hm, rescaled_boxes)
        return tf.concat([hm_target, box_target, regW_target], -1)