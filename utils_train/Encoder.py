import tensorflow as tf

class LabelEncoder():
    def __init__(self, config):
        self.FeatureMapResolution=config['model_config']['target_size']//4
        self.targetSize=config["model_config"]["target_size"]
        self.classNum=config["training_config"]["num_classes"]
        self.downSampleRate=self.targetSize/self.FeatureMapResolution

        #x_grid, y_gird = tf.meshgrid(tf.range(self.FeatureMapResolution, dtype=tf.float32), tf.range(self.FeatureMapResolution, dtype=tf.float32))
        x_grid, y_gird = tf.meshgrid(tf.range(0.0, 1.0, 1.0/self.FeatureMapResolution, dtype=tf.float32), tf.range(0.0, 1.0, 1.0/self.FeatureMapResolution, dtype=tf.float32))
        grid = tf.stack([y_gird, x_grid], -1)
        self.grid = tf.expand_dims(grid, -2)
        self.mode = config["model_config"]["head"]["name"].upper()

    def _make_box_target(self, hm, gt_boxes):
        bbox_area = tf.math.log(tf.reduce_prod(gt_boxes[..., 2:]-gt_boxes[..., :2], axis=-1))+2*tf.math.log(tf.cast(self.FeatureMapResolution, tf.float32))
        ids = tf.argsort(-bbox_area)

        sorted_area = tf.gather(bbox_area, ids, axis = -1) #N
        sorted_boxes = tf.gather(gt_boxes, ids)
        sorted_hm = tf.gather(hm, ids, axis=-1) #80 80 N
        
        sorted_mask = tf.where(sorted_hm > 0, 1, 0)
        sorted_mask = sorted_mask*(tf.range(tf.shape(gt_boxes)[0])+1)
        sorted_mask -= tf.reduce_max(sorted_mask, -1, keepdims=True)
        priority_mask = tf.where(sorted_mask >= 0, True, False)
        #priority_mask = tf.logical_and(tf.equal(tf.reduce_max(sorted_mask, axis=-1, keepdims=True), sorted_mask), sorted_mask > 0)

        box_target = tf.tile(sorted_boxes[tf.newaxis, tf.newaxis, ...], [self.FeatureMapResolution,self.FeatureMapResolution,1,1])
        box_target = tf.reduce_max(tf.where(priority_mask[..., tf.newaxis], box_target, 0.0), -2)

        normalizer = tf.reduce_sum(sorted_hm, [0, 1])
        regW_target = tf.math.divide_no_nan(sorted_hm*sorted_area, normalizer)
        regW_target = tf.reduce_max(tf.where(priority_mask, regW_target, 0.0), -1, keepdims=True)
        return box_target, regW_target

    def _make_box_target2(self, hm, gt_boxes):
        bbox_area = tf.math.log(tf.reduce_prod(gt_boxes[..., 2:]-gt_boxes[..., :2], axis=-1))+2*tf.math.log(tf.cast(self.FeatureMapResolution, tf.float32))
        ids = tf.argsort(-bbox_area) #ids = tf.argsort(tf.cast(-bbox_area, tf.float64))

        priority = tf.gather(tf.range(tf.shape(gt_boxes)[0])+1, ids, axis = 0)
        priority_mask = tf.where(hm > 0, 1, 0)*priority
        priority_mask -= tf.reduce_max(priority_mask, -1, keepdims=True)
        priority_mask = tf.where(priority_mask >= 0, True, False)
        
        box_target = tf.gather(gt_boxes, ids, axis = 0)
        box_target = tf.tile(box_target[tf.newaxis, tf.newaxis, ...], [self.FeatureMapResolution,self.FeatureMapResolution,1,1])
        box_target = tf.reduce_max(tf.where(priority_mask[..., tf.newaxis], box_target, 0.0), -2)

        normalizer = tf.reduce_sum(hm, [0, 1])
        regW_target = tf.math.divide_no_nan(hm*bbox_area, normalizer)
        regW_target = tf.gather(regW_target, ids, axis = -1)
        regW_target = tf.reduce_max(tf.where(priority_mask, regW_target, 0.0), -1, keepdims=True)
        return box_target, regW_target

    def _make_hm_target(self, hm, cls_ids):
        channel_onehot = tf.one_hot(cls_ids, self.classNum, axis=-1) #[n 80]

        reshaped_gaussian_map = tf.expand_dims(hm, axis=-1)
        reshaped_channel_onehot = channel_onehot[tf.newaxis, tf.newaxis, ...]

        gaussian_per_box_per_class_map = reshaped_gaussian_map*reshaped_channel_onehot
        return tf.reduce_max(gaussian_per_box_per_class_map, axis=2)
    
    def _calculate_hm(self, gt_boxes):
        def _cal_gaussian_radius(a, b, c):
            discriminant = tf.sqrt(b**2 - 4*a*c)
            return (-b + discriminant)/2.0

        hw = gt_boxes[..., 2:] - gt_boxes[..., :2]
        center = (gt_boxes[..., 2:]+gt_boxes[..., :2])/2.0
        dist_from_center = self.grid-center
        dist_from_center = tf.floor(dist_from_center*self.FeatureMapResolution)/self.FeatureMapResolution
        
        if self.mode == "TTFNET":
            alpha = 0.54
            radius = hw/2.0*alpha
            mask = tf.reduce_all(tf.abs(dist_from_center) <= tf.minimum(radius, hw/2.0), axis=-1)
        else:
            min_iou = 0.7
            h_gt = hw[..., 0]
            w_gt = hw[..., 1]

            distance_detection_offset=_cal_gaussian_radius(a=1, b=-(h_gt+w_gt), c=w_gt*h_gt*((1 - min_iou)/(1 + min_iou)))
            distance_detection_in_gt =_cal_gaussian_radius(a=4, b=-2*(h_gt + w_gt), c=(1 - min_iou)*w_gt*h_gt)
            distance_gt_in_detection =_cal_gaussian_radius(a=4*min_iou, b=2*min_iou*(w_gt + h_gt), c=(min_iou - 1)*w_gt*h_gt)
            radius=tf.reduce_min([distance_detection_offset,
                                distance_gt_in_detection,
                                distance_detection_in_gt], axis=0)
            radius = radius[tf.newaxis, tf.newaxis, ..., tf.newaxis]
            mask = tf.reduce_all(tf.abs(dist_from_center) <= radius, axis=-1)
        
        sigma=radius/3.0
        hm = tf.math.exp(-tf.reduce_sum(0.5*tf.square(dist_from_center/sigma), -1))
        hm = tf.where(mask, hm, 0.0)
        hm = tf.where(hm>1e-4, hm, 0.0)
        return hm

    def _encode_sample(self, gt_boxes, cls_ids):
        '''
            input gt_boxes format [y1 x1 y2 x2]
        '''
        assert self.mode in ['TTFNET', "CENTERNET"]
        if len(gt_boxes) <= 0:
            if self.mode == "TTFNET":
                return tf.zeros([self.FeatureMapResolution, self.FeatureMapResolution, self.classNum+4+1])
            else:
                return tf.zeros([self.FeatureMapResolution, self.FeatureMapResolution, self.classNum+4])

        if self.mode == "TTFNET":
            hm=self._calculate_hm(gt_boxes)
            hm_target = self._make_hm_target(hm, cls_ids)
            box_target, regW_target = self._make_box_target(hm, gt_boxes)
            return tf.concat([hm_target, box_target, regW_target], -1)
        
        else:
            '''# calculate center, size of the bounding box
            center = (bboxes[:, :, 0:2] + bboxes[:, :, 2:4]) / 2.0
            size = -bboxes[:, :, 0:2] + bboxes[:, :, 2:4]

            # downsample center and size
            center = center / float(downsample)
            size = size / float(downsample)

            # calculate point indices so that we can easily get the values from prediction matrices
            center_int = tf.cast(center, dtype=tf.int32)
            heatmap_width = image_size // downsample
            indices = center_int[:, :, 0] * heatmap_width + center_int[:, :, 1]

            # calculate offset
            local_offset = center - tf.cast(center_int, dtype=tf.float32)
            return size, local_offset, indices #hw: size/FeatureMapSize'''
            pass
            
        '''
        Center
        return (
                {"input": images},
                {"heatmap": heatmap_dense, "size": size, "offset": local_offset},
                {
                    "indices": indices,
                    "mask": mask,
                    "bboxes": bboxes,
                    "labels": labels,
                    "ids": image_ids,
                    "heights": heights,
                    "widths": widths,
                },
            )
        TTF:
            # otherwise we are fittint TTF net
            heatmap_dense, box_target, reg_weight, _ = tf.numpy_function(
                func=draw_heatmaps_ttf,
                inp=[heatmap_shape, bboxes, labels],
                Tout=[tf.float32, tf.float32, tf.float32, tf.float32],
            )
            heatmap_dense = tf.reshape(heatmap_dense, heatmap_shape)
            box_target = tf.reshape(box_target, [tf.shape(images)[0], heatmap_size, heatmap_size, 4])
            reg_weight = tf.reshape(reg_weight, [tf.shape(images)[0], heatmap_size, heatmap_size, 1])
            return (
                {"input": images},
                {"heatmap": heatmap_dense, "size": size, "offset": local_offset},
                {
                    "indices": indices,
                    "mask": mask,
                    "bboxes": bboxes,
                    "labels": labels,
                    "box_target": box_target,
                    "reg_weight": reg_weight,
                    "ids": image_ids,
                    "heights": heights,
                    "widths": widths,
                },
            )
        '''