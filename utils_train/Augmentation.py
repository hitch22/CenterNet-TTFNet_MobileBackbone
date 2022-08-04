import tensorflow as tf
import random
import tensorflow_addons as tfa

from utils_train.utils import CalculateIOA, scale

def randomCutout(image):
    h, w, _ = tf.unstack(tf.shape(image))
    mask_size=random.choice([40, 60])
    x_start = random.choice([-5, 10, 20])
    y_start = random.choice([-5, 10, 20])

    y_iter = h//(mask_size*2)+1
    x_iter = w//(mask_size*2)+1
    if len(image.get_shape().as_list()) == 3:
        image = tf.expand_dims(image, 0)
        for i in range(x_iter):
            for j in range(y_iter):
                image = tfa.image.cutout(image, mask_size, [y_start+mask_size*j*2, x_start+mask_size*i*2])
        return image[0]
    else:
        for i in range(x_iter):
            for j in range(y_iter):
                image = tfa.image.cutout(image, mask_size, [y_start+mask_size*j*2, x_start+mask_size*i*2])
        return image

def randomGaussian(image,
                          min_patch_size=1,
                          max_patch_size=250,
                          min_gaussian_stddev=0.0,
                          max_gaussian_stddev=1.0,
                          p=1.0):
    '''
        This patch code from TFOD API 
            https://github.com/tensorflow/models/blob/master/research/object_detection/core/preprocessor.py
     
    '''
    def _get_patch_mask(y, x, patch_size, image_shape):

        image_hw = image_shape[:2]
        mask_center_yx = tf.stack([y, x])
        mask_center_yx = tf.identity(mask_center_yx)

        half_patch_size = tf.cast(patch_size, dtype=tf.float32) / 2
        start_yx = mask_center_yx - tf.cast(tf.floor(half_patch_size), dtype=tf.int32)
        end_yx = mask_center_yx + tf.cast(tf.math.ceil(half_patch_size), dtype=tf.int32)

        start_yx = tf.maximum(start_yx, 0)
        end_yx = tf.minimum(end_yx, image_hw)

        start_y = start_yx[0]
        start_x = start_yx[1]
        end_y = end_yx[0]
        end_x = end_yx[1]

        lower_pad = image_hw[0] - end_y
        upper_pad = start_y
        left_pad = start_x
        right_pad = image_hw[1] - end_x
        mask = tf.ones([end_y - start_y, end_x - start_x], dtype=tf.bool)
        return tf.pad(mask, [[upper_pad, lower_pad], [left_pad, right_pad]])

    if tf.random.uniform([], minval=0, maxval=1) > p:
        return image

    patch_size = tf.random.uniform([], minval=min_patch_size, maxval=max_patch_size, dtype=tf.int32)
    gaussian_stddev = tf.random.uniform([], minval=min_gaussian_stddev, maxval=max_gaussian_stddev, dtype=tf.float32)

    image_shape = tf.shape(image)
    y = tf.random.uniform([], minval=0, maxval=image_shape[0], dtype=tf.int32)
    x = tf.random.uniform([], minval=0, maxval=image_shape[1], dtype=tf.int32)
    gaussian = tf.random.normal(image_shape, stddev=gaussian_stddev, dtype=tf.float32)

    scaled_image = image / 255.0
    image_plus_gaussian = tf.clip_by_value(scaled_image + gaussian, 0.0, 1.0)
    patch_mask = _get_patch_mask(y, x, patch_size, image_shape)
    patch_mask = tf.expand_dims(patch_mask, -1)
    patch_mask = tf.tile(patch_mask, [1, 1, image_shape[2]])
    patched_image = tf.where(patch_mask, image_plus_gaussian, scaled_image)
    return patched_image * 255.0


def randomResize(image, boxes, targetH, targetW, p = 1.0):
    def _keep_aspect_ratio(img, boxes, h, w):
        image_shape = tf.cast(tf.shape(img), tf.float32)
        image_height, image_width = image_shape[0], image_shape[1]

        img = tf.image.resize_with_pad(img, h, w)
        
        h, w = tf.cast(h, dtype=tf.float32), tf.cast(w, dtype=tf.float32)
        resize_coef = tf.math.minimum(h / image_height, w / image_width)

        resized_height, resized_width = image_height * resize_coef, image_width * resize_coef
        pad_y, pad_x = (h - resized_height) / 2, (w - resized_width) / 2
        boxes = boxes * tf.stack([resized_height, resized_width, resized_height, resized_width]) + \
                        tf.stack([pad_y, pad_x, pad_y, pad_x,]
        )
        boxes /= tf.stack([h, w, h, w])
        return img, boxes

    def _dont_keep_aspect_ration(img, boxes, h, w):
        img = tf.image.resize(img, (h, w), antialias=True)
        return img, boxes

    if tf.random.uniform([], minval=0, maxval=1) < p:
        keep_aspect_ratio = True
    else:
        keep_aspect_ratio = False
    image, boxes = tf.cond(keep_aspect_ratio,
                            lambda: _keep_aspect_ratio(image, boxes, targetH, targetW),
                            lambda: _dont_keep_aspect_ration(image, boxes, targetH, targetW))
    return image, boxes


def flipHorizontal(image, boxes, p = 1.0):
    if tf.random.uniform([], minval=0, maxval=1) > p:
        return image, boxes

    image = tf.image.flip_left_right(image)
    boxes = tf.stack([boxes[:, 0], 1.0 - boxes[:, 3], boxes[:, 2], 1.0 - boxes[:, 1]], axis=-1)

    return image, boxes


def flipVertical(image, boxes, p = 1.0):
    if tf.random.uniform([], minval=0, maxval=1) > p:
        return image, boxes

    image = tf.image.flip_up_down(image)
    boxes = tf.stack([1.0 - boxes[:, 2], boxes[:, 1], 1.0 - boxes[:, 0], boxes[:, 3]], axis=-1)

    return image, boxes

def randomCrop(image, bbox, class_id, p = 1.0):
    '''
        This crop code from TFOD API 
            https://github.com/tensorflow/models/blob/master/research/object_detection/core/preprocessor.py
     
    '''

    if tf.random.uniform([], minval=0, maxval=1) > p:
        return image, bbox, class_id

    def _prune_completely_outside_window(bbox, window):
        y_min, x_min, y_max, x_max = tf.split(value=bbox, num_or_size_splits=4, axis=1)
        win_y_min, win_x_min, win_y_max, win_x_max = tf.unstack(window)

        coordinate_violations = tf.concat([
            tf.greater_equal(y_min, win_y_max), tf.greater_equal(x_min, win_x_max),
            tf.less_equal(y_max, win_y_min), tf.less_equal(x_max, win_x_min)
        ], 1)

        valid_mask = tf.reshape(tf.logical_not(tf.reduce_any(coordinate_violations, 1)), [-1])
        return tf.boolean_mask(bbox, valid_mask), valid_mask

    def _prune_non_overlapping_boxes(bbox, window, min_overlap=0.3):
        #ioa_ = ioa(boxlist2, boxlist1)  # [M, N] tensor
        ioa_  = CalculateIOA(window, bbox)  # [1, N] tensor
        ioa_ = tf.reduce_max(ioa_, axis=0)  # [N] tensor
        keep_bool = tf.greater_equal(ioa_, min_overlap)
        keep_inds = tf.squeeze(tf.where(keep_bool), axis=[1])
        #new_bbox = tf.gather(bbox, keep_inds)
        new_bbox = tf.boolean_mask(bbox, keep_bool)
        return new_bbox, keep_bool

    def _change_coordinate_frame(boxlist, window):
        win_height = window[2] - window[0]
        win_width = window[3] - window[1]
        boxlist_new = scale(boxlist - [window[0], window[1], window[0], window[1]],
                            1.0 / win_height, 
                            1.0 / win_width)
        return boxlist_new
    image_shape = tf.shape(image)

    boxes_expanded = tf.expand_dims(bbox, 1) # [N, 4] -> [N, 1, 4].
    random_option = [0.2, 0.3, 0.5, 0.7, 0.9, 1.0]

    im_box_begin, im_box_size, im_box = tf.image.sample_distorted_bounding_box(image_shape,
                                                            bounding_boxes=boxes_expanded,
                                                            min_object_covered=random.choice(random_option),
                                                            aspect_ratio_range=[0.5, 2.0],
                                                            area_range=[0.1, 1],
                                                            max_attempts=100,
                                                            use_image_if_no_bounding_boxes=False)


    new_image = tf.slice(image, im_box_begin, im_box_size)
    
    im_box_rank2 = tf.squeeze(im_box, axis=0) #[1 1 4] -> [1 4]
    im_box_rank1 = tf.squeeze(im_box) #[1 1 4] -> [4]

    #boxlist, inside_window_mask = _prune_completely_outside_window(bbox, im_box_rank1)
    overlapping_boxlist, keep_mask = _prune_non_overlapping_boxes(bbox, im_box_rank2, tf.reduce_min(random_option))

    new_bbox = _change_coordinate_frame(overlapping_boxlist, im_box_rank1)
    new_bbox = tf.clip_by_value(new_bbox, clip_value_min=0.0, clip_value_max=1.0)
    
    #new_label = tf.boolean_mask(class_id, inside_window_mask)
    new_label = tf.boolean_mask(class_id, keep_mask)

    return new_image, new_bbox, new_label

def colorJitter(image, p = 1.0):
    if tf.random.uniform([], minval=0, maxval=1) > p:
        return image

    image = tf.cast(image, tf.float32)/255.0

    image = tf.image.random_brightness(image, 0.2)
    image = tf.image.random_contrast(image, 0.8, 1.25)
    image = tf.image.random_hue(image, 0.05)
    image = tf.image.random_saturation(image, 0.5, 1.5)
    
    return tf.clip_by_value(image, 0.0, 1.0)*255.0



def mixUp(images_one, images_two, bboxes_one, bboxes_two, classes_one, classes_two):
    def _sample_beta_distribution(size, concentration_0=0.5, concentration_1=0.5):
        gamma_1_sample = tf.random.gamma(shape=[size], alpha=concentration_1)
        gamma_2_sample = tf.random.gamma(shape=[size], alpha=concentration_0)
        return gamma_1_sample / (gamma_1_sample + gamma_2_sample)

    images = images_one * 0.5 + images_two * (1 - 0.5)
    return images, tf.concat([bboxes_one, bboxes_two], 1), tf.concat([classes_one, classes_two], 1)


def randomExpand(image, bbox, expandMax=200):
    def _change_coordinate_frame(boxlist, window):
            win_height = window[2] - window[0]
            win_width = window[3] - window[1]
            boxlist_new = scale(boxlist - [window[0], window[1], window[0], window[1]],
                                1.0 / win_height, 
                                1.0 / win_width)
            return boxlist_new

    image_shape = tf.shape(image)
    image_height = image_shape[0]
    image_width = image_shape[1]

    min_image_size = tf.shape(image)[:2]
    max_image_size = min_image_size + tf.cast([expandMax, expandMax], dtype=tf.int32)

    target_height = tf.cond(
        max_image_size[0] > min_image_size[0],
        lambda: tf.random.uniform([], minval=min_image_size[0], maxval=max_image_size[0], dtype=tf.int32),
        lambda: max_image_size[0])

    target_width = tf.cond(
        max_image_size[1] > min_image_size[1],
        lambda: tf.random.uniform([], minval=min_image_size[1], maxval=max_image_size[1], dtype=tf.int32),
        lambda: max_image_size[1])


    offset_height = tf.cond(
        target_height > image_height,
        lambda: tf.random.uniform([], minval=0, maxval=target_height - image_height, dtype=tf.int32),
        lambda: tf.constant(0, dtype=tf.int32))

    offset_width = tf.cond(
        target_width > image_width,
        lambda: tf.random.uniform([], minval=0, maxval=target_width - image_width, dtype=tf.int32),
        lambda: tf.constant(0, dtype=tf.int32))

    new_image = tf.image.pad_to_bounding_box(image,
                                            offset_height=offset_height,
                                            offset_width=offset_width,
                                            target_height=target_height,
                                            target_width=target_width)

    '''image_ones = tf.ones_like(image)
    image_ones_padded = tf.image.pad_to_bounding_box(
        image_ones,
        offset_height=offset_height,
        offset_width=offset_width,
        target_height=target_height,
        target_width=target_width)
    image_color_padded = (1.0 - image_ones_padded) * pad_color
    new_image += image_color_padded'''

    new_window = tf.cast(tf.stack([-offset_height, -offset_width, target_height - offset_height, target_width - offset_width]),dtype=tf.float32)
    new_window /= tf.cast(tf.stack([image_height, image_width, image_height, image_width]), dtype=tf.float32)
    new_bbox = _change_coordinate_frame(bbox, new_window)
    return new_image, new_bbox

def mixUp(ds1, ds2):
    def _sample_beta_distribution(size, concentration_0=0.5, concentration_1=0.5):
        gamma_1_sample = tf.random.gamma(shape=[size], alpha=concentration_1)
        gamma_2_sample = tf.random.gamma(shape=[size], alpha=concentration_0)
        return gamma_1_sample / (gamma_1_sample + gamma_2_sample)
    
    images_one, bboxes_one, classes_one = ds1
    images_two, bboxes_two, classes_two = ds2

    images = images_one * 0.5 + images_two * (1 - 0.5)
    return images, tf.concat([bboxes_one, bboxes_two], 0), tf.concat([classes_one, classes_two], 0)

def mosaic(ds1, ds2, ds3, ds4):
    images1, bboxes1, classes1 = ds1
    images2, bboxes2, classes2 = ds2
    images3, bboxes3, classes3 = ds3
    images4, bboxes4, classes4 = ds4
    
    h, w, _ = tf.unstack(tf.shape(images1))

    images1, bboxes1, classes1  = randomCrop(images1, bboxes1, classes1)
    images2, bboxes2, classes2  = randomCrop(images2, bboxes2, classes2)
    images3, bboxes3, classes3  = randomCrop(images3, bboxes3, classes3)
    images4, bboxes4, classes4  = randomCrop(images4, bboxes4, classes4)
    
    border = tf.random.uniform([2], h//5*2, h//5*3, tf.int32)

    images1 = tf.image.resize(images1, [border[0], border[1]])
    images2 = tf.image.resize(images2, [border[0], w-border[1]])
    images3 = tf.image.resize(images3, [h-border[0], border[1]])
    images4 = tf.image.resize(images4, [h-border[0], w-border[1]])
    output_image = tf.concat([tf.concat([images1, images2], 1), tf.concat([images3, images4], 1)], 0)

    border = tf.cast(border/h, tf.float32)
    bboxes1 = tf.concat([border,border], -1)*bboxes1
    bboxes2 = tf.stack([border[0], 1-border[1], border[0], 1-border[1]], -1)*bboxes2+tf.stack([0.0,border[1],0.0,0.0], -1)
    bboxes3 = tf.stack([1-border[0], border[1], 1-border[0], border[1]], -1)*bboxes3+tf.stack([border[0],0.0,0.0,0.0], -1)
    bboxes4 = tf.stack([1-border[0], 1-border[1], 1-border[0], 1-border[1]], -1)*bboxes4+tf.stack([border[0],border[1],0.0,0.0], -1)

    output_boxes = tf.concat([bboxes1, bboxes2, bboxes3, bboxes4], 0)
    output_classes = tf.concat([classes1, classes2, classes3, classes4], 0)
    return output_image, output_boxes, output_classes