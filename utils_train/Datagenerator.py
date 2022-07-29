import tensorflow as tf
import tensorflow_datasets as tfds
import keras_cv

from utils_train.Augmentation import *
from utils_train.Encoder import LabelEncoder

class DatasetBuilder():
    def __init__(self, config, mode='train'):
        assert(mode in ['train', 'validation', 'bboxtest'])
        self._dataset = None

        self._label_encoder = LabelEncoder(config)
        self._target_size = config["model_config"]["target_size"]
        self._batch_size = config["training_config"]["batch_size"]
        self.mode = mode
        self._build_dataset()
    
    def _prepare_proto(self, samples):
        pass
    
    def __len__(self):
        pass

    def _preprocess_before_batch(self, samples):
        '''
            in_bbox_format: [ymin xmin ymax xmax]
            out_bbox_format: [cy cx h w]
        '''

        image, bbox, classes, inferMetric = self._prepare_proto(samples)

        if self.mode == 'train' or self.mode == 'bboxtest':
            image, bbox, classes  = randomCrop(image, bbox, classes, p = 0.75)
            image, bbox           = randomExpand(image, bbox, expandMax=200.0)
            image, bbox           = randomResize(image, bbox, self._target_size, self._target_size, p = 0.0)
            #image                 = randomCutout(image)
            image, bbox           = flipHorizontal(image, bbox, p = 0.5)
            #image                 = randomGaussian(image)
            image                 = colorJitter(image, p = 0.7)

        else:
            image, bbox           = randomResize(image, bbox, self._target_size, self._target_size, p = 0.0)


        if self.mode == 'train':
            return (image/127.5) -1.0,  self._label_encoder._encode_sample(bbox, classes)
        else:
            return (image/127.5) -1.0, self._label_encoder._encode_sample(bbox, classes), inferMetric

    def _build_dataset(self):
        pass
            
    @property
    def dataset(self):
        return self._dataset

class Dataset_COCO(DatasetBuilder):
    def __init__(self, config, mode='train'):
        super().__init__(config, mode)

    def _prepare_proto(self, samples):
        image = samples["image"]
        originalShape = tf.shape(image)[:2]
        classes = tf.cast(samples["objects"]["label"], dtype=tf.int32)
        bbox = samples["objects"]["bbox"]
        ####################################
        noCrowMask =  tf.logical_not(samples["objects"]["is_crowd"])
        classes = tf.boolean_mask(classes, noCrowMask)
        bbox = tf.boolean_mask(bbox, noCrowMask)
        ####################################
        validboxMask = tf.reduce_all(bbox[..., 2:] > bbox[..., :2], -1)
        classes = tf.boolean_mask(classes, validboxMask)
        bbox = tf.boolean_mask(bbox, validboxMask)

        cocoLabel = {"original_shape": originalShape, "image_id": samples['image/id']}

        return image, bbox, classes, cocoLabel

    def __len__(self):
        return int(117266/self._batch_size)

    def _build_dataset(self):
        if self.mode == 'train' or self.mode == 'bboxtest':
             self._tfrecords, dataset_info = tfds.load(name="coco/2017", split="train", with_info=True, shuffle_files=True)
             self.labelMapFunc = dataset_info.features["objects"]["label"].int2str
        else:
             self._tfrecords = tfds.load(name="coco/2017", split='validation', with_info=False, shuffle_files=False)

        self._tfrecords = self._tfrecords.filter(lambda samples: len(samples["objects"]["label"]) >= 1) #117266 #4952  and tf.reduce_any(samples["objects"]["label"] == 0)
        
        if self.mode == 'train':
            self._dataset = (
                self._tfrecords
                .shuffle(8*self._batch_size, reshuffle_each_iteration=False)
                .map(self._preprocess_before_batch, num_parallel_calls=tf.data.AUTOTUNE)
                .batch(batch_size=self._batch_size, drop_remainder = True)
                .prefetch(tf.data.AUTOTUNE)
            )

        elif self.mode == 'validation':
            self._dataset = (
                self._tfrecords
                .map(self._preprocess_before_batch, num_parallel_calls=tf.data.AUTOTUNE)
                .batch(batch_size=self._batch_size, drop_remainder = False)
                .prefetch(tf.data.AUTOTUNE)
            )
        else:
            self._dataset = (
                self._tfrecords
                .map(self._preprocess_before_batch, num_parallel_calls=tf.data.AUTOTUNE)
            )

class Datase_Custom(DatasetBuilder):
    def __init__(self,  config, mode='train'):
        super().__init__(config, mode)

    def _prepare_proto(self, samples):
        feature_description = {
            'image/encoded': tf.io.FixedLenFeature([], tf.string),
            'image/object/bbox/xmin': tf.io.VarLenFeature(tf.float32),
            'image/object/bbox/ymin': tf.io.VarLenFeature(tf.float32),
            'image/object/bbox/xmax': tf.io.VarLenFeature(tf.float32),
            'image/object/bbox/ymax': tf.io.VarLenFeature(tf.float32),
            'image/object/class/label': tf.io.VarLenFeature(tf.int64),
        }

        parsed_example = tf.io.parse_single_example(samples,
                                                    feature_description)
        classes = tf.sparse.to_dense(parsed_example['image/object/class/label'])
        classes = tf.cast(classes, tf.int32)-1

        image = tf.io.decode_image(parsed_example['image/encoded'], channels=3)
        image = tf.cast(image, dtype=tf.uint8)
        image.set_shape([None, None, 3])

        bbox = tf.stack([
            tf.sparse.to_dense(parsed_example['image/object/bbox/ymin']),
            tf.sparse.to_dense(parsed_example['image/object/bbox/xmin']),
            tf.sparse.to_dense(parsed_example['image/object/bbox/ymax']),
            tf.sparse.to_dense(parsed_example['image/object/bbox/xmax']),
        ], axis=-1)

        return image, bbox, classes, None
   
    def _build_dataset(self):
        self._tfrecords = tf.data.Dataset.list_files("data/P.tfrecord")
        if self.mode == 'train':
            options = tf.data.Options()
            options.deterministic = False
            self._dataset = (
                self._tfrecords.interleave(tf.data.TFRecordDataset,
                                            cycle_length=256,
                                            block_length=16,
                                            num_parallel_calls=tf.data.AUTOTUNE)
                .with_options(options)
                .shuffle(8*self._batch_size)
                .map(self._preprocess_before_batch, num_parallel_calls=tf.data.AUTOTUNE)
                .batch(batch_size=self._batch_size, drop_remainder = False)
                .prefetch(tf.data.AUTOTUNE)
            )

        else:
            self._dataset = (
                self._tfrecords.interleave(tf.data.TFRecordDataset,
                                            cycle_length=256,
                                            block_length=16,
                                            num_parallel_calls=tf.data.AUTOTUNE)
                .map(self._preprocess, num_parallel_calls=tf.data.AUTOTUNE)
            )

class Dataset_COCO_Temp(DatasetBuilder):
    def __init__(self, config, mode='train'):
        super().__init__(config, mode)
    
    def _prepare_proto(self, samples):
        image = samples["image"]
        originalShape = tf.shape(image)[:2]
        classes = tf.cast(samples["objects"]["label"], dtype=tf.int32)
        bbox = samples["objects"]["bbox"]
        ####################################
        noCrowMask =  tf.logical_not(samples["objects"]["is_crowd"])
        classes = tf.boolean_mask(classes, noCrowMask)
        bbox = tf.boolean_mask(bbox, noCrowMask)
        ####################################
        validboxMask = tf.reduce_all(bbox[..., 2:] > bbox[..., :2], -1)
        classes = tf.boolean_mask(classes, validboxMask)
        bbox = tf.boolean_mask(bbox, validboxMask)

        cocoLabel = {"original_shape": originalShape, "image_id": samples['image/id']}

        return image, bbox, classes, cocoLabel

    def _preprocess_before_batch(self, samples):
        '''
            in_bbox_format: [ymin xmin ymax xmax]
            out_bbox_format: [cy cx h w]
        '''

        image, bbox, classes, inferMetric = self._prepare_proto(samples)

        if self.mode == 'train' or self.mode == 'bboxtest':
            #image                 = colorJitter(image, p = 0.7)
            #image, bbox           = randomExpand(image, bbox, expandMax=200.0)
            image, bbox           = randomResize(image, bbox, self._target_size, self._target_size, p = 0.0)
            image, bbox           = flipHorizontal(image, bbox, p = 0.5)
            
        else:
            image, bbox           = randomResize(image, bbox, self._target_size, self._target_size, p = 0.0)


        if self.mode == 'train':
            return (image/127.5) -1.0,  bbox, classes
        else:
            return (image/127.5) -1.0, self._label_encoder._encode_sample(bbox, classes), inferMetric

    def _preprocess_after_batch(self, ds1, ds2, ds3, ds4):
        inner_p = tf.random.uniform([], minval=0, maxval=1)

        image, bbox, classes = mosaic(ds1, ds2, ds3, ds4)

        #return image, self._label_encoder._encode_sample(bbox, classes)
        return image, bbox, classes

    def _build_dataset(self):
        if self.mode == 'train' or self.mode == 'bboxtest':
             self._tfrecords, dataset_info = tfds.load(name="coco/2017", split="train", with_info=True, shuffle_files=True)
             self.labelMapFunc = dataset_info.features["objects"]["label"].int2str
        else:
             self._tfrecords = tfds.load(name="coco/2017", split='validation', with_info=False, shuffle_files=False)

        self._tfrecords = self._tfrecords.filter(lambda samples: len(samples["objects"]["label"]) >= 1) #117266 #4952  and tf.reduce_any(samples["objects"]["label"] == 0)
        
        if self.mode == 'train' or self.mode == 'bboxtest':
            ds1 = (
                self._tfrecords
                .shuffle(2*self._batch_size)
                .map(self._preprocess_before_batch, num_parallel_calls=tf.data.AUTOTUNE)
            )
            ds2 = (
                self._tfrecords
                .shuffle(2*self._batch_size)
                .map(self._preprocess_before_batch, num_parallel_calls=tf.data.AUTOTUNE)
            )
            ds3 = (
                self._tfrecords
                .shuffle(2*self._batch_size)
                .map(self._preprocess_before_batch, num_parallel_calls=tf.data.AUTOTUNE)
            )
            ds4 = (
                self._tfrecords
                .shuffle(2*self._batch_size)
                .map(self._preprocess_before_batch, num_parallel_calls=tf.data.AUTOTUNE)
            )
            self._dataset = (
                tf.data.Dataset.zip((ds1, ds2, ds3, ds4))
                .map(self._preprocess_after_batch, num_parallel_calls=tf.data.AUTOTUNE)
                .batch(self._batch_size)
                .prefetch(tf.data.AUTOTUNE)
            )

        elif self.mode == 'validation':
            self._dataset = (
                self._tfrecords
                .map(self._preprocess_before_batch, num_parallel_calls=tf.data.AUTOTUNE)
                .batch(batch_size=self._batch_size, drop_remainder = False)
                .prefetch(tf.data.AUTOTUNE)
            )


