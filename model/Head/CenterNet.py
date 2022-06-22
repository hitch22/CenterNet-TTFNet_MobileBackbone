import tensorflow as tf
import numpy as np

from model.customLayer import Sigmoid, _SeparableConv, _Conv, ReLU, _SeparableDepthwiseConv

def _CenterNetHeatmapHead(x, filters, **config_dict):
	prefix="HeatmapHead/"
	classNum = config_dict["classNum"]
	config_dict.pop('classNum')

	x=_SeparableDepthwiseConv(x, filters=filters, kernel_size=3, strides=1, prefix=prefix+"Sep/", **config_dict)
	x=_Conv(x, filters=classNum, kernel_size=1, strides=1, normalization=None, activation=None, prefix=prefix+"Conv/", **config_dict)
	return x

def _CenterNetSizepHead(x, filters, **config_dict):
	prefix="SizeHead/"
	box_coord_size = 4 # center:2
	weights = 16.0 # center:1
	x=_SeparableDepthwiseConv(x, filters=filters, kernel_size=3, strides=1, prefix=prefix+"Sep/", **config_dict)
	x=_Conv(x, filters=box_coord_size, kernel_size=1, strides=1, normalization=None, activation=ReLU, prefix=prefix+"Conv/", **config_dict)
	return x * weights

def CenterNet(x, config=None):
	Size_config_dict = {
        'kernel_regularizer': tf.keras.regularizers.l2(config["model_config"]["head"]["regularization"]),
        'kernel_initializer': tf.initializers.RandomNormal(mean=0.0, stddev=0.03),
        'trainable':not config["model_config"]["head"]["isFreeze"],
        'use_bias':True,
    }

	Heatmap_config_dict = {
        'kernel_regularizer': tf.keras.regularizers.l2(config["model_config"]["head"]["regularization"]),
        'kernel_initializer': tf.initializers.RandomNormal(mean=0.0, stddev=0.03),
        'bias_initializer': tf.constant_initializer(-2.19),
        'trainable':not config["model_config"]["head"]["isFreeze"],
        'use_bias':True,
        'classNum':config["training_config"]["num_classes"],
    }

	Headmap_outputs=_CenterNetHeatmapHead(x, config["model_config"]["head"]["filters"], **Heatmap_config_dict)
	Size_outputs=_CenterNetSizepHead(x, config["model_config"]["head"]["filters"], **Size_config_dict)

	return tf.keras.layers.Concatenate(axis=-1, name="LastConcat")([Headmap_outputs, Size_outputs])
