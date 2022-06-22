import tensorflow as tf
import numpy as np

from model.customLayer import Sigmoid, _SeparableConv, _Conv, ReLU

def _CenterNetHeatmapHead(x, filters, **config_dict):
	prefix="HeatmapHead/"
	x=_SeparableConv(x, filters=filters, kernel_size=3, strides=1, prefix=prefix+"Sep/", **config_dict)
	x=_Conv(x, filters=config_dict["classNum"], kernel_size=1, strides=1, use_bias=True, normalization=None, activation=None, prefix=prefix+"Conv/", **config_dict)
	return x

def _CenterNetSizepHead(x, filters, **config_dict):
	prefix="SizeHead/"
	box_coord_size = 4 # center:2
	weights = 16.0 # center:1
	x=_SeparableConv(x, filters=filters, kernel_size=3, strides=1, prefix=prefix+"Sep/", **config_dict)
	x=_Conv(x, filters=box_coord_size, kernel_size=1, strides=1, use_bias=False, normalization=None, activation=ReLU, prefix=prefix+"Conv/", **config_dict)
	return x * weights

def _CenterNetOffsetpHead(x, filters, **config_dict):
	prefix="OffsetHead/"
	x=_SeparableConv(x, filters=filters, kernel_size=3, strides=1, prefix=prefix+"Sep/", **config_dict)
	x=_Conv(x, filters=2, kernel_size=1, strides=1, use_bias=True, normalization=None, activation=None, prefix=prefix+"Conv/", **config_dict)
	return x

def CenterNet(x, config=None):
	Size_config_dict={
			'reg': config["model_config"]["head"]["regularization"],
			'bias_initializer': tf.constant_initializer(0.0),
        	'trainable': not config["model_config"]["neck"]["isFreeze"]
			}

	Heatmap_config_dict={
			'reg': config["model_config"]["head"]["regularization"],
			'bias_initializer': tf.constant_initializer(-4.6),
			'trainable': not config["model_config"]["neck"]["isFreeze"],
			'classNum': config["training_config"]["num_classes"]
			}

	Offset_config_dict={
			'reg': config["model_config"]["head"]["regularization"],
			'trainable': not config["model_config"]["neck"]["isFreeze"]
 			}

	Headmap_outputs=_CenterNetHeatmapHead(x, config["model_config"]["head"]["filters"], **Heatmap_config_dict)
	Size_outputs=_CenterNetSizepHead(x, config["model_config"]["head"]["filters"], **Size_config_dict)
	#Offset_outputs=_CenterNetOffsetpHead(x, config["model_config"]["head"]["filters"], **Offset_config_dict) #center

	return tf.keras.layers.Concatenate(axis=-1, name="LastConcat")([Headmap_outputs, Size_outputs])
