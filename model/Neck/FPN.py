import tensorflow as tf
from tensorflow.keras.layers import UpSampling2D, Add
from model.customLayer import _Conv, _SeparableConv, _SeparableDepthwiseConv

def FPN(x, config=None):
    isLite=config["model_config"]["neck"]["isLite"]
    filters=config["model_config"]["neck"]["filters"]

    config_dict = {
        'kernel_regularizer': tf.keras.regularizers.l2(config["model_config"]["neck"]["regularization"]),
        'kernel_initializer': tf.initializers.RandomNormal(mean=0.0, stddev=0.03),
        'trainable':not config["model_config"]["backbone"]["isFreeze"],
        'use_bias':False
    }
    
    C2, C3, C4, C5=x #stride 4 8 16 32// 80 40 20 10
    #C5 = tf.keras.layers.Dropout(rate=0.5)(C5)
    #C4 = tf.keras.layers.Dropout(rate=0.4)(C4)
    #C3 = tf.keras.layers.Dropout(rate=0.3)(C3)
    #C2 = tf.keras.layers.Dropout(rate=0.2)(C2)
    ratio = 0.5

    P5=_Conv(C5, filters=filters[0], kernel_size=1, strides=1, prefix="C5Residual/", **config_dict)

    P5_upsampled=UpSampling2D(name='P5_upsample')(P5)
    P4=_Conv(C4, filters=filters[0], kernel_size=1, strides=1, prefix="C4/", **config_dict)
    P4=Add(name='P4_merged')([P5_upsampled*(1.0-ratio), P4*ratio])
    if isLite:
        P4=_SeparableDepthwiseConv(P4, filters=filters[1], kernel_size=3, strides=1, prefix="P4/", **config_dict)
    else:
        P4=_Conv(P4, filters=filters[1], kernel_size=3, strides=1, prefix="P4/", **config_dict)

    P4_upsampled=UpSampling2D(name='P4_upsample')(P4)
    P3=_Conv(C3, filters=filters[1], kernel_size=1, strides=1, prefix="C3/", **config_dict)
    P3=Add(name='P3_merged')([P4_upsampled*(1.0-ratio), P3*ratio])
    if isLite:
        P3=_SeparableDepthwiseConv(P3, filters=filters[2], kernel_size=3, strides=1, prefix="P3/", **config_dict)
    else:
        P3=_Conv(P3, filters=filters[2], kernel_size=3, strides=1, prefix="P3/", **config_dict)

    P3_upsampled=UpSampling2D(name='P3_upsample')(P3)
    P2=_Conv(C2, filters=filters[2], kernel_size=1, strides=1, prefix="C2/", **config_dict)
    P2=Add(name='P2_merged')([P3_upsampled*(1.0-ratio), P2*ratio])
    if isLite:
        P2=_SeparableDepthwiseConv(P2, filters=filters[3], kernel_size=3, strides=1, prefix="P2/", **config_dict)
    else:
        P2=_Conv(P2, filters=filters[3], kernel_size=3, strides=1, prefix="P2/", **config_dict)

    return P2