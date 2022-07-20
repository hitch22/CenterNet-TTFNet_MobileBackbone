import tensorflow as tf
from tensorflow.keras.layers import UpSampling2D, Add
from model.customLayer import _Conv, _SeparableConv, _SeparableDepthwiseConv

def FPN(x, config=None):
    isLite=config["model_config"]["neck"]["isLite"]
    filters=config["model_config"]["neck"]["filters"]

    config_dict = {
        'kernel_regularizer': tf.keras.regularizers.l2(config["model_config"]["neck"]["regularization"]),
        'kernel_initializer': tf.initializers.TruncatedNormal(mean=0.0, stddev=0.03),
        'trainable':not config["model_config"]["backbone"]["isFreeze"],
        'use_bias':False
    }
    
    C2, C3, C4, C5=x
    ##################################################
    P5=_Conv(C5, filters=filters[0], kernel_size=1, strides=1, prefix="C5P5/", **config_dict)
    P5_upsampled=UpSampling2D(name='P5_upsample')(P5)

    P4=_Conv(C4, filters=filters[0], kernel_size=1, strides=1, prefix="C4P4/", **config_dict)
    P4=Add(name='P5P4_merge')([P5_upsampled, P4])
    ##################################################

    ##################################################
    if isLite:
        P4=_SeparableConv(P4, filters=filters[1], kernel_size=3, strides=1, prefix="P4_project/", **config_dict)
    else:
        P4=_Conv(P4, filters=filters[1], kernel_size=3, strides=1, prefix="P4_project/", **config_dict)
    P4_upsampled=UpSampling2D(name='P4_upsample')(P4)

    P3=_Conv(C3, filters=filters[1], kernel_size=1, strides=1, prefix="C3P3/", **config_dict)
    P3=Add(name='P4P3_merge')([P4_upsampled, P3])
    ##################################################

    ##################################################
    if isLite:
        P3=_SeparableConv(P3, filters=filters[2], kernel_size=3, strides=1, prefix="P3_project/", **config_dict)
    else:
        P3=_Conv(P3, filters=filters[2], kernel_size=3, strides=1, prefix="P3_project/", **config_dict)
    P3_upsampled=UpSampling2D(name='P3_upsample')(P3)

    P2=_Conv(C2, filters=filters[2], kernel_size=1, strides=1, prefix="C2P2/", **config_dict)
    P2=Add(name='C2P2_merge')([P3_upsampled, P2])
    ##################################################
    
    if isLite:
        P2=_SeparableConv(P2, filters=filters[3], kernel_size=3, strides=1, prefix="P2_project/", **config_dict)
    else:
        P2=_Conv(P2, filters=filters[3], kernel_size=3, strides=1, prefix="P2_project/", **config_dict)

    return P2