import numpy as np
import tensorflow as tf

from model.BackBone.builder import BackBoneBuild
from model.Neck.builder import NeckBuild
from model.Head.builder import HeadBuild

_policy=tf.keras.mixed_precision.global_policy()

def get_scaled_losses(loss, regularization_losses=None):
    loss = tf.reduce_mean(loss)
    if regularization_losses:
        loss = loss + tf.math.add_n(regularization_losses)
    return loss

def nms(heat, kernel=3):
    heat_max = tf.keras.layers.MaxPooling2D(kernel, 1, padding="same", dtype = _policy.compute_dtype)(heat)
    heat_max_mask = tf.math.abs(heat - heat_max) < 1e-4 #32 16
    heat_max_peak = tf.where(heat_max_mask, heat_max, 0.0)
    return heat_max_peak

def decode(detections, k=100, isCenter=False):
    heatmap=detections[..., :80]
    wh=detections[..., 80:84]

    heat_max_peak = nms(heatmap)
    batch, height, width, cat = tf.unstack(tf.shape(heatmap))

    heat_max_peak_flat = tf.reshape(heat_max_peak, (batch, -1))
    topk_scores, topk_inds = tf.math.top_k(heat_max_peak_flat, k=k)

    topk_clf = topk_inds % cat
    topk_xs = tf.cast(topk_inds // cat % width, tf.float32)
    topk_ys = tf.cast(topk_inds // cat // width, tf.float32)
    topk_inds = tf.cast(topk_ys * tf.cast(width, tf.float32) + topk_xs, tf.int32)

    ys = tf.expand_dims(topk_ys, axis=-1)
    xs = tf.expand_dims(topk_xs, axis=-1)

    if isCenter:
        reg = tf.reshape(reg, (batch, -1, tf.shape(reg)[-1]))
        reg = tf.gather(reg, topk_inds, axis=1, batch_dims=-1)
        ys, xs = ys + reg[..., 0:1], xs + reg[..., 1:2]

    wh = tf.reshape(wh, (batch, -1, tf.shape(wh)[-1]))
    wh = tf.gather(wh, topk_inds, axis=1, batch_dims=-1)

    clf = tf.cast(tf.expand_dims(topk_clf, axis=-1), tf.float32)
    scores = tf.expand_dims(topk_scores, axis=-1)

    if isCenter:
        wh = tf.math.abs(wh)
        ymin = ys - wh[..., 0:1] / 2
        xmin = xs - wh[..., 1:2] / 2
        ymax = ys + wh[..., 0:1] / 2
        xmax = xs + wh[..., 1:2] / 2
    else:
        height = tf.cast(height, tf.float32)
        width = tf.cast(width, tf.float32)
        
        ys/=height
        xs/=width
        ymin = ys - wh[..., 0:1]
        xmin = xs - wh[..., 1:2]
        ymax = ys + wh[..., 2:3]
        xmax = xs + wh[..., 3:4]

    bboxes = tf.concat([ymin, xmin, ymax, xmax], axis=-1)
    detections = tf.concat([bboxes, clf, scores], axis=-1)
    return detections

class ModelBuilder(tf.keras.Model):
    def __init__(self, config, **kwargs):
        backbone=BackBoneBuild(config)
        neck=NeckBuild(config)
        head=HeadBuild(config)

        inputs=tf.keras.Input((config["model_config"]["target_size"], config["model_config"]["target_size"], 3), name="inputlayer")
        features=backbone(inputs, config)
        features=neck(features, config)
        outputs=head(features, config)

        super().__init__(inputs=[inputs],
                        outputs=outputs,
                        name='Detector')

        self.config=config
        self.isCenter = config["model_config"]["head"]["name"].upper() in ["CENTERNET"]
        self.total_loss_tracker = tf.keras.metrics.Mean(name="TotalL")
        self.heat_loss_tracker = tf.keras.metrics.Mean(name="HeatL")
        self.box_loss_tracker = tf.keras.metrics.Mean(name="BoxL")

    def compile(self, loss, optimizer, **kwargs):
        super().compile(**kwargs)
        self.loss_fn=loss
        self.optimizer=optimizer

    def train_step(self, data):
        images, y_true=data

        with tf.GradientTape() as tape:
            y_pred=self(images, training=True)
            loss_values=self.loss_fn(y_true, y_pred)

            heat_loss=loss_values[0]
            box_loss=loss_values[1]

            total_loss=heat_loss+box_loss
            _scaled_losses=get_scaled_losses(total_loss, self.losses)
            _scaled_losses=self.optimizer.get_scaled_loss(_scaled_losses)
        
        scaled_gradients=tape.gradient(_scaled_losses, self.trainable_variables)
        scaled_gradients=self.optimizer.get_unscaled_gradients(scaled_gradients)
        self.optimizer.apply_gradients(zip(scaled_gradients, self.trainable_variables))

        self.total_loss_tracker.update_state(total_loss)
        self.heat_loss_tracker.update_state(heat_loss)
        self.box_loss_tracker.update_state(box_loss)

        loss_dict={
                    'HeatL': self.heat_loss_tracker.result(),
                    'BoxL': self.box_loss_tracker.result(),
                    'RegL': tf.reduce_mean(self.losses),
                    'TotalL': self.total_loss_tracker.result()
                }
        
        return loss_dict

    def test_step(self, data):
        images, y_true, _ = data
        
        y_pred=self(images, training=False)
        loss_values=self.loss_fn(y_true, y_pred)
        
        heat_loss=loss_values[0]
        box_loss=loss_values[1]

        total_loss=heat_loss+box_loss

        self.total_loss_tracker.update_state(total_loss)
        self.heat_loss_tracker.update_state(heat_loss)
        self.box_loss_tracker.update_state(box_loss)

        loss_dict={
                    'HeatL': self.heat_loss_tracker.result(),
                    'BoxL': self.box_loss_tracker.result(),
                    'TotalL': self.total_loss_tracker.result()
                }
                    
        return loss_dict

    def predict_step(self, images):
        predictions=self(images, training=False)
        return decode(predictions, isCenter = self.isCenter)

    def __repr__(self, table=False):
        print_str=''
        if table:
            print_str += '%25s | %16s | %20s | %10s | %16s | %6s | %6s\n' % (
                'Layer Name', 'Input Shape', 'Output Shape', 'Kernel Size', 'Filters', 'Strides', 'FLOPS')
            print_str += '-' * 170
            print_str += '\n'
        t_flops = 0
        t_macc = 0

        for l in self.layers:
            o_shape, i_shape, strides, ks, filters = ['', '', ''], ['', '', ''], [1, 1], 'N/A', 'N/A'
            flops = 0
            macc = 0
            name = l.name

            if ('InputLayer' in str(l)):
                i_shape = l.input.get_shape()[1:4].as_list()
                o_shape = i_shape

            if ('Reshape' in str(l)):
                i_shape = l.input.get_shape()[1:4].as_list()
                o_shape = l.output.get_shape()[1:4].as_list()

            if ('Add' in str(l) or 'Maximum' in str(l) or 'Concatenate' in str(l)):
                i_shape = l.input[0].get_shape()[1:4].as_list() + [len(l.input)]
                o_shape = l.output.get_shape()[1:4].as_list()
                flops = (len(l.input) - 1) * i_shape[0] * i_shape[1] * i_shape[2]

            if ('Average' in str(l) and 'pool' not in str(l)):
                i_shape = l.input[0].get_shape()[1:4].as_list() + [len(l.input)]
                o_shape = l.output.get_shape()[1:4].as_list()
                flops = len(l.input) * i_shape[0] * i_shape[1] * i_shape[2]

            if ('BatchNormalization' in str(l)):
                i_shape = l.input.get_shape()[1:4].as_list()
                o_shape = l.output.get_shape()[1:4].as_list()

                bflops = 1
                for i in range(len(i_shape)):
                    bflops *= i_shape[i]
                flops = bflops

            if ('ReLU6' in str(l) or 'HSwish' in str(l)):
                i_shape = l.input.get_shape()[1:4].as_list()
                o_shape = l.output.get_shape()[1:4].as_list()
                bflops = 1
                for i in range(len(i_shape)):
                    bflops *= i_shape[i]
                flops = bflops

            if ('pool' in str(l) and ('Global' not in str(l))):
                i_shape = l.input.get_shape()[1:4].as_list()
                strides = l.strides
                ks = l.pool_size
                flops = ((i_shape[0] / strides[0]) * (i_shape[1] / strides[1]) * (ks[0] * ks[1] * i_shape[2]))

            if ('Flatten' in str(l)):
                i_shape = l.input.shape[1:4].as_list()
                flops = 1
                out_vec = 1
                for i in range(len(i_shape)):
                    flops *= i_shape[i]
                    out_vec *= i_shape[i]
                o_shape = flops
                flops = 0

            if ('Dense' in str(l)):
                i_shape = l.input.shape[1:4].as_list()[0]
                if (i_shape == None):
                    i_shape = out_vec

                o_shape = l.output.shape[1:4].as_list()
                flops = 2 * (o_shape[0] * i_shape)
                macc = flops / 2

            if ('Padding' in str(l)):
                flops = 0

            if (('Global' in str(l))):
                i_shape = l.input.get_shape()[1:4].as_list()
                flops = ((i_shape[0]) * (i_shape[1]) * (i_shape[2]))
                o_shape = [l.output.get_shape()[1:4].as_list(), 1, 1]
                out_vec = o_shape

            if ('Conv2D ' in str(l) and 'DepthwiseConv2D' not in str(l) and 'SeparableConv2D' not in str(l)):
                strides = l.strides
                ks = l.kernel_size
                filters = l.filters
                i_shape = l.input.get_shape()[1:4].as_list()
                o_shape = l.output.get_shape()[1:4].as_list()

                if (filters == None):
                    filters = i_shape[2]

                flops = 2 * ((filters * ks[0] * ks[1] * i_shape[2]) * (
                        (i_shape[0] / strides[0]) * (i_shape[1] / strides[1])))
                macc = flops / 2

            if ('Conv2D ' in str(l) and 'DepthwiseConv2D' in str(l) and 'SeparableConv2D' not in str(l)):
                strides = l.strides
                ks = l.kernel_size
                filters = l.filters
                i_shape = l.input.get_shape()[1:4].as_list()
                o_shape = l.output.get_shape()[1:4].as_list()

                if (filters == None):
                    filters = i_shape[2]

                flops = 2 * (
                        (ks[0] * ks[1] * i_shape[2]) * ((i_shape[0] / strides[0]) * (i_shape[1] / strides[1]))) 
                macc = flops / 2

            t_macc += macc
            t_flops += flops

            if table:
                print_str += '%25s | %16s | %20s | %10s | %16s | %6s | %5.2f[M]\n' % (
                    name, str(i_shape), str(o_shape), str(ks), str(filters), str(strides), flops/1e6)

        trainable_params = sum([np.prod(w.get_shape().as_list()) for w in self.trainable_weights])
        none_trainable_params = sum([np.prod(w.get_shape().as_list()) for w in self.non_trainable_weights])
        total_params = trainable_params+none_trainable_params

        print_str += 'Total Params: %6.3f[M]\n' % (total_params/1e6)+'Trainable Params: %6.3f[M]' % (trainable_params/1e6)+ \
        '\nTotal FLOPS: %6.3f[G]\n' % (t_flops/1e9)+'Total MACCs: %6.3f[G]' % (t_macc/1e9)
        return print_str
