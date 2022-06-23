import weakref
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

def reduce_losses(losses_dict):
    for key, value in losses_dict.items():
        losses_dict[key] = tf.reduce_mean(value)
    return losses_dict

def nms(heat, kernel=3):
    heat_max = tf.keras.layers.MaxPooling2D(kernel, 1, padding="same")(heat)
    heat_max_mask = tf.math.abs(heat - heat_max) < 1e-4
    heat_max_peak = tf.where(heat_max_mask, heat_max, 0.0)
    return heat_max_peak

def decode(detections, k=100, relative=False):
    heatmap=tf.nn.sigmoid(detections[..., :80])
    wh=detections[..., 80:84]

    heat_max_peak = nms(heatmap)
    batch, height, width, cat = tf.unstack(tf.shape(heatmap))
    
    heat_max_peak_flat = tf.reshape(heat_max_peak, (batch, -1))
    topk_scores, topk_inds = tf.math.top_k(heat_max_peak_flat, k=k)

    topk_clf = topk_inds % cat
    topk_xs = tf.cast(topk_inds // cat % width, tf.float32)
    topk_ys = tf.cast(topk_inds // cat // width, tf.float32)
    topk_inds = tf.cast(topk_ys * tf.cast(width, tf.float32) + topk_xs, tf.int32)

    #scores, inds, clf, ys, xs = topk(heat_max_peak_flat, k=k)

    ys = tf.expand_dims(topk_ys, axis=-1)
    xs = tf.expand_dims(topk_xs, axis=-1)

    if False: #center
        reg = tf.reshape(reg, (batch, -1, tf.shape(reg)[-1]))
        reg = tf.gather(reg, inds, axis=1, batch_dims=-1)
        ys, xs = ys + reg[..., 0:1], xs + reg[..., 1:2]

    wh = tf.reshape(wh, (batch, -1, tf.shape(wh)[-1]))
    wh = tf.gather(wh, topk_inds, axis=1, batch_dims=-1)

    clf = tf.cast(tf.expand_dims(topk_clf, axis=-1), tf.float32)
    scores = tf.expand_dims(topk_scores, axis=-1)

    if False: #center
        wh = tf.math.abs(wh)
        ymin = ys - wh[..., 0:1] / 2
        xmin = xs - wh[..., 1:2] / 2
        ymax = ys + wh[..., 0:1] / 2
        xmax = xs + wh[..., 1:2] / 2
    elif True:
        #ys/=height
        #xs/=width
        ymin = ys - wh[..., 0:1]
        xmin = xs - wh[..., 1:2]
        ymax = ys + wh[..., 2:3]
        xmax = xs + wh[..., 3:4]

    bboxes = tf.concat([ymin, xmin, ymax, xmax], axis=-1)

    if True:
        bboxes /= tf.cast(tf.stack([height, width, height, width]), dtype=tf.float32)

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

    def compile(self, loss, optimizer, **kwargs):
        super().compile(**kwargs)
        self.loss_fn=loss
        self.optimizer=optimizer

    def train_step(self, data):
        images, y_true=data

        with tf.GradientTape() as tape:
            y_pred=self(images, training=True)
            loss_values=self.loss_fn(y_true, y_pred)

            heat_loss=loss_values[0]    #[batch]
            loc_loss=loss_values[1]    #[batch]

            loss=heat_loss+loc_loss
            _scaled_losses=get_scaled_losses(loss, self.losses)
        
        scaled_gradients=tape.gradient(_scaled_losses, self.trainable_variables)
        self.optimizer.apply_gradients(zip(scaled_gradients, self.trainable_variables))

        loss_dict={
                    'HeatL': heat_loss,
                    'BoxL': loc_loss,
                    'RegL': self.losses,
                    'TotalL': loss
                }
        
        return reduce_losses(loss_dict)

    def test_step(self, data):
        images, y_true, _ = data
        
        y_pred=self(images, training=False)
        loss_values=self.loss_fn(y_true, y_pred)
        heat_loss=loss_values[0]
        loc_loss=loss_values[1]

        loss=heat_loss+loc_loss

        loss_dict={
                    'HeatL': heat_loss,
                    'BoxL': loc_loss,
                    'TotalL': loss
                }
                    
        return reduce_losses(loss_dict)

    def predict_step(self, images):
        predictions=self(images, training=False)
        return decode(predictions)