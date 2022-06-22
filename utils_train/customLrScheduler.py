import tensorflow as tf

class CosineDecayWithLinearWarmup(tf.keras.optimizers.schedules.CosineDecay):
    def __init__(
            self,
            initial_learning_rate,
            warmup_learning_rate,
            warmup_steps,
            total_steps,
            alpha=0.0,
            **kwargs):
        decay_steps_with_warmup = total_steps - warmup_steps
        super().__init__(initial_learning_rate=initial_learning_rate,
                        decay_steps=decay_steps_with_warmup,
                        alpha=alpha,
                        name='CosineDecayWarmUp',
                        **kwargs)

        self.warmup_learning_rate = warmup_learning_rate
        self.warmup_steps = warmup_steps
        self._step_size = initial_learning_rate - self.warmup_learning_rate

    def __call__(self, step):
        learning_rate = tf.cond(tf.less(tf.cast(step, dtype=tf.float32), self.warmup_steps),
            lambda: (self.warmup_learning_rate + tf.cast(step, dtype=tf.float32) / self.warmup_steps * self._step_size),
            lambda: (super(CosineDecayWithLinearWarmup, self).__call__(step)))
        return learning_rate

    def get_config(self):
        config = {
            "warmup_learning_rate": self.warmup_learning_rate,
            "warmup_steps": self.warmup_steps,
        }
        base_config = super(CosineDecayWithLinearWarmup,self).get_config()
        return dict(list(base_config.items()) + list(config.items()))