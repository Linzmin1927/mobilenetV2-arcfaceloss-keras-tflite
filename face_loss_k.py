import os
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer
from tensorflow.keras import regularizers
import tensorflow as tf
import math

class ArcFace(Layer):
    def __init__(self, n_classes=10, s=64, m=0.50, regularizer=None , use_fp16=False,**kwargs):
        super(ArcFace, self).__init__(**kwargs)
        self.n_classes = n_classes
        self.s = s
        self.m = m 
        self.regularizer = regularizers.get(regularizer)
        self.use_fp16 = use_fp16 
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.mm = self.sin_m * m   
        self.threshold = math.cos(math.pi - m)

    def get_config(self):
        config = {
                  "n_classes":self.n_classes,
                  "s":self.s,
                  "m":self.m, 
                  #  "regularizer":self.regularizer
                  "regularizer":self.regularizer
                  }
        base_config = super(ArcFace, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
    
    def build(self, input_shape):
        #  print("input_shape:",input_shape,type(input_shape))
        super(ArcFace, self).build(input_shape[0].as_list())
        
        self.W = self.add_weight(name='W',
                                shape=(input_shape[0].as_list()[-1], self.n_classes),
                                #  initializer='he_normal',
                                initializer='glorot_uniform',
                                trainable=True,
                                regularizer=self.regularizer)

    def call(self, inputs):
        x, y = inputs 
        if self.use_fp16:
            x = tf.cast(x,dtype=tf.float16)
        else:
            x = tf.cast(x,dtype=tf.float32)
#
        #  c = K.shape(x)[-1]
        c = x.get_shape().as_list()[-1]
        # normalize feature
        x = tf.nn.l2_normalize(x, axis=1)
        # normalize weights
        W = tf.nn.l2_normalize(self.W, axis=0)

        # dot product
        cos_t = x @ W  # cos(t)
        # add margin
        # clip logits to prevent zero division when backward
        #  theta = tf.acos(K.clip(logits, -1.0 + K.epsilon(), 1.0 - K.epsilon()))
        #  target_logits = tf.cos(theta + self.m)
        
        cos_t2 = tf.square(cos_t, name='cos_2')
        #  sin_t2 = tf.subtract(1., cos_t2, name='sin_2')
        sin_t2 = tf.subtract(tf.cast(1,dtype=cos_t2.dtype) , cos_t2, name='sin_2')
        sin_t = tf.sqrt(sin_t2, name='sin_t')
        cos_mt = tf.subtract(tf.multiply(cos_t, self.cos_m), tf.multiply(sin_t, self.sin_m), name='cos_mt') #cos(t+m)

        # 当 0< theta+m < pi 时 cos(theta+m)单调递减，当theta+m>pi 时 选一个单调递减函数替换，且要比-1小 
        cond_v = cos_t - self.threshold
        cond = tf.cast(tf.nn.relu(cond_v, name='if_else'), dtype=tf.bool)
        keep_val = (cos_t - self.mm)
        target_logits = tf.where(cond, cos_mt, keep_val)
        
        
        # sin = tf.sqrt(1 - logits**2)
        # cos_m = tf.cos(logits)
        # sin_m = tf.sin(logits)
        # target_logits = logits * cos_m - sin * sin_m
        #
        y = tf.one_hot(y, depth=self.n_classes, name='one_hot',dtype=cos_t.dtype)
        #  print("++++", target_logits.dtype, y.dtype  )
        logits = cos_t * tf.cast(1-y,dtype=cos_t.dtype) + target_logits * y
        # feature re-scale
        logits *= self.s
        logits  = tf.cast(logits,dtype=tf.float32)  # softmax 必须要float32
        logits = tf.nn.softmax(logits)  
        #  logits = tf.keras.layers.Softmax()(logits)
        return logits

    def compute_output_shape(self, input_shape):
        return (None, self.n_classes)


