import keras.backend as K
from keras.engine.topology import Layer
from keras import initializers
from keras import regularizers
from keras import constraints
import numpy as np
import tensorflow as tf


class Attention(Layer):
    def __init__(self, W_regularizer=None, b_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 bias=True, **kwargs):
        """
        Keras Layer that implements an Content Attention mechanism.
        Supports Masking.
        """
        self.supports_masking = True
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        assert type(input_shape) == list

        self.steps = input_shape[0][1]

        self.W = self.add_weight(shape=(input_shape[0][-1], input_shape[1][-1]),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        if self.bias:
            self.b = self.add_weight(shape=(1,),
                                     initializer='zeros',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)
        else:
            self.b = None

        self.built = True

    def compute_mask(self, input_tensor, mask=None):
        assert type(input_tensor) == list
        assert type(mask) == list
        return None

    def call(self, input_tensor, mask=None):
        x = input_tensor[0]
        y = input_tensor[1]
        mask = mask[0]

        y = K.transpose(K.dot(self.W, K.transpose(y)))
        y = K.expand_dims(y, axis=-2)
        y = K.repeat_elements(y, self.steps, axis=1)
        eij = K.sum(x * y, axis=-1)

        if self.bias:
            b = K.repeat_elements(self.b, self.steps, axis=0)
            eij += b

        eij = K.tanh(eij)

        a = K.exp(eij)

        if mask is not None:
            a *= K.cast(mask, K.floatx())

        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

        return a

    def get_output_shape_for(self, input_shape):
        return (input_shape[0][0], input_shape[0][1])


class WeightedSum(Layer):
    def __init__(self, **kwargs):
        self.supports_masking = True
        super(WeightedSum, self).__init__(**kwargs)

    def call(self, input_tensor, mask=None):
        assert type(input_tensor) == list
        assert type(mask) == list

        x = input_tensor[0]
        a = input_tensor[1]

        a = K.expand_dims(a)
        weighted_input = x * a

        return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], input_shape[0][-1])

    def compute_mask(self, x, mask=None):
        return None


class WeightedAspectEmb(Layer):
    def __init__(self, input_dim, output_dim,
                 init='uniform', input_length=None,
                 W_regularizer=None, activity_regularizer=None,
                 W_constraint=None,
                 weights=None, dropout=0., **kwargs):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.init = initializers.get(init)
        self.input_length = input_length
        self.dropout = dropout

        self.W_constraint = constraints.get(W_constraint)
        self.W_regularizer = regularizers.get(W_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)

        if 0. < self.dropout < 1.:
            self.uses_learning_phase = True
        self.initial_weights = weights
        kwargs['input_shape'] = (self.input_length,)
        kwargs['input_dtype'] = K.floatx()
        super(WeightedAspectEmb, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight((self.input_dim, self.output_dim),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
        self.built = True

    def compute_mask(self, x, mask=None):
        return None

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], self.output_dim)

    def call(self, x, mask=None):
        return K.dot(x, self.W)


class Average(Layer):
    def __init__(self, mask_zero=True, **kwargs):
        self.mask_zero = mask_zero
        self.supports_masking = True
        super(Average, self).__init__(**kwargs)

    def call(self, x, mask=None):
        if self.mask_zero:
            mask = K.cast(mask, K.floatx())
            mask = K.expand_dims(mask)
            x = x * mask
            return K.sum(x, axis=1) / K.sum(mask, axis=1) #Originally axis=-2
        else:
            return K.mean(x, axis=1)

    def get_output_shape_for(self, input_shape):
        return input_shape[0] + input_shape[-1]

    def compute_mask(self, x, mask=None):
        return None


class MaxMargin(Layer):
    def __init__(self, **kwargs):
        super(MaxMargin, self).__init__(**kwargs)

    def call(self, input_tensor, mask=None):
        z_s = input_tensor[0]
        z_n1 = input_tensor[1]
        z_n2 = input_tensor[2]
        z_n3 = input_tensor[3]
        z_n4 = input_tensor[4]
        z_n5 = input_tensor[5]
        r_s = input_tensor[6]

        z_s = z_s / K.cast(K.epsilon() + K.sqrt(K.sum(K.square(z_s), axis=-1, keepdims=True)), K.floatx())
        z_n1 = z_n1 / K.cast(K.epsilon() + K.sqrt(K.sum(K.square(z_n1), axis=-1, keepdims=True)), K.floatx())
        z_n2 = z_n2 / K.cast(K.epsilon() + K.sqrt(K.sum(K.square(z_n2), axis=-1, keepdims=True)), K.floatx())
        z_n3 = z_n3 / K.cast(K.epsilon() + K.sqrt(K.sum(K.square(z_n3), axis=-1, keepdims=True)), K.floatx())
        z_n4 = z_n4 / K.cast(K.epsilon() + K.sqrt(K.sum(K.square(z_n4), axis=-1, keepdims=True)), K.floatx())
        z_n5 = z_n5 / K.cast(K.epsilon() + K.sqrt(K.sum(K.square(z_n5), axis=-1, keepdims=True)), K.floatx())
        r_s = r_s / K.cast(K.epsilon() + K.sqrt(K.sum(K.square(r_s), axis=-1, keepdims=True)), K.floatx())

        pos = K.sum(z_s * r_s, axis=-1, keepdims=True)
        neg1 = K.sum(z_n1 * r_s, axis=-1)
        neg2 = K.sum(z_n2 * r_s, axis=-1)
        neg3 = K.sum(z_n3 * r_s, axis=-1)
        neg4 = K.sum(z_n4 * r_s, axis=-1)
        neg5 = K.sum(z_n5 * r_s, axis=-1)

        loss = K.cast(tf.maximum(0., (1. - (5*pos) + neg1 + neg2 + neg3 + neg4 + neg5)), K.floatx())

        #loss = K.cast(tf.maximum(0., (1. - pos + neg1)) + tf.maximum(0., (1. - pos + neg2)) + tf.maximum(0., (1. - pos + neg3)) +
        #              tf.maximum(0., (1. - pos + neg4)) + tf.maximum(0., (1. - pos + neg5)), K.floatx())
        return loss

    def compute_mask(self, input_tensor, mask=None):
        return None

    def get_output_shape_for(self, input_shape):
        return (input_shape[0][0], 1)
