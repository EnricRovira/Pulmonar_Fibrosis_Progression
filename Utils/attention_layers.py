
import tensorflow as tf
from tensorflow.keras import layers, constraints

# Additive
class BahdanauAttention(layers.Layer):
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.W1 = layers.Dense(units)
        self.W2 = layers.Dense(units)
        self.V = layers.Dense(1)

    def call(self, features, hidden):
        # features(CNN_encoder output) shape == (batch_size, 64, embedding_dim)

        # hidden shape == (batch_size, hidden_size)
        # hidden_with_time_axis shape == (batch_size, 1, hidden_size)
        hidden_with_time_axis = tf.expand_dims(hidden, 1)

        # score shape == (batch_size, 64, hidden_size)
        score = tf.nn.tanh(self.W1(features) + self.W2(hidden_with_time_axis))

        # attention_weights shape == (batch_size, 64, 1)
        # you get 1 at the last axis because you are applying score to self.V
        attention_weights = tf.nn.softmax(self.V(score), axis=1)

        # context_vector shape after sum == (batch_size, hidden_size)
        context_vector = attention_weights * features
        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector, attention_weights


# Baswani
class ScaledDotProductAttention(layers.Layer):
    def __init__(self, units):
        super(ScaledDotProductAttention, self).__init__()
        self.units = units
        self.square_root = tf.math.sqrt(tf.cast(self.units,tf.float32))

    def call(self, source, target, training=False):
        # target is also referred as query, and source as values.

        # target must be a hidden vector of shape=(None, hidden_size), 
        # while source has shape=(None, steps, hidden_size)

        # expand the dimension of the target, (None, 1, hidden_size)
        target_with_time_axis = tf.expand_dims(target, axis=1)

        # score shape == (batch_size, max_length, 1)
        # we get 1 at the last axis because we are performing the dot product along that dim
        score = tf.matmul(a=target_with_time_axis, b=source, transpose_b=True)
        score = tf.divide(score, self.square_root)
        score = tf.transpose(score, perm=[0, 2, 1])

        # attention_weights shape == (batch_size, max_length, 1)
        attention_weights = tf.nn.softmax(score, axis=1)

        # context_vector shape after sum == (batch_size, hidden_size)
        context_vector = attention_weights * source
        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector, attention_weights

# Luong
class GeneralAttention(layers.Layer):
    def __init__(self, units, kernel_mn=2):
        super(GeneralAttention, self).__init__()
        self.units = units
        self.kernel_mn = kernel_mn
        self.score_weight = layers.Dense(units, kernel_constraint=constraints.max_norm(kernel_mn))  

    def call(self, inputs, training=False):
        # target is also referred as query, and source as values.
        target, source = inputs
        # target must be a hidden vector of shape=(None, hidden_size), 
        # while source has shape=(None, steps, hidden_size)

        # expand the dimension of the target, (None, 1, hidden_size)
        target_with_time_axis = tf.expand_dims(target, axis=1)

        # score shape == (batch_size, max_length, 1)
        # we get 1 at the last axis because we are performing the dot product along that dim
        score = tf.matmul(a=target_with_time_axis, b=self.score_weight(source), transpose_b=True)
        score = tf.transpose(score, perm=[0, 2, 1])

        # attention_weights shape == (batch_size, max_length, 1)
        attention_weights = tf.nn.softmax(score, axis=1)

        # context_vector shape after sum == (batch_size, hidden_size)
        context_vector = attention_weights * source
        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector, attention_weights
