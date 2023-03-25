import tensorflow as tf
from tensorflow import Tensor
from keras.layers import Layer, Dense
from keras.activations import softmax

class MultiHeadAttention(Layer):
    def __init__(self, heads: int, d_model: int, trainable=True, name=None, dtype=None, dynamic=False, **kwargs):
        super().__init__(trainable, name, dtype, dynamic, **kwargs)
        self.heads = heads
        self.d_model = d_model

        self.head_samples = self.d_model // self.heads

        self.dense_q = Dense(units=d_model)
        self.dense_k = Dense(units=d_model)
        self.dense_v = Dense(units=d_model)

        self.dense_output = Dense(units=d_model)

    def scaled_dot_product_attention(self, q: Tensor, k: Tensor, v: Tensor, mask: Tensor):
        dk = tf.cast(k.shape[-1], dtype=tf.float32)

        attention_scores = tf.matmul(q, k, transpose_b=True)
        attention_scores = attention_scores/(tf.sqrt(dk))

        if mask is not None:
            attention_scores += mask*(-1e15)

        attention_weights = softmax(attention_scores, axis=-1)
        output = tf.matmul(attention_weights, v)

        return output
    
    def split(self, x: Tensor):
        batch_size = x.shape[0]
        length = x.shape[1]

        x = tf.reshape(x, (batch_size, length, self.heads, self.head_samples))
        x = tf.transpose(x, [0, 2, 1, 3])

        return x
    
    def call(self, q: Tensor, k: Tensor, v: Tensor, mask: Tensor):
        batch_size = q.shape[0]
        length = q.shape[1]

        qw = self.dense_q(q)
        kw = self.dense_k(k)
        vw = self.dense_v(v)

        q_heads = self.split(qw)
        k_heads = self.split(kw)
        v_heads = self.split(vw)

        attention_output = self.scaled_dot_product_attention(q_heads, k_heads, v_heads, mask)

        attention_output = tf.transpose(attention_output, [0, 2, 1, 3])
        attention_output = tf.reshape(attention_output, (batch_size, length, self.d_model))

        attention_output = self.dense_output(attention_output)

        return attention_output