import tensorflow as tf
from tensorflow import Tensor
from keras.layers import Layer

class PositionalEncoding(Layer):
    def __init__(self, trainable=True, name=None, dtype=None, dynamic=False, **kwargs):
        super().__init__(trainable, name, dtype, dynamic, **kwargs)

    def __encode_length(self, length: int) -> Tensor:
        pos = tf.range(length)
        pos = tf.expand_dims(pos, axis=-1)

        return pos
    
    def __encode_embedding(self, embedding_dim: int) -> Tensor:
        angles = tf.range(embedding_dim)

        angles[1::2] = angles[0::2]
        angles = 1/(tf.pow(10000, angles/embedding_dim))

        angles = tf.expand_dims(angles, axis=0)

        return angles
    
    def call(self, x: Tensor) -> Tensor:
        pos = self.__encode_length(x.shape[1])
        angles = self.__encode_embedding(x.shape[-1])

        pos_angles = tf.matmul(pos, angles)

        pos_angles[0::2] = tf.sin(pos_angles[0::2])
        pos_angles[1::2] = tf.cos(pos_angles[1::2])

        pos_angles = tf.expand_dims(pos_angles, axis=0)

        return pos_angles
