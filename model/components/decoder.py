import tensorflow as tf
from tensorflow import Tensor
from keras.layers import Layer

from model.utils.layer import DecoderLayer

from typing import Callable

class Decoder(Layer):
    def __init__(self,n: int, d_model: int, heads: int, d_ff: int, dropout_rate: float, eps: float, activation:Callable[[Tensor], Tensor],  trainable=True, name=None, dtype=None, dynamic=False, **kwargs):
        super().__init__(trainable, name, dtype, dynamic, **kwargs)
        self.layers = [DecoderLayer(d_model=d_model, heads=heads, d_ff=d_ff, dropout_rate=dropout_rate, eps=eps, activation=activation) for _ in range(n)]

    def call(self, x: Tensor, mask: Tensor, training: bool):
        for layer in self.layers:
            x = layer(x, mask, training)

        return x
