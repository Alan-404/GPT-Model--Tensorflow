import tensorflow as tf
from tensorflow import Tensor
from keras.layers import Layer, Dense

from typing import Callable

class PositionWiseFeedForwardNetworks(Layer):
    def __init__(self, d_ff: int, d_model: int, activation: Callable[[Tensor], Tensor], trainable=True, name=None, dtype=None, dynamic=False, **kwargs):
        super().__init__(trainable, name, dtype, dynamic, **kwargs)
        self.hidden_layer = Dense(units=d_ff)
        self.activation = activation
        self.output_layer = Dense(units=d_model)

    def call(self, x: Tensor) -> Tensor:
        x = self.hidden_layer(x)
        x = self.activation(x)
        x = self.output_layer(x)

        return x
