from tensorflow import Tensor
from keras.layers import Layer, Dropout, LayerNormalization


class ResidualConnection(Layer):
    def __init__(self, dropout_rate: float, eps: float, trainable=True, name=None, dtype=None, dynamic=False, **kwargs):
        super().__init__(trainable, name, dtype, dynamic, **kwargs)
        self.dropout_layer = Dropout(rate=dropout_rate)
        self.layer_norm = LayerNormalization(epsilon=eps)

    def call(self, x: Tensor, pre: Tensor, training: bool) -> Tensor:
        x = self.dropout_layer(x, training)
        x = x + pre
        x = self.layer_norm(x)

        return x