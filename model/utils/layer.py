import tensorflow as tf
from tensorflow import Tensor
from keras.layers import Layer

from .attention import MultiHeadAttention
from .ffn import PositionWiseFeedForwardNetworks
from .residual import ResidualConnection

from typing import Callable


class DecoderLayer(Layer):
    def __init__(self, d_model: int, heads: int, d_ff: int, dropout_rate: float, eps: float, activation: Callable[[Tensor], Tensor],  trainable=True, name=None, dtype=None, dynamic=False, **kwargs):
        super().__init__(trainable, name, dtype, dynamic, **kwargs)
        self.masked_multi_head_attention = MultiHeadAttention(heads=heads, d_model=d_model)
        self.ffn = PositionWiseFeedForwardNetworks(d_ff=d_ff, d_model=d_model, activation=activation)

        self.residual_connection_1 = ResidualConnection(dropout_rate=dropout_rate, eps=eps)
        self.residual_connection_2 = ResidualConnection(dropout_rate=dropout_rate, eps=eps)

    def call(self, x: Tensor, mask: Tensor, training: bool):
        # sub layer 1
        q = k = v = x
        attention_output = self.masked_multi_head_attention(q, k, v, mask)
        sub_layer_1 = self.residual_connection_1(attention_output, x, training)

        # sub layer 2
        ffn_output = self.ffn(sub_layer_1)
        sub_layer_2 = self.residual_connection_2(ffn_output)

        return sub_layer_2
