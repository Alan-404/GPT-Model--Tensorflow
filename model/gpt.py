import tensorflow as tf
from tensorflow import Tensor
from keras.layers import Embedding, Dense
from model.utils.position import PositionalEncoding
from keras.models import Model 
from model.utils.mask import generate_look_ahead_mask
from model.components.decoder import Decoder

from keras.optimizers import Adam
from keras.losses import SparseCategoricalCrossentropy

from typing import Callable

class GPTModel(Model):
    def __init__(self, token_size: int, n: int, d_model: int, heads: int, d_ff: int, dropout_rate: float, eps: float, activation:Callable[[Tensor], Tensor],  *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.embedding_layer = Embedding(input_dim=token_size, output_dim=d_model)
        self.positional_encoding = PositionalEncoding()
        self.decoder = Decoder(n=n, d_model=d_model, heads=heads, d_ff=d_ff, dropout_rate=dropout_rate, eps=eps, activation=activation)
        self.dense = Dense(units=token_size)

    def call(self, x: Tensor, training: bool):
        mask = generate_look_ahead_mask(x)
        x = self.embedding_layer(x)
        x = self.positional_encoding(x)
        x = self.decoder(x, mask, training)
        x = self.dense(x)

        return x
    

class GPT:
    def __init__(self,
                token_size: int,
                n: int,
                d_model: int,
                heads: int,
                d_ff: int,
                dropout_rate: float,
                eps: float,
                activation: Callable[[Tensor], Tensor],
                checkpoint: str = None) -> None:
        self.model = GPTModel(
            token_size=token_size,
            n=n,
            d_model=d_model,
            d_ff=d_ff,
            heads=heads,
            dropout_rate=dropout_rate,
            eps=eps,
            activation=activation
        )

        self.cross_entropy = SparseCategoricalCrossentropy(from_logits=True, reduction='none')
        self.optimizer = Adam()
    
    