import tensorflow as tf
from tensorflow import Tensor
from keras.layers import Embedding, Dense
from model.utils.position import PositionalEncoding
from keras.models import Model 
from model.utils.mask import generate_look_ahead_mask
from model.components.decoder import Decoder

from keras.optimizers import Adam
from keras.losses import SparseCategoricalCrossentropy

from keras.metrics import Mean

import os

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

        self.checkpoint = tf.train.Checkpoint(model=self.model, optimizer = self.optimizer)
        self.checkpoint_manager = tf.train.CheckpointManager(self.checkpoint, checkpoint, max_to_keep=3)

        self.train_loss = Mean(name='train_loss')

    def loss_function(self, outputs: Tensor, labels: Tensor):
        mask = tf.math.logical_not(labels == 0)

        loss = self.cross_entropy(labels, outputs)
        
        mask = tf.cast(mask, dtype=loss.dtype)
        
        loss = loss*mask
        
        return tf.math.reduce_sum(loss) / tf.math.reduce_sum(mask)

    def train_step(self, inputs: Tensor, labels: Tensor):
        with tf.GradientTape() as tape:
            outputs = self.model(inputs, True)
            loss = self.loss_function(outputs, labels)

        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
    
        self.train_loss.update_state(loss)

    """ def fit(self, train_data: Tensor, epochs: int = 1, batch_size: int = 1, mini_batch: int = 1):
         """