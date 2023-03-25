import tensorflow as tf
from tensorflow import Tensor
from keras.layers import Embedding
from keras.models import Model 
from model.components.decoder import Decoder

class GPTModel(Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
