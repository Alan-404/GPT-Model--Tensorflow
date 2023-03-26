import tensorflow as tf
from tensorflow import Tensor
import numpy as np

def generate_padding_mask(x: Tensor):
    return tf.cast(x == 0, dtype=tf.float32)[:, np.newaxis, np.newaxis, :]

def generate_trig(length: int):
    mask = 1 - tf.linalg.band_part(tf.ones((length, length)), -1, 0)
    return mask

def generate_look_ahead_mask(x: Tensor):
    padding_mask = generate_padding_mask(x)
    trig = generate_trig(x.shape[1])

    look_ahead_mask = tf.math.maximum(trig, padding_mask)
    return look_ahead_mask