import tensorflow as tf
import numpy as np

def init_weight(name, shape, stddev=1.0):
    return tf.Variable(tf.truncated_normal(shape, stddev=stddev/np.sqrt(shape[0])), name=name)

def init_bias(name, shape):
    return tf.Variable(tf.zeros(shape), name=name)