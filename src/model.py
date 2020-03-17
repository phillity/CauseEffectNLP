import os
import h5py
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Bidirectional, LSTM, Dense, Flatten, Activation, RepeatVector, Permute, Multiply, Lambda
from argparse import ArgumentParser


np.random.seed(42)
tf.random.set_seed(42)


class Attention:
    def __call__(self, inp, combine=True, return_attention=True):
        repeat_size = int(inp.shape[-1])

        x_a = Dense(repeat_size, kernel_initializer="glorot_uniform",
                    activation="tanh", name="tanh_mlp")(inp)

        x_a = Dense(1, kernel_initializer="glorot_uniform",
                    activation="linear", name="word-level_context")(x_a)
        x_a = Flatten()(x_a)
        att_out = Activation("softmax")(x_a)

        x_a2 = RepeatVector(repeat_size)(att_out)
        x_a2 = Permute([2, 1])(x_a2)
        out = Multiply()([inp, x_a2])

        if combine:
            out = Lambda(lambda x: tf.keras.backend.sum(x, axis=1),
                         name="expectation_over_words")(out)

        if return_attention:
            out = (out, att_out)

        return out


def LSTMModel(maxlen=10, edgelen=1100):
    net_input = Input(shape=(maxlen, edgelen))
    lstm = Bidirectional(LSTM(256, return_sequences=True))(net_input)
    x, attention = Attention()(lstm)
    dense = Dense(1, activation="sigmoid")(x)
    model = Model(inputs=net_input, outputs=dense)
    track = Model(inputs=net_input, outputs=attention)

    return model, track
