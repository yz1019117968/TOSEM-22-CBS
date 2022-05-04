#!/usr/bin/env python
# encoding: utf-8
'''
@author: YANG Zhen
@contact: zhyang8-c@my.cityu.edu.hk
@file: lstm.py
@time: 2/16/2022 11:24 AM
@desc:
'''

from tensorflow.keras import optimizers, Input, Model, losses, metrics
from tensorflow.keras.layers import Embedding, LSTM, Bidirectional, Dense, Flatten
import tensorflow as tf

class lstm:
    def __init__(self, token_vocab_len, action_vocab_len, embed_size, units, lr):
        self.token_embedding = Embedding(token_vocab_len, embed_size)
        self.action_embedding = Embedding(action_vocab_len, embed_size)
        self.bi_lstm_code = Bidirectional(LSTM(units, return_sequences=True))
        self.bi_lstm_comms = Bidirectional(LSTM(units, return_sequences=True))
        self.dense_code = Dense(units * 2)
        self.dense_out = Dense(1, activation="sigmoid")
        self.lr = lr
        self.flatten = Flatten()

    def build_model(self):
        old_code = Input(shape=(100,))
        new_code = Input(shape=(100,))
        actions = Input(shape=(100,))
        old_comms = Input(shape=(15,))

        old_code_embed = self.token_embedding(old_code)
        new_code_embed = self.token_embedding(new_code)
        actions_embed = self.action_embedding(actions)
        old_comms_embed = self.token_embedding(old_comms)

        code_embed = tf.concat([old_code_embed, new_code_embed, actions_embed], axis=-1)
        code_enc = self.bi_lstm_code(code_embed)
        code_embed_1 = self.dense_code(code_enc)
        old_comms_enc = self.bi_lstm_comms(old_comms_embed)
        common_sem = self.flatten(tf.concat([code_embed_1, old_comms_enc], axis=1))
        final_out = self.dense_out(common_sem)

        model = Model(inputs=[old_code, new_code, actions, old_comms], outputs=[final_out])
        model.compile(optimizer=optimizers.Adam(learning_rate=self.lr),
                    loss=losses.binary_crossentropy,
                    metrics=[metrics.binary_accuracy])

        return model




