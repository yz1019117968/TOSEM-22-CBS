#!/usr/bin/env python
# encoding: utf-8
'''
@author: YANG Zhen
@contact: zhyang8-c@my.cityu.edu.hk
@file: cnn.py
@time: 2/18/2022 2:48 PM
@desc:
'''

from tensorflow.keras import optimizers, Input, Model, losses, metrics
from tensorflow.keras.layers import Embedding, Conv1D, Dense, Flatten
import tensorflow as tf

class cnn:
    def __init__(self, token_vocab_len, action_vocab_len, embed_size, filters, kernel_size, lr):
        self.token_embedding = Embedding(token_vocab_len, embed_size)
        self.action_embedding = Embedding(action_vocab_len, embed_size)
        self.cnn_code = Conv1D(filters=filters, kernel_size=kernel_size, activation="relu")
        self.cnn_comm = Conv1D(filters=filters, kernel_size=kernel_size, activation="relu")
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
        # (None, 100, 384)
        code_embed = tf.concat([old_code_embed, new_code_embed, actions_embed], axis=-1)
        code_enc = self.cnn_code(code_embed)
        old_comms_enc = self.cnn_comm(old_comms_embed)
        common_sem = self.flatten(tf.concat([code_enc, old_comms_enc], axis=1))
        final_out = self.dense_out(common_sem)

        model = Model(inputs=[old_code, new_code, actions, old_comms], outputs=[final_out])
        model.compile(optimizer=optimizers.Adam(learning_rate=self.lr),
                    loss=losses.binary_crossentropy,
                    metrics=[metrics.binary_accuracy])

        return model
