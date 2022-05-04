#!/usr/bin/env python
# encoding: utf-8
'''
@author: YANG Zhen
@contact: zhyang8-c@my.cityu.edu.hk
@file: eval_clfs.py
@time: 2/11/2022 3:41 PM
@desc:
'''


from multiprocessing import Pool
import multiprocessing as mp
from Ours.classification import discuss_clf, get_params
from Ours.classification_features import get_featurs_labels
import pickle as pkl
import json
import tensorflow as tf
from Ours.classification import extract_data


if __name__ == "__main__":
    import sys
    if sys.argv[1].lower() == "lightgbm":
        CUR_CLF = "lightGBM"
        params = {"max_depth": [7, 9, 11, 13, 15], "n_estimators": [50, 100, 150, 200, 250], "learning_rate": [0.001, 0.005, 0.01, 0.05, 0.1]}
    elif sys.argv[1].lower() == "naive_bayes":
        CUR_CLF = "naive_bayes"
        params = {"var_smoothing": [0, 1e-9, 1e-7, 1e-5, 1e-3]}
    elif sys.argv[1].lower() == "mlp":
        CUR_CLF = "mlp"
        params = {"hidden_layer_sizes": [(32, ), (64, ), (128, ), (256, ), (512, )], "learning_rate_init": [0.001, 0.005, 0.01, 0.05, 0.1], "early_stopping": [True], "n_iter_no_change": [3], "max_iter": [50]}
    elif sys.argv[1].lower() == "lstm":
        CUR_CLF = "lstm"
        params = {"units": [32, 64, 128, 256, 512], "embed_size": [32, 64, 128, 256, 512], "lr": [0.001, 0.005, 0.01, 0.05, 0.1]}
    elif sys.argv[1].lower() == "cnn":
        CUR_CLF = "cnn"
        params = {"filters": [32, 64, 128, 256, 512], "embed_size": [32, 64, 128, 256, 512], "kernel_size": [1, 3, 5, 7, 9], "lr": [0.001, 0.005, 0.01, 0.05, 0.1]}
    else:
        assert False, "pls add other classifiers by yourself."

    if CUR_CLF == "lstm" or CUR_CLF == "cnn":
        import os
        os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
        print("GPU: ", tf.test.is_gpu_available())
        with open("../Dataset/mix_vocab.json", 'r') as f:
            entry = json.load(f)
        token_word2id = entry['token_word2id']
        action_word2id = entry['action_word2id']
        train_x, train_y = extract_data("../Dataset/train_clean_labels_1.jsonl", 100, 15, token_word2id, action_word2id)
        valid_x, valid_y = extract_data("../Dataset/valid_clean_labels_1.jsonl", 100, 15, token_word2id, action_word2id)
        params.update({"token_vocab_len": len(token_word2id), "action_vocab_len": len(action_word2id)})
        mp_lst = []
        discuss_clf(CUR_CLF, train_x, train_y, valid_x, valid_y, params, mp_lst)
        print(mp_lst)
    else:
        train_x, train_y, _ = get_featurs_labels("../Dataset/train_clean_labels_1.jsonl")
        valid_x, valid_y, _ = get_featurs_labels("../Dataset/valid_clean_labels_1.jsonl")
        params_list = get_params(params)
        print(params_list)
        # for i in params_list:
        #     discuss_clf(CUR_CLF, train_x, train_y, valid_x, valid_y, i, [])
        #     break
        # 多进程共享变量需要manager
        manager = mp.Manager
        # list不会按执行顺序存储，只会按照每个进程的结束顺序存储，需要有序的话，需要后续主动排序
        mp_lst = manager().list()
        p = Pool(processes=3)
        for id, params in enumerate(params_list):
            p.apply_async(discuss_clf, args=(CUR_CLF, train_x, train_y, valid_x, valid_y, params, mp_lst, ))

        p.close()
        p.join()
        print(mp_lst)
        print(max(mp_lst, key=lambda item: item[0][3]))
