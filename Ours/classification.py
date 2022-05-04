#-*- coding:utf-8 -*-
import copy
import os.path

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
import lightgbm as lgb
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from tensorflow.keras.callbacks import EarlyStopping
from Ours.lstm import lstm
from Ours.cnn import cnn
from collections import Counter
from Ours.classification_features import get_featurs_labels
import numpy as np
import pickle as pkl
import jsonlines
import random
from itertools import product
import json
import tensorflow as tf
import time

def get_code_tokens(code_change_seqs):
    old_code_tokens = []
    new_code_tokens = []
    actions = []
    for seq in code_change_seqs:
        old_code_tokens.append(seq[0])
        new_code_tokens.append(seq[1])
        actions.append(seq[2])
    return old_code_tokens, new_code_tokens, actions

def tailor_lens(seq, max_len):
    if len(seq) < max_len:
        seq += ['<pad>'] * (max_len - len(seq))
    else:
        seq = seq[: max_len]
    return seq

def word_to_ids(sents, word2id):
    _sents = []
    for token in sents:
        if token in word2id.keys():
            _sents.append(word2id[token])
        else:
            _sents.append(word2id["<unk>"])
    return _sents

def extract_data(path, code_max_len, nl_max_len, tokens_word2id, actions_word2id):
    ys = []
    xs = []
    with open(path, "r", encoding="utf-8") as fr:
        for line in fr.readlines():
            line = json.loads(line)
            ys.append(int(line['HebCUP_correct_labeled']))
            old_code_tokens, new_code_tokens, actions = get_code_tokens(line['code_change_seq'])
            old_code_tokens = tailor_lens(old_code_tokens, code_max_len)
            old_code_ids = word_to_ids(old_code_tokens, tokens_word2id)
            new_code_tokens = tailor_lens(new_code_tokens, code_max_len)
            new_code_ids = word_to_ids(new_code_tokens, tokens_word2id)
            actions = tailor_lens(actions, code_max_len)
            actions_ids = word_to_ids(actions, actions_word2id)
            old_comm_tokens = tailor_lens(line['src_desc_tokens'], nl_max_len)
            old_comm_ids = word_to_ids(old_comm_tokens, tokens_word2id)

            xs.append((old_code_ids, new_code_ids, actions_ids, old_comm_ids))
    return xs, ys

def input_data(dataset):
    old_code_ids = []
    new_code_ids = []
    action_ids = []
    old_comm_ids = []
    for item in dataset:
        old_code_ids.append(item[0])
        new_code_ids.append(item[1])
        action_ids.append(item[2])
        old_comm_ids.append(item[3])
    return [np.array(old_code_ids).astype('float64'), np.array(new_code_ids).astype('float64'),
            np.array(action_ids).astype('float64'), np.array(old_comm_ids).astype('float64')]

def get_params(params_dict):
    loop_val = []
    keys = params_dict.keys()
    for key in params_dict.keys():
        loop_val.append(params_dict[key])
    combs = []
    for comb in product(*loop_val):
        comb_dict = {}
        for item, key in zip(comb, keys):
            comb_dict[key] = item
        combs.append(comb_dict)
    return combs

def clf_exps(clf_name, train_x, train_y, valid_x, valid_y, params):

    if clf_name == "tree":
        clf = tree.DecisionTreeClassifier(**params)
    elif clf_name == "random_forest":
        clf = RandomForestClassifier(**params)
    elif clf_name == "lightGBM":
        clf = lgb.LGBMClassifier(**params)
    elif clf_name == "naive_bayes":
        clf = GaussianNB(**params)
    elif clf_name == "mlp":
        clf = MLPClassifier(**params)
    elif clf_name == "lstm":
        _clf = lstm(params['token_vocab_len'], params['action_vocab_len'], int(params['embed_size']), int(params['units']), float(params['lr']))
        clf = _clf.build_model()
    elif clf_name == "cnn":
        _clf = cnn(params['token_vocab_len'], params['action_vocab_len'], int(params['embed_size']), int(params['filters']), int(params['kernel_size']), float(params['lr']))
        clf = _clf.build_model()
    else:
        assert False, "the clf_name is not included!"
    if clf_name == "lstm" or clf_name == "cnn":
        cb_early_stop = EarlyStopping(monitor="val_loss", patience=3, mode="min", min_delta=0.0001)
        clf.fit(input_data(train_x), np.array(train_y).astype('float64'), epochs=50, validation_split=0.1, callbacks=[cb_early_stop], batch_size=128, verbose=1)
        preds_valid = clf.predict(input_data(valid_x)).squeeze()
        preds_valid = [1 if i >= 0.5 else 0 for i in preds_valid]
    else:
        clf.fit(train_x, train_y)
        start_pred = time.time()
        preds_valid = clf.predict(valid_x)
        end_pred = time.time()
        print(f"pred once time cost: {end_pred - start_pred}")
    print("preds_valid: ", preds_valid)


    return preds_valid

def msel(clf_name, train_x, train_y, valid_x, valid_y, params):
    train_stat = Counter(train_y)
    num_1 = train_stat.get(1)
    indexes_0 = []
    indexes_1 = []
    for idx, i in enumerate(train_y):
        if i == 0:
            indexes_0.append(idx)
        else:
            indexes_1.append(idx)
    preds = np.zeros((len(valid_y)))
    init_indexes_0 = copy.deepcopy(indexes_0)
    loops = int(len(indexes_0)/len(indexes_1) + 0.5)
    if loops % 2 == 0:
        print("loops % 2 == 0 !")
        loops += 1
    for loop_i in range(1,  loops+1):
        if loop_i == loops:
            if loops % 2 != 0:
                indexes_0_sample = indexes_0
            else:
                indexes_0_sample = random.sample(init_indexes_0, num_1)
        else:
            indexes_0_sample = random.sample(indexes_0, num_1)
            indexes_0 = list(set(indexes_0).difference(set(indexes_0_sample)))
        train_x1 = train_x[indexes_1+indexes_0_sample]
        train_y1 = train_y[indexes_1+indexes_0_sample]
        preds += clf_exps(clf_name, train_x1,train_y1,valid_x,valid_y, params)
    preds_valid = [1 if i > loops/2 else 0 for i in preds]
    print("for each fold: ", Counter(valid_y))
    print("MSEL done once !")
    return preds_valid


def k_fold_exps(clf_name, train_x, train_y, valid_x, valid_y, params):
    if clf_name == "lstm" or clf_name == "cnn":
        train_xl = np.concatenate([train_x, valid_x], axis=0)
    else:
        train_xl = np.vstack((train_x, valid_x))
    train_yl = np.hstack((train_y, valid_y))
    kf = KFold(n_splits=10, shuffle=True)
    preds_valid_lst = []
    valid_y_lst = []
    print("train_x: ", len(train_xl))
    for iter, (train_index, valid_index) in enumerate(kf.split(train_xl)):
        train_x = train_xl[train_index]
        train_y = train_yl[train_index]
        valid_x = train_xl[valid_index]
        valid_y = train_yl[valid_index]

        if clf_name == "lstm" or clf_name == "cnn":
            with open(f"../Dataset/{clf_name}/fold_{iter}.pkl", "wb") as fw:
                pkl.dump({"train_x":train_x, "train_y": train_y, "valid_x": valid_x, "valid_y": valid_y}, fw)
            if iter == 9:
                return
            continue

        preds_valid = msel(clf_name, train_x,train_y,valid_x,valid_y, params)

        preds_valid_lst += preds_valid
        valid_y_lst += valid_y.tolist()
    # y_true, y_pred
    acc = accuracy_score(valid_y_lst, preds_valid_lst)
    pre = precision_score(valid_y_lst, preds_valid_lst)
    recall = recall_score(valid_y_lst, preds_valid_lst)
    f1 = f1_score(valid_y_lst, preds_valid_lst)
    print(f"valid acc: {acc}")
    print(f"valid pre: {pre}")
    print(f"valid recall: {recall}")
    print(f"valid f1: {f1}")

    return [acc, pre, recall, f1]

def k_fold_dl(clf_name, params):
    import gc
    preds_valid_lst = []
    valid_y_lst = []
    for iter in range(10):
        with open(f"../Dataset/{clf_name}/fold_{iter}.pkl", "rb") as fr:
            dataset = pkl.load(fr)
        preds_valid = msel(clf_name, dataset['train_x'], dataset['train_y'], dataset['valid_x'], dataset['valid_y'], params)
        preds_valid_lst += preds_valid
        valid_y_lst += dataset['valid_y'].tolist()
        del dataset
        gc.collect()
        print(f"{iter}th finished!")
    acc = accuracy_score(valid_y_lst, preds_valid_lst)
    pre = precision_score(valid_y_lst, preds_valid_lst)
    recall = recall_score(valid_y_lst, preds_valid_lst)
    f1 = f1_score(valid_y_lst, preds_valid_lst)
    print(f"valid acc: {acc}")
    print(f"valid pre: {pre}")
    print(f"valid recall: {recall}")
    print(f"valid f1: {f1}")

def discuss_clf(clf_name, train_x, train_y, valid_x, valid_y, params, mp_lst):
    if clf_name == "lstm" or clf_name == "cnn":
        if not os.path.exists(f'../Dataset/{clf_name}/fold_0.pkl'):
            os.mkdir(f'../Dataset/{clf_name}/')
            k_fold_exps(clf_name, train_x, train_y, valid_x, valid_y, params)
        k_fold_dl(clf_name, params)
    else:
        mp_lst.append([k_fold_exps(clf_name, train_x, train_y, valid_x, valid_y, params), params])

def final_exps(clf_name, params, train_data_path, valid_data_path, test_data_path, out_suffix, use_msel=True):
    print("extracting feature ...")
    if clf_name == 'lstm' or clf_name == "cnn":
        os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
        print("GPU: ", tf.test.is_gpu_available())
        with open("../Dataset/mix_vocab.json", 'r') as f:
            entry = json.load(f)
        token_word2id = entry['token_word2id']
        action_word2id = entry['action_word2id']
        train_x, train_y = extract_data(train_data_path, 100, 15, token_word2id, action_word2id)
        valid_x, valid_y = extract_data(valid_data_path, 100, 15, token_word2id, action_word2id)
        test_x, test_y = extract_data(test_data_path, 100, 15, token_word2id, action_word2id)
        with open(test_data_path, "r", encoding="utf-8") as fr:
            test_items = []
            for line in fr.readlines():
                test_items.append(json.loads(line))
        params.update({"token_vocab_len": len(token_word2id), "action_vocab_len": len(action_word2id)})
    else:
        train_x, train_y, _ = get_featurs_labels(train_data_path)
        valid_x, valid_y, _ = get_featurs_labels(valid_data_path)
        start = time.time()
        test_x, test_y, test_items = get_featurs_labels(test_data_path)
        end = time.time()
        print(f"data loading time cost: {end - start}s")
    print("feature extraction completed!")
    train_xl = np.vstack((train_x, valid_x))
    train_yl = np.hstack((train_y, valid_y))
    if use_msel:
        preds_test = msel(clf_name, train_xl, train_yl, test_x, test_y, params)
    else:
        preds_test = clf_exps(clf_name, train_xl, train_yl,test_x, test_y, params).tolist()
    acc = accuracy_score(test_y, preds_test)
    pre = precision_score(test_y, preds_test)
    recall = recall_score(test_y, preds_test)
    f1 = f1_score(test_y, preds_test)
    print(f"valid acc: {acc}")
    print(f"valid pre: {pre}")
    print(f"valid recall: {recall}")
    print(f"valid f1: {f1}")
    print(Counter(preds_test))

    HCUP_items = []
    CUP_items = []
    for y, item in zip(preds_test, test_items):
        if y == 1:
            HCUP_items.append(item)
        else:
            CUP_items.append(item)

    if not os.path.exists(f"../Dataset/splitted_test_sets/{clf_name}/"):
        os.mkdir(f"../Dataset/splitted_test_sets/{clf_name}/")
    with jsonlines.open(f"../Dataset/splitted_test_sets/{clf_name}/HCUP_test{out_suffix}.jsonl", "w") as writer:
        writer.write_all(HCUP_items)
    with jsonlines.open(f"../Dataset/splitted_test_sets/{clf_name}/CUP_test{out_suffix}.jsonl", "w") as writer:
        writer.write_all(CUP_items)
    with jsonlines.open(f"../Dataset/splitted_test_sets/{clf_name}/preds_test{out_suffix}.jsonl", "w") as writer:
        writer.write_all(preds_test)


def random_classify(data_path, seed):
    import random
    import json
    hebcup = []
    cup = []
    with open(data_path, "r", encoding="utf-8") as f:
        data = f.readlines()
    half_0 = int(len(data) / 2)
    half_1 = len(data) - half_0
    random_list = [0] * half_0 +[1] * half_1
    random.seed(seed)
    random.shuffle(random_list)
    correct_list = []
    for i, item in zip(random_list, data):
        item = json.loads(item)
        correct_list.append(item['HebCUP_correct_labeled'])
        if i == 1:
            hebcup.append(item)
        elif i == 0:
            cup.append(item)
        else:
            raise Exception("sth wrong!")

    acc = accuracy_score(correct_list, random_list)
    pre = precision_score(correct_list, random_list)
    recall = recall_score(correct_list, random_list)
    f1 = f1_score(correct_list, random_list)
    print(acc, pre, recall, f1)
    with jsonlines.open(f"../Dataset/splitted_test_sets/lightGBM/HCUP_test_random.jsonl", "w") as writer:
        writer.write_all(hebcup)
    with jsonlines.open(f"../Dataset/splitted_test_sets/lightGBM/CUP_test_random.jsonl", "w") as writer:
        writer.write_all(cup)

if __name__ == "__main__":
    # final_exps("lightGBM", {'max_depth': 11, 'min_child_samples': 20, 'n_estimators': 250, 'learning_rate': 0.1},
    #            "../Dataset/train_clean_labels_1.jsonl", "../Dataset/valid_clean_labels_1.jsonl",
    #            "../Dataset/test_clean_labels_1.jsonl", out_suffix="20220304")
    # random_classify("../Dataset/test_clean_labels.jsonl", 0)
    import sys
    clf_name = sys.argv[1]
    out_suffix = sys.argv[2]
    params = {'max_depth': 11, 'min_child_samples': 20, 'n_estimators': 250, 'learning_rate': 0.1}
    final_exps(clf_name, params, f"../Dataset/train_clean_labels_1.jsonl", f"../Dataset/valid_clean_labels_1.jsonl", f"../Dataset/test_clean_labels_1.jsonl", out_suffix=out_suffix)


