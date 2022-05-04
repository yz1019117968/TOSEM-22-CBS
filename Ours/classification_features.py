#-*- coding:utf-8 -*-
import random

import numpy as np
import pickle as pkl
from collections import Counter

import scipy.stats.mstats_basic
from tqdm import tqdm
import json
from typing import List
from Ours.token_level_check import get_cnt
from scipy import stats
import re
import matplotlib.pyplot as plt
from matplotlib import gridspec
from statsmodels.stats.multitest import multipletests
connectOp = {'.', '<con>', ''}
def get_method_len(code_change_seq):
    src_code_ch = [token[0] for token in code_change_seq if token[0] not in connectOp]
    dst_code_ch = [token[1] for token in code_change_seq if token[1] not in connectOp]
    return len(src_code_ch), len(dst_code_ch)

def itemIsConnect(item):
    if item[0] in connectOp and item[1] in connectOp:
        return True
    else:
        return False

def del_duplicate(code_change_seq: List):
    # print("code_change_seq: ", code_change_seq)
    change_parts = []
    part = []
    for idx, item in enumerate(code_change_seq):
        if item[2] != "equal":
            part.append((idx, item))
        if len(part) == 0 or part[-1][0] != idx:
            if len(part) != 0:
                change_parts.append(part)
            part = []

    # print(change_parts)
    no_idx_change_parts = []
    for part in change_parts:
        new_part = []
        for sub_part in part:
            new_part.append(sub_part[1])
        no_idx_change_parts.append(new_part)

    # self-duplicate
    sd_change_parts_a = []
    for idx, part in enumerate(no_idx_change_parts):
        new_part = []
        # del trivial tokens
        for sub_idx, sub_part in enumerate(part):
            if not itemIsConnect(sub_part):
                new_part.append(sub_part)

        # del duplicate tokens in parts
        new_part = [" ".join(token).lower() for token in new_part]
        _new_part = list(set(new_part))
        _new_part.sort(key=new_part.index)
        _new_part = [token.split(" ") for token in _new_part]
        sd_change_parts_a.append(_new_part)


    # cross-duplicate
    sd_change_parts_a.sort(key=lambda x: len(x))
    # print("sd_change_parts_a: ", sd_change_parts_a)
    if sum([len(part) for part in sd_change_parts_a]) == 0:
        return 100, 100, 100, []
    else:
        del_list = []
        for idx, part in enumerate(sd_change_parts_a):
            del_item = []
            for idxj, sub_part in enumerate(part):
                for _idx in range(idx+1, len(sd_change_parts_a)):
                    if sub_part in sd_change_parts_a[_idx]:
                        del_item.append(idxj)
                        break
            del_list.append(del_item)
            if idx == len(sd_change_parts_a) - 1:
                break
        # print("del_list: ", del_list)
        sd_change_parts_b = []
        for idx, (part, del_item) in enumerate(zip(sd_change_parts_a, del_list)):
            new_part = []
            for idxj, sub_part in enumerate(part):
                if idxj not in del_item:
                    new_part.append(sub_part)
            if len(new_part) != 0:
                sd_change_parts_b.append(new_part)
        count = sum([len(part) for part in sd_change_parts_b])
        longest_count = max([len(part) for part in sd_change_parts_b])
        change_times = len(sd_change_parts_b)
        return count, longest_count, change_times, sd_change_parts_b

def diff_code_desc(sd_change_parts_b, src_desc_tokens):
    src_code_ch = [token[0] for part in sd_change_parts_b for token in part if token[0] not in connectOp]
    dst_code_ch = [token[1] for part in sd_change_parts_b for token in part if token[1] not in connectOp]
    uniq_src_code_ch = set(src_code_ch) - set(dst_code_ch)
    uniq_dst_code_ch = set(dst_code_ch) - set(src_code_ch)
    count_non_alpha = 0
    count_alpha = 0
    rgx = re.compile(r"[a-z]+|[A-Z]+")

    for token in list(uniq_src_code_ch) + list(uniq_dst_code_ch):
        if not rgx.fullmatch(token):
            count_non_alpha += 1

    for token in uniq_src_code_ch:
        if rgx.fullmatch(token) and token not in src_desc_tokens:
            count_alpha += 1

    return count_non_alpha, count_alpha

def only_replace_sub(code_change_seq):
    total_count = 0
    others = 0
    for token in code_change_seq:

        if not itemIsConnect(token):
            if token[2] != "equal":
                total_count += 1
            if token[2] == "insert":
                others += 1
            elif token[2] == "delete":
                others += 1
    # print("others: ", others)
    # print("total_count: ", total_count)
    if total_count != 0:
        return (total_count - others) / total_count
    else:
        return 0
    # return (total_count - others) / total_count

def get_featurs_labels(dataPath, for_clf=True):
    item_list = []
    count_list = []
    longest_continous_count_list = []
    change_times_list = []
    changed_non_alpha_list = []
    changed_alpha_list = []
    method_len_diff_list = []
    cnt_list = []
    only_replace_list = []
    labels = []
    with open(dataPath, "r", encoding="utf-8") as f:
        for i, item in enumerate(tqdm(f.readlines())):
            item = json.loads(item)
            max_rank, cnt = get_cnt(item)
            only_replace_list.append(only_replace_sub(item['code_change_seq']))
            src_method_len, dst_method_len = get_method_len(item['code_change_seq'])
            method_len_diff_list.append(abs(dst_method_len - src_method_len))
            count, longest_count, change_times, sd_change_parts_b = del_duplicate(item['code_change_seq'])
            count_non_alpha, count_alpha = diff_code_desc(sd_change_parts_b, item['src_desc_tokens'])
            cnt_list.append(cnt)
            changed_non_alpha_list.append(count_non_alpha)
            changed_alpha_list.append(count_alpha)
            count_list.append(count)
            longest_continous_count_list.append(longest_count)
            change_times_list.append(change_times)
            item_list.append(item)
            labels.append(int(item['HebCUP_correct_labeled']))
    if for_clf:
        features = np.stack([only_replace_list, cnt_list, changed_non_alpha_list, longest_continous_count_list, count_list], axis=1)
        # features = np.stack([cnt_list, changed_non_alpha_list, longest_continous_count_list, count_list], axis=1)
        return features, np.array(labels), item_list
    else:
        return count_list, longest_continous_count_list, method_len_diff_list, change_times_list, changed_non_alpha_list,changed_alpha_list, cnt_list, only_replace_list, labels

def significance_test(feature, label):
    heus = []
    non_heus = []
    for f, l in zip(feature, label):
        if l == 1:
            heus.append(f)
        elif l == 0:
            non_heus.append(f)
        else:
            assert False, "incorrect value in features."
    ratio = len(heus) / len(non_heus)
    random.seed(0)
    heus = random.sample(heus, 100)
    non_heus = random.sample(non_heus, int(100 * ratio))
    return scipy.stats.mannwhitneyu(heus, non_heus, use_continuity = False, alternative="two-sided")

def check_correlations(dataPath):
    count_list,longest_continous_count_list, method_len_diff_list, change_times_list, changed_non_alpha_list,\
    changed_alpha_list, cnt_list, only_replace_list, labels = get_featurs_labels(dataPath, for_clf=False)
    _, count_list_p = significance_test(count_list, labels)
    _, longest_continous_count_list_p = significance_test(longest_continous_count_list, labels)
    _, changed_non_alpha_list_p = significance_test(changed_non_alpha_list, labels)
    _, cnt_list_p = significance_test(cnt_list, labels)
    _, only_replace_list_p = significance_test(only_replace_list, labels)
    print(f"count_list - corr_coef: {stats.pointbiserialr(labels, count_list)}, sig_test:{count_list_p}")
    print(f"longest_continous_count_list- corr_coef: {stats.pointbiserialr(labels, longest_continous_count_list)} , sig_test: {longest_continous_count_list_p}")
    # print(f"method_len_diff_list: corr_coef: {stats.pointbiserialr(labels, method_len_diff_list)}, sig_test: {significance_test(method_len_diff_list, labels)}")
    # print(f"change_times_list - corr_coef: {stats.pointbiserialr(labels, change_times_list)} , sig_test: {significance_test(change_times_list, labels)}")
    print(f"changed_non_alpha_list- corr_coef: {stats.pointbiserialr(labels, changed_non_alpha_list)} , sig_test: {changed_non_alpha_list_p}")
    # print(f"changed_alpha_list- corr_coef: {stats.pointbiserialr(labels, changed_alpha_list)} , sig_test: {significance_test(changed_alpha_list, labels)}")
    print(f"cnt_list- corr_coef: {stats.pointbiserialr(labels, cnt_list)} , sig_test: {cnt_list_p}")
    print(f"only_replace_list- corr_coef: {stats.pointbiserialr(labels, only_replace_list)} , sig_test: {only_replace_list_p}")
    print(multipletests([count_list_p, longest_continous_count_list_p, changed_non_alpha_list_p, cnt_list_p, only_replace_list_p], method="fdr_bh"))

def proposed_features_violinplot(dataPath):
    def adjacent_values(vals, q1, q3):
        upper_adjacent_value = q3 + (q3 - q1) * 1.5
        upper_adjacent_value = np.clip(upper_adjacent_value, q3, vals[-1])

        lower_adjacent_value = q1 - (q3 - q1) * 1.5
        lower_adjacent_value = np.clip(lower_adjacent_value, vals[0], q1)
        return lower_adjacent_value, upper_adjacent_value
    def setting(abcd, ax, data1, data2):
        ax.set_xticks([1,2])
        ax.set_xticklabels(['heuristic-prone', 'non-heuristic-prone'])
        for idx, pc in enumerate(abcd['bodies']):
            if idx % 2 == 0:
                pc.set_facecolor('b')
            else:
                pc.set_facecolor('r')
            pc.set_edgecolor('black')
            pc.set_alpha(1)
        data1.sort()
        data2.sort()
        quartile1_1, medians_1, quartile1_3 = np.percentile(data1, [25, 50, 75])
        quartile2_1, medians_2, quartile2_3 = np.percentile(data2, [25, 50, 75])
        quartile1 = [quartile1_1, quartile2_1]
        medians = [medians_1, medians_2]
        quartile3 = [quartile1_3, quartile2_3]
        whiskers = np.array([
            adjacent_values(sorted_array, q1, q3)
            for sorted_array, q1, q3 in zip([data1,data2], quartile1, quartile3)])
        whiskersMin, whiskersMax = whiskers[:, 0], whiskers[:, 1]
        inds = np.arange(1, len(medians) + 1)
        ax.scatter(inds, medians, marker='o', color='white', s=30, zorder=3)
        ax.vlines(inds, quartile1, quartile3, color='k', linestyle='-', lw=5)
        ax.vlines(inds, whiskersMin, whiskersMax, color='k', linestyle='-', lw=1)

    count_list,longest_continous_count_list, method_len_diff_list, change_times_list, changed_non_alpha_list,\
    changed_alpha_list, cnt_list, only_replace_list, labels = get_featurs_labels(dataPath, for_clf=False)
    h_samples = {'only_replace_list':[], 'cnt_list':[], 'changed_non_alpha_list':[],
                 'longest_continous_count_list':[], 'count_list':[]}
    nh_samples = {'only_replace_list':[], 'cnt_list':[], 'changed_non_alpha_list':[],
                 'longest_continous_count_list':[], 'count_list':[]}
    for item_1, item_2, item_3, item_4, item_5, label in zip(only_replace_list, cnt_list, changed_non_alpha_list,
                                                             longest_continous_count_list, count_list, labels):
        if label == 1:
            h_samples['only_replace_list'].append(item_1)
            h_samples['cnt_list'].append(item_2)
            h_samples['changed_non_alpha_list'].append(item_3)
            h_samples['longest_continous_count_list'].append(item_4)
            h_samples['count_list'].append(item_5)
        else:
            nh_samples['only_replace_list'].append(item_1)
            nh_samples['cnt_list'].append(item_2)
            nh_samples['changed_non_alpha_list'].append(item_3)
            nh_samples['longest_continous_count_list'].append(item_4)
            nh_samples['count_list'].append(item_5)
    gs = gridspec.GridSpec(4, 6)
    fig = plt.figure(figsize=(12, 8))
    ax1 = plt.subplot(gs[0:2, :2])
    a = ax1.violinplot(dataset=[h_samples['only_replace_list'], nh_samples['only_replace_list']])
    ax1.set_title("ReplaceRate")
    setting(a, ax1, h_samples['only_replace_list'], nh_samples['only_replace_list'])
    ax2 = plt.subplot(gs[0:2, 2:4])
    b = ax2.violinplot([h_samples['cnt_list'], nh_samples['cnt_list']])
    ax2.set_title("MatchedLevelsNum")
    setting(b, ax2, h_samples['cnt_list'], nh_samples['cnt_list'])
    ax3 = plt.subplot(gs[0:2, 4:6])
    c = ax3.violinplot([h_samples['changed_non_alpha_list'], nh_samples['changed_non_alpha_list']])
    ax3.set_title("NonLetterNum")
    setting(c,ax3,h_samples['changed_non_alpha_list'], nh_samples['changed_non_alpha_list'])
    ax4 = plt.subplot(gs[2:4, 1:3])
    d = ax4.violinplot([h_samples['longest_continous_count_list'], nh_samples['longest_continous_count_list']])
    ax4.set_title("LongestChangedSeq")
    setting(d,ax4,h_samples['longest_continous_count_list'], nh_samples['longest_continous_count_list'])
    ax5 = plt.subplot(gs[2:4, 3:5])
    e = ax5.violinplot(dataset=[h_samples['count_list'], nh_samples['count_list']])
    ax5.set_title("TotalChangedNum")
    setting(e, ax5, h_samples['count_list'], nh_samples['count_list'])
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # check_correlations("../Dataset/train_clean_labels_1.jsonl")
    # proposed_features_violinplot("../Dataset/train_clean_labels.jsonl")
    print(get_featurs_labels("../Dataset/test_clean_labels_1.jsonl",False))
