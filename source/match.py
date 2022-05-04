from nltk.stem import WordNetLemmatizer
import re
from nlgeval import NLGEval
from typing import List, Iterable
import numpy as np
import jsonlines
from collections import Counter

wnl = WordNetLemmatizer()
symbol = set(",.")
def equal(a:str,b:str, level = 0):
    if level == 0 and (a == b or a.lower() == b.lower()):
        return True
    elif level == 1 and (a == b or a + 's' == b or a == b + 's'):
        return True
    elif level == 2 and wnl.lemmatize(a.lower(), 'n') == wnl.lemmatize(b.lower(), 'n'):
        return True
    else:
        return False


def match(string, substring, level=0):
    # print("match string: ", string)
    # print("match substring: ", substring)
    res = []
    for i, x in enumerate(string):
        if equal(x, substring[0], level):
            j = 0
            for y in substring[1:]:
                if i+j+1 < string.__len__() and equal(y, string[i+j+1],level):
                    j += 1
                    continue
                else:
                    break
            if j == substring.__len__() - 1:
                res.append(i)

    return res

def match_token(string, substring, level=0):
    res = []
    # print("match_token string: ", string)
    # print("match_token substring: ", substring)
    for i, x in enumerate(string):
        if isinstance(x, str) and equal(x, substring, level):
            res.append(i)

    return res

def word_level_edit_distance(a: List[str], b: List[str]) -> int:
    max_dis = max(len(a), len(b))
    distances = [[max_dis for j in range(len(b)+1)] for i in range(len(a)+1)]
    for i in range(len(a)+1):
        distances[i][0] = i
    for j in range(len(b)+1):
        distances[0][j] = j

    for i in range(1, len(a)+1):
        for j in range(1, len(b)+1):
            cost = 0 if a[i-1] == b[j-1] else 1
            distances[i][j] = min(distances[i-1][j] + 1,
                                  distances[i][j-1] + 1,
                                  distances[i-1][j-1] + cost)
    return distances[-1][-1]

class EditDistance():
    @staticmethod
    def edit_distance(sent1: List[str], sent2: List[str]) -> int:
        return word_level_edit_distance(sent1, sent2)

    def eval(self, hypos, references_hypo: Iterable[List[str]], references_src: Iterable[List[str]],
             src_references: Iterable[List[str]], *args) -> dict:
        src_distances = []
        hypo_distances = []
        edit_diffs = []
        new_items = []
        if len(args) > 0:
            for idx, (hypo_list, ref_h, ref_s, src_ref, item) in enumerate(zip(hypos, references_hypo, references_src, src_references, args[2])):
                if not isinstance(hypo_list[0], List):
                    hypo = hypo_list
                else:
                    hypo = hypo_list[0]
                hypo_ref_dis = self.edit_distance(hypo, ref_h)
                src_ref_dis = self.edit_distance(src_ref, ref_s)
                src_distances.append(src_ref_dis)
                hypo_distances.append(hypo_ref_dis)
                edit_diffs.append(np.sign(hypo_ref_dis - src_ref_dis))

                if args[0]:
                    if hypo_ref_dis - src_ref_dis < 0:
                        item['HebCUP_correct_labeled'] = 1
                    else:
                        item['HebCUP_correct_labeled'] = 0
                    item['edit_distance_reduced_num'] = hypo_ref_dis - src_ref_dis
                new_items.append(item)
            with jsonlines.open(re.sub("\.jsonl", "_RED.jsonl", args[1]), "w") as writer:
                writer.write_all(new_items)
        else:
            for idx, (hypo_list, ref_h, ref_s, src_ref) in enumerate(zip(hypos, references_hypo, references_src, src_references)):
                if not isinstance(hypo_list[0], List):
                    hypo = hypo_list
                else:
                    hypo = hypo_list[0]
                hypo_ref_dis = self.edit_distance(hypo, ref_h)
                src_ref_dis = self.edit_distance(src_ref, ref_s)
                src_distances.append(src_ref_dis)
                hypo_distances.append(hypo_ref_dis)
                edit_diffs.append(np.sign(hypo_ref_dis - src_ref_dis))
            print("edit_diff: ", Counter([1 if i != 0 else 0 for i in edit_diffs]))
        src_dis = float(np.mean(src_distances))
        hypo_dis = float(np.mean(hypo_distances))
        rel_dis = float(hypo_dis / src_dis)
        print(Counter(edit_diffs))
        dist_reduced_rate = Counter(edit_diffs)[-1] / len(edit_diffs)
        dist_increased_rate = Counter(edit_diffs)[1] / len(edit_diffs)

        return {"rel_distance": rel_dis, "hypo_distance": hypo_dis, "src_distance": src_dis,
                "ESS Ratio": dist_reduced_rate, "dist_increased_rate": dist_increased_rate}

def recover_desc(sent: Iterable[str]) -> str:
    return re.sub(r' <con> ', "", " ".join(sent))

class NLGMetrics():
    def __init__(self, *args, **kwargs):
        self.nlgeval = NLGEval(no_glove=True, no_skipthoughts=True)

    @staticmethod
    def prepare_sent(tokens: List[str]) -> str:
        return recover_desc(tokens)

    def eval(self, hypos, references: Iterable[List[str]]) -> dict:
        # List[str]
        first_hypos = [self.prepare_sent(hypo_list) for hypo_list in hypos]
        # List[List[str]]
        references_lists = [[self.prepare_sent(ref) for ref in references]]
        # distinct
        metrics_dict = self.nlgeval.compute_metrics(references_lists, first_hypos)
        return {
            'Bleu_4': metrics_dict['Bleu_4']
        }


