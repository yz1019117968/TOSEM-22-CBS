# coding:utf-8

import json
import re
from tqdm import tqdm
from typing import List,Dict
stripAll = re.compile('[\s]+')
from source.match import match, match_token, equal, EditDistance, NLGMetrics
from collections import defaultdict
from copy import deepcopy
stop_words = {}
connectOp = {'.', '<con>'}
symbol = {"{","}",":",",","_",".","-","+",";","<con>"}
import jsonlines

def lookBack(code_change_seq):
    """
    :param code_change_seq: code change operation sequence
    :return: possible mapping from old comment to new comment
    """
    def itemIsConnect(item):
        if item[0] in connectOp or item[1] in connectOp:
            return True
        else:
            return False

    def combineTuple(mixedTuple):
        res = tuple()
        for x in mixedTuple:
            if isinstance(x, tuple):
                res += x
            else:
                res += tuple((x,))

        if res.__len__() and res[0] in connectOp:
            res = tuple(res[1:])
        if res.__len__() and res[-1] in connectOp:
            res = tuple(res[:-1])
        return res

    def getSubsetMapping(modifiedMapping):
        tempMapping = deepcopy(modifiedMapping)
        for buggyWord in tempMapping:
            for fixedWord in tempMapping[buggyWord]:
                if buggyWord.__len__() == fixedWord.__len__():
                    for i in range(buggyWord.__len__()):
                        for j in range(i + 1, buggyWord.__len__() + 1):
                            if buggyWord[i:j][0] not in connectOp and buggyWord[i:j][-1] not in connectOp \
                                    and fixedWord[i:j][0] not in connectOp and fixedWord[i:j][-1] not in connectOp \
                                    and buggyWord[i:j] != fixedWord[i:j]:
                                modifiedMapping[tuple(buggyWord[i:j])].add(tuple(fixedWord[i:j]))
                else:
                    tempBuggy = list(buggyWord)
                    tempFixed = list(fixedWord)

                    '''
                    Find different part
                                    (pop ->)___________x___(<- pop)
                                    (pop ->)___________xx___(<- pop)
                    '''
                    left_i, left_j, right_i, right_j = 0, 0, tempBuggy.__len__() - 1, tempFixed.__len__() - 1
                    while left_i < tempBuggy.__len__() and left_j < tempFixed.__len__():
                        if tempBuggy[left_i].lower() == tempFixed[left_i].lower():
                            left_i += 1
                            left_j += 1
                        else:
                            left_i = max(0, left_i - 1)
                            left_j = max(0, left_j - 1)
                            break
                    if left_i == tempBuggy.__len__() or left_j == tempFixed.__len__():
                        left_i = max(0, left_i - 1)
                        left_j = max(0, left_j - 1)

                    while right_i >= left_i and right_j >= left_j:
                        if tempBuggy[right_i].lower() == tempFixed[right_j].lower():
                            right_i -= 1
                            right_j -= 1
                        else:
                            right_i += 1
                            right_j += 1
                            break
                    if right_i < 0 or right_j < 0:
                        return modifiedMapping
                    alignedBuggy = tempBuggy[:left_i] + [tuple(tempBuggy[left_i:right_i + 1])] + tempBuggy[right_i + 1:]
                    alignedFixed = tempFixed[:left_j] + [tuple(tempFixed[left_j:right_j + 1])] + tempFixed[right_j + 1:]

                    for i in range(alignedBuggy.__len__()):
                        for j in range(i + 1, alignedFixed.__len__() + 1):
                            key = combineTuple(alignedBuggy[i:j])
                            value = combineTuple(alignedFixed[i:j])
                            if key != value and key.__len__() != 0 and value.__len__() != 0:
                                modifiedMapping[key].add(value)
        return modifiedMapping

    buggyWords = []
    fixedWords = []
    allIndex = []
    lastItem = ['', '', 'equal']
    preHasValidOp = False
    modifiedMapping = defaultdict(set)
    for i, x in enumerate(code_change_seq):
        if x[2] != 'equal':
            allIndex.append(i)
            preHasValidOp = True
        elif (itemIsConnect(lastItem) or itemIsConnect(x)) and preHasValidOp:
            allIndex.append(i)
        else:
            preHasValidOp = False
        lastItem = x

    for i, index in enumerate(allIndex):
        connectFlag = False
        lastItem = code_change_seq[index]
        reversedSeq = list(reversed(code_change_seq[:index]))
        curBuggyWords = []
        curFixedWords = []
        for j, seq in enumerate(reversedSeq):
            if j < index and reversedSeq[j][0] in connectOp or connectFlag:
                curBuggyWords.append(lastItem[0]) if not curBuggyWords.__len__() else None
                curBuggyWords.append(reversedSeq[j][0])
                connectFlag = True

            if j < index and reversedSeq[j][1] in connectOp or connectFlag:
                curFixedWords.append(lastItem[1]) if not curFixedWords.__len__() else None
                curFixedWords.append(reversedSeq[j][1])
                connectFlag = True
            if j < index and reversedSeq[j][0] not in connectOp and reversedSeq[j][1] not in connectOp:
                if connectFlag is False:
                    break
                connectFlag = False
        buggyWords.append(tuple(reversed(tuple(x for x in curBuggyWords if x!=''))))
        fixedWords.append(tuple(reversed(tuple(x for x in curFixedWords if x!=''))))
        if buggyWords[-1].__len__() != 0 and fixedWords[-1].__len__() != 0:
            modifiedMapping[buggyWords[-1]].add(fixedWords[-1])
        if code_change_seq[index][2] == 'replace' and code_change_seq[index][0] not in symbol and code_change_seq[index][1] not in symbol:
            modifiedMapping[tuple((code_change_seq[index][0],))].add(tuple((code_change_seq[index][1],)))

    modifiedMapping = getSubsetMapping(modifiedMapping)
    return modifiedMapping


def getTokenStream(fileInfo):
    """
    Extract infomation stream from preprocessed data file.
    :param fileInfo: Preprocessed data of single file
    :return: old code token stream, new code token stream, old comment token stream, new comment token stream, changed token.
    """
    if "code_change_seq" not in fileInfo:
        return False
    codeSeq = fileInfo["code_change_seq"]
    # print("codeSeq len: ", len(codeSeq))
    buggyStream = []
    fixedStream = []
    changed = set()
    for x in codeSeq:
        buggyStream.append(x[0])
        fixedStream.append(x[1])
        if x[2] != "equal":
            changed.add(x[0].lower()) if x[0] != '' and x[0] != '<con>' and x[0].isalpha() and x[0] not in stop_words else None
            changed.add(x[1].lower()) if x[1] != '' and x[1] != '<con>' and x[1].isalpha() and x[1] not in stop_words else None
    buggyStream = [x.lower() for x in buggyStream if x != '' and x !='<con>' and x not in stop_words]
    fixedStream = [x.lower() for x in fixedStream if x != ''and x != '<con>' and x not in stop_words]
    oldComment = [x for x in fileInfo["src_desc_tokens"] if x != '']
    newComment = [x for x in fileInfo["dst_desc_tokens"] if x != '']
    return buggyStream, fixedStream, oldComment, newComment, changed



def sortMapping(streamPair):
    """
    Sort mapping by complexity
    :param streamPair:
    :return:
    """
    modifiedMapping = streamPair[5]
    possibleMapping = []
    for x in modifiedMapping:
        modifiedMapping[x] = list(modifiedMapping[x])
        modifiedMapping[x].sort(key=lambda x:x.__len__(), reverse=True)
        possibleMapping.append((x, modifiedMapping[x]))
    possibleMapping.sort(key=lambda x: x[0].__len__(), reverse=True)
    return possibleMapping


def evaluateCorrectness_mixup(possibleMapping, streamPair, k=1, ignore_case=True):
    """
    Evaluate the effectiveness of HebCup.
    :param possibleMapping: Possible token mapping from old comment to new comment
    :param streamPair:
    :param k: top k
    :return:
    """
    def genAllpossible(pred):
        allCur = [[]]
        if pred is None:
            return []
        for x in pred:
            tepAllCur = allCur.copy()
            for i in range(allCur.__len__()):
                if isinstance(x, str):
                    tepAllCur[i].append(x)
                elif isinstance(x, list):
                    cur = tepAllCur[i].copy()
                    tepAllCur[i] = None
                    for dst in x:
                        tepAllCur.append(cur + list(dst))
            allCur = [x for x in tepAllCur if x is not None]
        return allCur


    def isEqual_token(pred: List[str], oracle, k):
        if k==1 and pred:
            return Equal_1(pred[0], oracle)
        elif k > 1:
            return Equal_k(pred, oracle, k)
        else:
            return False

    def isEqual(pred, oracle):
        predStr = stripAll.sub(' ', " ".join(pred).replace("<con>", '')).strip(' !"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~_')
        oracleStr = stripAll.sub(' ', " ".join(oracle).replace("<con>", '')).strip(' !"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~_')
        predStr = "".join([x for x in predStr if x.isalnum()])
        oracleStr = "".join([x for x in oracleStr if x.isalnum()])
        if predStr.lower() == oracleStr.lower():
            return True
        else:
            return False

    def Equal_1(pred, oracle):
        predStr = "".join(pred).replace("<con>", '')
        oracleStr = "".join(oracle).replace("<con>", '')
        if predStr.lower() == oracleStr.lower():
            return True
        else:
            return False

    def Equal_k(pred: List[str], oracle, k):
        pred.sort(key=lambda x:x.__len__(), reverse=True)
        pred = pred[:k]
        for x in pred:
            if Equal_1(x, oracle):
                return True
        return False

    def split(comment: List[str]):
        comment = " ".join(comment).replace(" <con> ,", " ,").replace(" <con> #", " #").replace(" <con> (", " (") \
            .replace(" <con> )", " )").replace(" <con> {", " {").replace(" <con> }", " }").replace(" <con> @", " @")\
            .replace("# <con> ", "# ").replace(" <con> ", "").strip(' !"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~_')
        return comment.split(" ")

    def tryAllPossible(possibleMapping, streamPair,matchLevel, k, ignore_case=True):
        cnt = 0
        predComment_token, predComment_subtoken = None, None
        oldComment_token, oldComment_subtoken = None, None
        newComment_token = split(streamPair[3])
        newComment_subtoken = streamPair[3]

        for x in possibleMapping:
            # print("x: ", x)
            if cnt >= 1:
                break
            if oldComment_token is None:
                oldComment_token = split(streamPair[2])
                oldComment_subtoken = streamPair[2]
                # print("oldComment_subtoken: ", oldComment_subtoken)
            pattern_token = " ".join(x[0]).replace(" <con> ", "").replace(" . ",".")
            # print("pattern_token: ", pattern_token)
            pattern_suboten = [x.lower() for x in x[0]]
            # print("pattern_suboten: ", pattern_suboten)
            pattern_splited = [x.lower() for x in x[0] if x !="<con>"]
            # print("pattern_splited: ", pattern_splited)
            indexes_token = match_token(oldComment_token, pattern_token, matchLevel)
            indexes_subtoken = match(oldComment_subtoken, pattern_suboten, matchLevel)
            indexes_splited = match(oldComment_subtoken, pattern_splited, matchLevel) if pattern_splited else None
            if not indexes_token:
                pass
            else:
                if equal(pattern_token, oldComment_token[indexes_token[0]],1) and not equal(pattern_token,oldComment_token[indexes_token[0]],0):
                    if pattern_token[-1] != 's':
                        x[1][0] = tuple((x[1][0][0] + 's',))
                    else:
                        x[1][0] = tuple((x[1][0][0][:-1],))
                # print("indexes_token: ", indexes_token)
                for index in indexes_token:
                    # print("indexes_token x: ", x)
                    oldComment_token[index] = x[1]
                    # print("updated comment: ", oldComment_token)
                    predComment_token = oldComment_token
                cnt += 1
            # print("indexes_subtoken:", indexes_subtoken)
            if indexes_subtoken:
                bias = 0
                for index in indexes_subtoken:
                    predComment_subtoken = oldComment_subtoken[:index + bias] + list(x[1][0]) + oldComment_subtoken[index + pattern_suboten.__len__() + bias:]
                    oldComment_subtoken = predComment_subtoken
                    bias = bias + x[1][0].__len__() - x[0].__len__()
                cnt += 1
            # print("indexes_splited:", indexes_splited)
            if indexes_splited:
                bias = 0
                for index in indexes_splited:
                    predComment_subtoken = oldComment_subtoken[:index + bias] + [y for y in list(x[1][0]) if y != "<con>"] + oldComment_subtoken[index + pattern_splited.__len__() + bias:]
                    oldComment_subtoken = predComment_subtoken
                    bias = bias + x[1][0].__len__() - x[0].__len__()
                cnt += 1

        predComment_token = genAllpossible(predComment_token)

        if predComment_token is not None and isEqual_token(predComment_token, newComment_token, k):
            if ignore_case:
                newComment_token = " ".join(streamPair[3]).lower().split()
            else:
                newComment_token = " ".join(streamPair[3]).split()
            return True, newComment_token, newComment_token, streamPair[3], "token"
        elif predComment_subtoken is not None and isEqual(predComment_subtoken, newComment_subtoken):
            if ignore_case:
                predComment_subtoken_b = " ".join(predComment_subtoken).lower().split()
                newComment_subtoken = " ".join(newComment_subtoken).lower().split()
            else:
                predComment_subtoken_b = " ".join(predComment_subtoken).split()
                newComment_subtoken = " ".join(newComment_subtoken).split()
            return True, predComment_subtoken_b, newComment_subtoken, predComment_subtoken, "subtoken"
        if cnt == 0:
            if ignore_case:
                return None, " ".join(streamPair[2]).lower().split(), " ".join(streamPair[3]).lower().split(), streamPair[2], "subtoken"
            else:
                return None, " ".join(streamPair[2]).split(), " ".join(streamPair[3]).split(), \
                       streamPair[2], "subtoken"
        else:
            if predComment_subtoken is not None:
                if ignore_case:
                    predComment_subtoken_1 = " ".join(predComment_subtoken).lower().split()
                    newComment_subtoken = " ".join(newComment_subtoken).lower().split()
                else:
                    predComment_subtoken_1 = " ".join(predComment_subtoken).split()
                    newComment_subtoken = " ".join(newComment_subtoken).split()
                return False, predComment_subtoken_1, newComment_subtoken, predComment_subtoken, "subtoken"
            else:
                if ignore_case:
                    return False, " ".join(streamPair[2]).lower().split(), " ".join(streamPair[3]).lower().split(), streamPair[2], "subtoken"
                else:
                    return False, " ".join(streamPair[2]).split(), " ".join(streamPair[3]).split(), \
                           streamPair[2], "subtoken"


    for i in range(3):
        matchRes, predComment, newComment, predComment_raw, granularity = tryAllPossible(possibleMapping, streamPair, matchLevel=i, k=k, ignore_case=ignore_case)
        if matchRes is None:
            continue
        elif matchRes is True:
            return True, predComment, newComment, predComment_raw, granularity
        else:
            return False, predComment, newComment, predComment_raw, granularity
    if ignore_case:
        return False, " ".join(streamPair[2]).lower().split(), " ".join(streamPair[3]).lower().split(), streamPair[2], "subtoken"
    else:
        return False, " ".join(streamPair[2]).split(), " ".join(streamPair[3]).split(), streamPair[2], "subtoken"

def predict(dataPath, K, store_labels=False, output_ds_splitted_by_hebcup=False, store_labels_RED=False, ignore_case=True, output_obs=False):
    """
    Run predict to conduct the comment update by HebCUP.
    :param dataPath:
    :param K:
    :param store_labels: if True, set a new key (HebCUP_correct_labeled) for each item of the dataset, when K=1(5), output new items for K=1(5).
    :param output_ds_splitted_by_hebcup: if True, output the new items with two dataset (CUP Side & HebCUP Side) that split by HebCUP.
    :return:
    """
    allRec = []
    newComments_1 = []
    oldComments = []
    all_items = []
    with open(dataPath, 'r', encoding='utf8') as f:
        for i, x in enumerate(tqdm(f.readlines())):
            fileInfo = json.loads(x)
            # if fileInfo['sample_id'] == 3506201:
            buggyStream, fixedStream, src_desc, dst_desc, changed = getTokenStream(fileInfo)
            modifiedMapping = lookBack(fileInfo["code_change_seq"])
            allRec.append([buggyStream, fixedStream, src_desc, dst_desc, changed, modifiedMapping])
            if ignore_case:
                newComments_1.append(" ".join(dst_desc).lower().split())
                oldComments.append(" ".join(src_desc).lower().split())
            else:
                newComments_1.append(" ".join(dst_desc).split())
                oldComments.append(" ".join(src_desc).split())
            all_items.append(fileInfo)
                # break
    correct = 0
    all_results = []
    predComments = []
    newComments_2 = []
    false_items = []
    true_items = []
    for i, streamPair in enumerate(allRec):
        possibleMapping = sortMapping(streamPair)
        evalRes_test, predComment, newComment, predComment_raw, granularity = evaluateCorrectness_mixup(possibleMapping, streamPair, k=K, ignore_case=ignore_case)

        if evalRes_test:
            if store_labels:
                all_items[i]['HebCUP_correct_labeled'] = 1
            if output_ds_splitted_by_hebcup:
                true_items.append(all_items[i])
            correct += 1
            all_results.append(1)
        else:
            if store_labels:
                all_items[i]['HebCUP_correct_labeled'] = 0
            if output_ds_splitted_by_hebcup:
                false_items.append(all_items[i])
            all_results.append(0)
        if K == 1:
            predComments.append(predComment)
            newComments_2.append(newComment)
    if K == 1:
        if store_labels_RED:
            print(EditDistance().eval(predComments, newComments_2, newComments_1, oldComments, store_labels_RED, dataPath, all_items))
        else:
            print(EditDistance().eval(predComments, newComments_2, newComments_1, oldComments))
        print(NLGMetrics().eval(predComments,newComments_2))
    print(f"correct count: {correct}, total count: {allRec.__len__()}")
    print("Accuracy:", correct/allRec.__len__())
    if store_labels:
        with jsonlines.open(re.sub("\.jsonl", f"_labels_{K}.jsonl", dataPath), "w") as writer:
            writer.write_all(all_items)

    if output_ds_splitted_by_hebcup:
        with jsonlines.open(re.sub("\.jsonl", "_false.jsonl", dataPath), "w") as writer:
            writer.write_all(false_items)
        with jsonlines.open(re.sub("\.jsonl", "_true.jsonl", dataPath), "w") as writer:
            writer.write_all(true_items)

    if output_obs:
        with open("../Dataset/hebcup_obs.txt", "w", encoding="utf-8") as fr:
            for item, pred in zip(all_items, predComments):
                fr.write(str(item['sample_id']) + ": \n")
                fr.write(item['src_desc'] + "\n")
                fr.write(item['dst_desc'] + "\n")
                fr.write(" ".join(pred) + "\n\n")


if __name__ == '__main__':
    # predict(f"../Dataset/test_clean.jsonl", K=1, output_obs=True, ignore_case=False)
    # predict(f"../Dataset/splitted_test_sets/lightGBM/test_clean_true.jsonl", K=1)
    # predict("../Dataset/test_clean.jsonl", K=1, output_ds_splitted_by_hebcup=True)
    # predict("../Dataset/splitted_test_sets/lightGBM/HCUP_test20220304_ReplaceRate_MatchedLevelNum_TotalChangedNum.jsonl", K=5)
    import sys
    dataPath = sys.argv[1]
    K = int(sys.argv[2])
    if sys.argv[3].lower() == "true":
        store_labels = True
    elif sys.argv[3].lower() == "false":
        store_labels = False
    else:
        raise Exception("pls input either true or false!")
    predict(dataPath, K, store_labels, False, False, False)
