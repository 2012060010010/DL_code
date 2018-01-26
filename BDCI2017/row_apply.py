import pandas as pd
import numpy as np
import re
from pyltp import SentenceSplitter
import os
import copy
from collections import Counter
from tqdm import tqdm
from collections import defaultdict
import jieba
import jieba.analyse
import jieba.posseg
import pickle
import json

pattern = re.compile(r"[\w\u4e00-\u9fa5]+")
no_chn = re.compile("^[a-zA-Z\d]+$")
punc_pattern = re.compile(r"[^\u4e00-\u9fa5\w\d]+")


def answer_process(row):
    themes = row["theme-主题"][:-1].split(";")
    themes = [x.strip() for x in themes]
    themes = ";".join(themes) + ";"
    if row["theme-主题"] != themes:
        print(row["theme-主题"], " 修正后:", themes)
    row["theme-主题"] = themes
    
    sentis = row["sentiment_word-情感关键词"][:-1].split(";")
    sentis = [x.strip() for x in sentis]
    sentis = ";".join(sentis) + ";"
    if row["sentiment_word-情感关键词"] != sentis:
        print(row["sentiment_word-情感关键词"], " 修正后:", sentis)
    row["sentiment_word-情感关键词"] = sentis
    
    anls = row["sentiment_anls-情感正负面"][:-1].split(";")
    anls = [x.strip() for x in anls]
    anls = ";".join(anls) + ";"
    if row["sentiment_anls-情感正负面"] != anls:
        print(row["sentiment_anls-情感正负面"], " 修正后:", anls)
    row["sentiment_anls-情感正负面"] = anls
    return row


def content_filter(row):
    content = row.iloc[1]
    content = content.replace("[追评]", "")
    return content


#需要去除的符号  
symbol = ['【', '】', '<br/>' ,'&hellip', '.', '&ldquo', '<br/>', '&', '(', ')', '（', '）', '[', ']']
def remove_symbol(row):
    content = row.iloc[1]
    for s in symbol:
        content = content.replace(s, ' ')
    content = content.replace("*", '')
    return content


#需要替换的短语  
with open('data/pair_replace1.txt', encoding='utf-8') as f:
    be_replace = {}
    for line in f:
        item = line.strip().split('\t')
        be_replace[item[1]] = item[0]
        
def replace_phrase(row):
    content = row.iloc[1]
    for br in iter(be_replace):
        content = content.replace(br, be_replace[br])
    return content


def tokenized_sub_sents(row):
    content = row.iloc[1]
    sub_sents = []
    sub_tags = []
    sents = SentenceSplitter.split(content)
    for sent in sents:
        subs = [x for x in re.split(punc_pattern, sent) if x]
        subss = [jieba.posseg.cut(x, HMM=False) for x in subs if not re.findall(no_chn, x)]
        tags =[]
        subs = []
        for s in subss:
            tag = []
            sub = []
            for t0, t1 in s:
                tag.append(t1)
                sub.append(t0)
            tags.append(tag)
            subs.append(sub)
        assert len(tags) == len(subs)
        sub_sents.extend(subs)
        sub_tags.extend(tags)
#     print(sub_sents, sub_tags)
    row["sub_sents_tokenized"] = sub_sents
    row["sub_sents_postagged"] = sub_tags
    return row



def tag_seq_gen_based_char_v2(row):
    content = row.iloc[1]
    themes = row.iloc[2][:-1].split(";")
    senti_w = row.iloc[3][:-1].split(";")
    senti_v = row.iloc[4][:-1].split(";")
    
    #对主题和情感词按长度进行排序
    def compute_length(string):
        if string == "NULL":
            return 0
        else:
            return len(string)

    tmp = [(theme, w, value, compute_length(theme)+compute_length(w)) for theme, w, value in zip(themes, senti_w, senti_v)]
    tmp = sorted(tmp, key = lambda x :x[3], reverse=True)
    
    themes = [x[0] for x in tmp]
    senti_w = [x[1] for x in tmp]
    senti_v = [x[2] for x in tmp]
    
    sub_sents = []
    sents = SentenceSplitter.split(content)
    for sent in sents:
        subs = [x for x in re.split(punc_pattern, sent) if x]
        subs = [x for x in subs if not re.findall(no_chn, x)]
        sub_sents.extend(subs)
    sub_sents = [[sent, 0] for sent in sub_sents] # 其中0代表目前句子中被标注出pair的数量
    
    sub_tags = [["o"] *len(sent[0]) for sent in sub_sents]
    sub_values = [["2"] for sent in sub_sents]
    not_found = []
    
    for theme, senti, value in zip(themes, senti_w, senti_v):
        is_found = False
        if senti == "":
            continue
        for i, sent in enumerate(sub_sents):
            try:
                p_senti = sent[0].index(senti)
                if theme != "NULL":
                    p_theme = sent[0].index(theme)
                    sl = len(senti)
                    tl = len(theme)
                    s_tags = copy.deepcopy(sub_tags[i])
                    conflict = False # 欲标注位置是否已被标注
                    # 标注情感词
                    for j in range(sl):
                        if s_tags[j+p_senti] == "o":
                            s_tags[j+p_senti] = "s{}".format(sent[1])
                        else:
                            conflict = True
                            break #当前句无法标注， 转入下一句
                            
                    # 以下if语句块解决同样的情感词实际有两个（991情况），但是都想标注第一个的冲突，可以直接去掉
                    # 1538->1460
                    conflict2 = False
                    if conflict and (sent[1]>0):
                        p_senti = sent[0][p_senti+sl:].index(senti) + p_senti +sl
                        # 标注情感词
                        for j in range(sl):
                            if s_tags[j+p_senti] == "o":
                                s_tags[j+p_senti] = "s{}".format(sent[1])
                            else:
                                conflict2 = True
                                break #当前句无法标注， 转入下一句
                    if conflict2 == False:
                        conflict = False
                    
                    if conflict == False:
                        # 标注主题词
                        for k in range(tl):
                            if s_tags[k+p_theme] == "o":
                                s_tags[k+p_theme] = "t{}".format(sent[1])
                            else:
                                conflict = True
                                break
                    
                     # 以下if语句块解决同样的主题词实际有两个（151情况），但是都想标注第一个的冲突，可以直接去掉
                    # 1460->1433
                    conflict2 = False
                    if conflict and (sent[1]>0):
                        p_theme = sent[0][p_theme+tl:].index(theme) + p_theme + tl
                        for j in range(tl):
                            if s_tags[j+p_theme] == "o":
                                s_tags[j+p_theme] = "t{}".format(sent[1])
                            else:
                                conflict2 = True
                                break 
                    if conflict2 == False:
                        conflict = False
                        
                        
                    if conflict == False:
                        # 总体标注下来,没有冲突
                        assert len(s_tags) == len(sub_tags[i])
                        sub_tags[i] = s_tags
                        if sub_sents[i][1] == 0:
                            sub_values[i][0] = value
                        else:
                            sub_values[i].append(value)
                        sub_sents[i][1] += 1
                        is_found = True
                        break # 当前pair已找到，转入下一个pair
                else:
                    sl = len(senti)
                    s_tags = copy.deepcopy(sub_tags[i])
                    conflict = False
                    # 标注情感词
                    for j in range(sl):
                        if s_tags[j+p_senti] == "o":
                            s_tags[j+p_senti] = "s{}".format(sent[1])
                        else:
                            conflict = True
                            break #当前句无法标注， 转入下一句 
                            
                    # 以下if语句块解决同样的情感词实际有两个（991情况），但是都想标注第一个的冲突，可以直接去掉
                    # 1538->1460
                    conflict2 = False
                    if conflict and (sent[1]>0):
                        p_senti = sent[0][p_senti+sl:].index(senti) + p_senti +sl
                        # 标注情感词
                        for j in range(sl):
                            if s_tags[j+p_senti] == "o":
                                s_tags[j+p_senti] = "s{}".format(sent[1])
                            else:
                                conflict2 = True
                                break #当前句无法标注， 转入下一句
                    if conflict2 == False:
                        conflict = False
                        
                    if conflict == False:
                        # 总体标注下来,没有冲突
                        assert len(s_tags) == len(sub_tags[i])
                        sub_tags[i] = s_tags
                        if sub_sents[i][1] == 0:
                            sub_values[i][0] = value
                        else:
                            sub_values[i].append(value)
                        sub_sents[i][1] += 1
                        is_found = True
                        break # 当前pair已找到，转入下一个pair
            except ValueError:
                # 当前句无法找到pair
                continue
        if is_found == False:
            not_found.append((theme, senti, value))
    
    row["not_found_pair"] = not_found if not_found else False
    row["sub_sents_tags_based_char"] = sub_tags
    row["sub_sents_values"] = sub_values
    return row

def tag_sort(row):
    # 对子句的标签进行重排，句子前端的pair，其标签必然是数字小的
    tags_based_char = row.loc["sub_sents_tags_based_char"]
    for i, sub_tags in enumerate(tags_based_char):
        tmp1 = [int(x[1]) for x in sub_tags if len(x) > 1]
        if tmp1:
            tmp2 = sorted(tmp1)
            if tmp1 == tmp2:
                new_subtags = sub_tags
            else:
                num = tmp2[-1]
                new_subtags = copy.deepcopy(sub_tags)
                tmp3 = list(set(tmp1))
                tmp3.sort(key=tmp1.index)
                for j, k in zip(range(num+1), tmp3):
                    for _, tag in enumerate(sub_tags):
                        if len(tag) >1  and tag[1] == str(k):
                            new_subtags[_] = tag[0] + str(j)
                        else:
                            pass
        else:
            new_subtags = sub_tags
        tags_based_char[i] = new_subtags
    return tags_based_char

def tag_seq_gen_based_word_v2(row):
    """将基于char的标注序列转换为基于词的"""
    def merge_tags(ts,word=None, words=None):
#         用于构建词表
#         tt = [int(x[1]) for x in ts if len(x) > 1]
#         if len(set(tt)) > 1:
#             print(ts,word,words)
#         return "o"
        ts = "".join(ts)
        if "t" in ts:
            return re.findall(r"t[0-9]", ts)[0]
        elif "s" in ts:
            return re.findall(r"s[0-9]", ts)[0]
        else:
            return "o"
    sub_sents_tokenized = row.loc["sub_sents_tokenized"]
    sub_sents_tags = row.loc["sub_sents_tags_based_char"]
    sub_sents_tags_based_word = []
    for words, tags, in zip(sub_sents_tokenized, sub_sents_tags):
        ts = []
        for word in words:
            ts.append(merge_tags(tags[:len(word)], word=word,words=words))
            tags = tags[len(word):]
        sub_sents_tags_based_word.append(ts)
    return sub_sents_tags_based_word

def tag_trans(tagtag):
    tag = copy.deepcopy(tagtag)
    l = len(tag)
#     print(tag)
#     if "ss" in tag:
#         print(tag)
    tmp = [int(x[1]) for x in tag if len(x)>1]
    if not tmp:
        return tag
    num = max(tmp)
    for i in range(num+1):
        num_t = len([x for x in tag if x=="t{}".format(i)])
        num_s = len([x for x in tag if x=="s{}".format(i)])
#         print(num_t, num_s)
        if num_t == 1:
            index = tag.index("t{}".format(i))
            tag[index] = "ts"
        elif num_t == 2:
            index = tag.index("t{}".format(i))
            tag[index] = "tb"
            index = tag.index("t{}".format(i))
            tag[index] = "te"
#             print(tag)
        elif num_t > 2:
            index = tag.index("t{}".format(i))
            tag[index] = "tb"
            index = tag[::-1].index("t{}".format(i))
            tag[-(index+1)] = "te"
            for j in range(num_t-2):
                index = tag.index("t{}".format(i))
                tag[index] = "ti"
        if num_s == 1:
#             print(tag)
            index = tag.index("s{}".format(i))
            tag[index] = "ss"
        elif num_s == 2:
            index = tag.index("s{}".format(i))
            tag[index] = "sb"
            index = tag.index("s{}".format(i))
            tag[index] = "se"
        elif num_s > 2:
            index = tag.index("s{}".format(i))
            tag[index] = "sb"
#             print(tag[::-1])
            index = tag[::-1].index("s{}".format(i))
#             print(index)
            tag[-(index+1)] = "se"
#             print(tag)
            for j in range(num_s-2):
                index = tag.index("s{}".format(i))
                tag[index] = "si" 
#             print(tag)
    assert len(tag) == l
    return tag
def tag2bies(row, source_columns = "sub_sents_tags_based_word"):
    sub_sents_tags = row[source_columns]
#     print(sub_sents_tags, row["sub_sents_tokenized"])
    assert len(sub_sents_tags) == len(row["sub_sents_tokenized"])
    new_tags = []
    for subtag in sub_sents_tags:
        new_tags.append(tag_trans(subtag))
#     print(sub_sents_tags)
#     print(new_tags)
    assert len(new_tags) == len(sub_sents_tags)
    return new_tags

def chartag2bies(row, source_columns = "sub_sents_tags_based_char"):
    sub_sents_tags = row[source_columns]
    assert len(sub_sents_tags) == len(row["sub_sents_tokenized"])
    new_tags = []
    for subtag in sub_sents_tags:
        new_tags.append(tag_trans(subtag))
    assert len(new_tags) == len(sub_sents_tags)
    return new_tags

def word2features(sent, postags, i):
    word = sent[i]
    postag = postags[i]
    features = {
        "word_location": i,
        "word": word,
        "word_first": word[0],
        "word_last": word[-1],
        "postag": postag,
        "postag_first": postag[0],
        "sent_length": len(sent),
    }

        
    if i > 0:
        word1eft = sent[i-1]
        postag1eft = postags[i-1]
        left_de = True if word1eft == "的" else False
        features.update({
            "left_is_de": left_de,
            "left_word":word1eft,
            'leftpostag': postag1eft,
            'left_postag_first': postag1eft[0],
            "2gram_pos_left": postag1eft + "|" + postag,
            "2gram_left": word1eft + "|" + word
        })
    else:
        features['BOS'] = True

    if i < len(sent)-1:
        wordright = sent[i+1]
        postagright = postags[i+1]
        right_de = True if wordright == "的" else False
        features.update({
            "right_is_de": right_de,
            "rightword":wordright,
            'right_postag': postagright,
            'right_postag[0]': postagright[0],
            "2gram_pos_right": postag + "|" + postagright,
            "2gram_right": word + "|" + wordright
        })
    else:
        features['EOS'] = True
    
    if i > 1:
        postagleft = postags[i-1]
        wordleft2 = sent[i-2]
        postag1eft2 = postags[i-2]
        features.update({
            "left_word2": wordleft2,
            'leftpostag2': postag1eft2,
            "3gram_pos_left": postag1eft2 + "|" + postagleft + "|" + postag,
            "3gram_left": wordleft2 + "|" + word1eft + "|" + word
        })
    
    if i < len(sent)-2:
        wordright2 = sent[i+2]
        postagright2 = postags[i+2]
                
        features.update({
            "rightword2":wordright2,
            'right_postag2': postagright2,
            "3gram_pos_right": postag + "|" + postagright + "|" + postagright2,
            "3gram_right": word + "|" + wordright + "|" + wordright2
        })
    
    
    features["num_noun_left"] = len([x for x in postags[:i] if x[0] == "n"])
    features["num_adj_left"] = len([x for x in postags[:i] if x[0] == "a"])
    features["num_adv_left"] = len([x for x in postags[:i] if x[0] == "d"])
    features["num_v_left"] = len([x for x in postags[:i] if x[0] == "v"])
    features["num_noun_right"] = len([x for x in postags[i:] if x[0] == "n"])
    features["num_adj_right"] = len([x for x in postags[i:] if x[0] == "a"])
    features["num_adv_right"] = len([x for x in postags[i:] if x[0] == "d"])
    features["num_v_right"] = len([x for x in postags[i:] if x[0] == "v"])
    
    return features

def sent2features(sent, postags):
    return [word2features(sent, postags, i) for i in range(len(sent))]

def features_gen(row):
    sub_sents_features = []
#     third_result = [row["third_party_theme"], row["third_party_senti"]]
#     compress_seq = row["compress_labels"]
#     compress_seq = None
    for sub_sents, sub_tags in zip(row.loc["sub_sents_tokenized"], row.loc["sub_sents_postagged"]):
        sub_sents_features.append(sent2features(sub_sents, sub_tags))
    return sub_sents_features


def tag2answer_v21(row, tag_column="pred_tags"):    
    pred_tags = copy.deepcopy(row.loc[tag_column])
    sub_sents = row.loc["sub_sents_tokenized"]
    sub_postags = row.loc["sub_sents_postagged"]
    themes = []
    senti_w = []
    
    for tags, words, postags in zip(pred_tags, sub_sents, sub_postags):
        print(tags, words)
        assert len(tags) == len(words)
        is_exist = [int(x[1]) for x in tags if len(x)>1]
        if is_exist:
            num = max(is_exist)
            for i in range(num+1):
                theme = []
                senti = []
                for tag, word, ptag in zip(tags,words, postags):
                    if tag == "t{}".format(i):
                        theme.append(word)
                    if tag == "s{}".format(i):
                        senti.append(word)
                theme = "".join(theme)
                senti = "".join(senti)
                if senti and theme:
                    themes.append(theme)
                    senti_w.append(senti)
                elif senti and not theme:
                    themes.append("NULL")
                    senti_w.append(senti)
                elif not senti and not theme:
                    continue
                    # print("All NULL")
                elif not senti and theme:
                    print("ERROR!!!! NO SENTI", tags,words, postags)
                    current_ptag_index = postags.index(ptag)
                    for tag2, word2, ptag2 in zip(tags[current_ptag_index:],words[current_ptag_index:], postags[current_ptag_index:]):
                        if tag2 == "o" and ptag2== "a":
                            themes.append(theme)
                            senti_w.append(word2)
                            print("POSTprocess for no senti", word2)
                            break
                else:
                    print("ERROR", tags, words)
        else:
            ## tag都是"o"
            continue
        
    try:
        assert len(themes) == len(senti_w)
    except:
        print(themes, senti_w)
    
#     themes, senti_w = pair_postpreprocess(themes, senti_w)
    
    row["theme"] = ";".join(themes) + ";"
    row["sentiment_word"] = ";".join(senti_w) + ";"
    return row

# stop_words = ["了", "的", "是", '邹', '让', '闪', '迟', '来', '饱', '这', '近', '要', '合', '个', '胖', '不']
# with open("./pre_adv.txt", "r", encoding="gb2312") as f:
    # pre_adv = f.readlines()
    # pre_adv = [_.strip() for _ in pre_adv]

def pair_postpreprocess(row):
    # 在产生答案pair之后，进行一些后处理，比如去重
    themes = row["theme"][:-1].split(";")
    senti_w = row["sentiment_word"][:-1].split(";")
    senti_v = row["sentiment_anls"][:-1].split(";")
    if themes == senti_w:
        print("not found pair. messages from pair_postpreprocess")
        row["theme"] = ""
        row["sentiment_word"] = ""
        row["sentiment_anls"] = ""
        return row
    pairs = []
    for theme, w, v in zip(themes, senti_w, senti_v):
        pairs.append("_".join([theme, w, v]))
    pairs = list(set(pairs))
    new_themes = []
    new_senti_w = []
    new_senti_v = []
    for pair in pairs:
        tmp = pair.split("_")
        new_themes.append(tmp[0])
        new_senti_w.append(tmp[1])
        new_senti_v.append(tmp[2])
    row["theme"] = ";".join(new_themes) + ";"
    row["sentiment_word"] = ";".join(new_senti_w) + ";"
    row["sentiment_anls"] = ";".join(new_senti_v) + ";"
    return row

with open('./data/high_senti.pkl', 'rb') as f:
	high_senti = pickle.load(f)
	
def content_cut(row):
    content = row.iloc[1]

    for word in high_senti:
        ws = word.split(" ")
        if ws[0] in content:
            t = "。".join(ws[1:])
            content = content.replace(ws[0], t)
    return content


import fasttext
senti_value_classifier = fasttext.load_model('./data/senti_value_classify_model.bin', label_prefix='__label__')
result = senti_value_classifier.test('./data/fasttext_senti_classify_supervised_corpus.txt')
def senti_gen_with_fasttext(row):
    global senti_value_classifier
    sentis = row["sentiment_word"]
    themes = row["theme"]
    content = row.iloc[1]
    anls = ""
    if sentis != ";":
        swords = sentis[:-1].split(";")
        twords = themes[:-1].split(";")
        test_X = []
        for t, w in zip(twords, swords):
            if t == "NULL":
                test_X.append("NULL"+ " " + w)
            else:
                test_X.append(t + " " + w)
        value_pred = senti_value_classifier.predict(test_X)
        value_pred = [_[0] for _ in value_pred]
        anls = ";".join(value_pred) + ";"
    else:
        anls = ";"
    row["sentiment_anls"] = anls
    return row
