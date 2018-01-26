
# coding: utf-8

import pandas as pd
import numpy as np
import re
import jieba.posseg
import copy
import sklearn
import scipy.stats
from sklearn.metrics import make_scorer
from sklearn.cross_validation import cross_val_score
from sklearn.grid_search import RandomizedSearchCV
import sklearn_crfsuite
from sklearn_crfsuite import scorers
from sklearn_crfsuite import metrics

import json

from collections import Counter

from tqdm import tqdm

from collections import defaultdict
import jieba
import jieba.analyse
import pickle
import json

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics.pairwise import cosine_similarity

import matplotlib.pyplot as plt
from score import score_v1, score_v2, score_v3, score_v3_with_debug
import fasttext



from row_apply import *


jieba.dt.tmp_dir = "./jiebatmp/"
jieba.load_userdict("./data/semi_dict.txt")



with open("./data/jieba_pairs.pkl", "rb") as f:
    neighbor_pairs = pickle.load(f)
    suggest_pairs = pickle.load(f)
    d = pickle.load(f)



for p in neighbor_pairs:
    jieba.suggest_freq(p, True)
for p in suggest_pairs:
    jieba.suggest_freq(p, True)
for w in d:
    jieba.suggest_freq(w, True)


train_df = pd.read_csv("./data/semi_train.csv", encoding="utf-8", header=None)
train_df = train_df.fillna(";")
train_df.iloc[:,1] = train_df.iloc[:,1].astype("str")
train_df.columns = ["row_id", "content-评论内容", "theme-主题", "sentiment_word-情感关键词", "sentiment_anls-情感正负面"]


print(train_df[train_df["sentiment_word-情感关键词"] == ";"].shape)



train_df.iloc[:, 1] = train_df.apply(content_filter, axis=1)


with open("./data/semi_dict.txt", "r") as f:
    semi_dict = [x.strip() for x in f.readlines()]


semi_dict = []
for themes, sentis in zip(train_df.loc[:, "theme-主题"], train_df.loc[:, "sentiment_word-情感关键词"]):
    semi_dict.extend(themes[:-1].split(";"))
    semi_dict.extend(sentis[:-1].split(";"))



def single_senti(train_df):
    high_senti_new = []
    for i in range(train_df.shape[0]):
        content = copy.deepcopy(train_df.iloc[i, 1])
        themes = copy.deepcopy(train_df.iloc[i, 2][:-1].split(";"))
        senti_w = copy.deepcopy(train_df.iloc[i, 3][:-1].split(";"))
        if themes != senti_w:
            for t1, w1 in zip(themes, senti_w):
                for t2, w2 in zip(themes, senti_w):
                    if w1+w2 in content and w1 not in t2 and t2 not in w1:
                        add_signal = True
                        __ = w1 + w2
                        for w in semi_dict:
                            if __ in w:
                                add_signal = False
                                print(__)
                                break
                        if add_signal:
                            high_senti_new.append(w1+w2+" "+w1+" "+w2)
                    elif w1+w2 in content and (t2 in w1 or w1 in t2 or t2+w2 in content):
                        print(w1+w2, t2+w2)
                    else:
                        continue
    return high_senti_new
high_senti = single_senti(train_df)

high_senti_counter = Counter(high_senti)
high_senti = [x[0] for x in high_senti_counter.most_common() if len(x[0]) >2]
print(len(high_senti))
with open('./data/high_senti.pkl','wb') as f:
    pickle.dump(high_senti,f)
print('high_senti write! done!')

def content_cut(row):
    content = row.iloc[1]
    with open('./data/high_senti.pkl', 'rb') as f:
        high_senti = pickle.load(f)
    for word in high_senti:
        ws = word.split(" ")
        if ws[0] in content:
            t = "。".join(ws[1:])
            content = content.replace(ws[0], t)
    return content


train_df.iloc[:, 1] = train_df.apply(content_cut, axis=1)



train_df = train_df.apply(answer_process, axis=1)


pattern = re.compile(r"[\w\u4e00-\u9fa5]+")
no_chn = re.compile("^[a-zA-Z\d]+$")
punc_pattern = re.compile(r"[^\u4e00-\u9fa5\w\d]+")


train_df = train_df.apply(tokenized_sub_sents, axis=1)



test_df = pd.read_csv("./data/semi_test.csv", header=None)
test_df = test_df.fillna(";")
test_df = test_df.astype("str")
test_df.columns = ["row_id","content"]
test_df.iloc[:, 1] = test_df.apply(content_filter, axis=1)
test_df.iloc[:, 1] = test_df.apply(content_cut, axis=1)
test_df = test_df.apply(tokenized_sub_sents, axis=1)




## 统计主题字、情感字的概率
theme_char_counter = Counter()
senti_char_counter = Counter()
total_char_counter = Counter()
for themes, sentis, sub_sents in zip(train_df.loc[:, "theme-主题"], train_df.loc[:, "sentiment_word-情感关键词"], train_df.loc[:, "sub_sents_tokenized"]):
    sents = []
    for _ in sub_sents:
        sents.extend(_)
    themes = themes[:-1].replace(";", "")
    sentis = sentis[:-1].replace(";", "")
    theme_char_counter.update(list(themes))
    senti_char_counter.update(list(sentis))
    total_char_counter.update(list("".join(sents)))

train_df = train_df.apply(tag_seq_gen_based_char_v2, axis=1)


train_df["sub_sents_tags_based_char"] = train_df.apply(tag_sort, axis=1)


train_df["sub_sents_tags_based_word"] = train_df.apply(tag_seq_gen_based_word_v2, axis=1)

train_df["bies_tags_based_word"] = train_df.apply(tag2bies, axis=1)

train_df["bies_tags_based_char"] = train_df.apply(chartag2bies, axis=1)


def tags_pred2tags_sample(pred, df, target_column="pred_tags", sequence_column="sub_sents_tokenized"):
    if target_column in df.columns:
        print("drop row column")
        df.drop(target_column, axis=1, inplace=True)
    start_pos = 0
    tmp = []
    for i in range(df.shape[0]):
        l = len(df.loc[i, sequence_column])
        tmp.append(copy.deepcopy(pred[start_pos:(l+start_pos)]))
        start_pos += l
    df[target_column] = tmp
    print("predicted tags inserted.")
    return df

def tags_predprob2tags_sample(pred_prob, df, target_column="pred_probs"):
    if target_column in df.columns:
        print("drop row column")
        df.drop(target_column, axis=1, inplace=True)
    start_pos = 0
    tmp = []
    for i in range(df.shape[0]):
        l = len(df.loc[i, "sub_sents_tokenized"])
        tmp.append(copy.deepcopy(pred_prob[start_pos:(l+start_pos)]))
        start_pos += l
    df[target_column] = tmp
    print("predicted tags probability inserted.")
    return df



def bietag2answer(row, tag_column="pred_tags"):    
    pred_tags = copy.deepcopy(row.loc[tag_column])
    sub_sents = row.loc["sub_sents_tokenized"]
    sub_postags = row.loc["sub_sents_postagged"]
    themes = []
    senti_w = []
    
    for tags, words in zip(pred_tags, sub_sents):
        new_tags = " ".join(tags)
        new_words = " ".join(words)
        if "tb te" in " ".join(tags):
            new_words = new_words.replace(words[tags.index("tb")] + " " + words[tags.index("te")], words[tags.index("tb")] + words[tags.index("te")])
            new_tags = new_tags.replace("tb te", "ts")
            assert len(new_words.split(" ")) == len(new_tags.split(" "))
            tags = new_tags.split(" ")
            words = new_words.split(" ")
        assert len(tags) == len(words)
        is_exist = [x for x in tags if len(x)>1]
        if is_exist:
            Second = False
            Third = False
            if "sbse" in "".join(tags):
                _sb = tags.index("sb")
                _se = tags.index("se")
                new_word = words[_sb] + words[_se]
                senti_w.append(new_word)
                tags[_sb] = "o"
                tags[_se] = "o"
                if "ts" in tags:
                    t_ = tags.index("ts")
                    themes.append(words[t_])
                    tags[t_] = "o"
                else:
                    t_ = "None"
                    themes.append("NULL")
            elif "ss" in is_exist:
                s_p1 = tags.index("ss")
                senti_w.append(words[s_p1])
                tags[s_p1] = "o"
                if "ts" in is_exist:
                    t_p1 = tags.index("ts")
                    themes.append(words[t_p1])
                    tags[t_p1] = "o"
                else:
                    t_p1 = "None"
                    themes.append("NULL")
                if "ss" in tags:
                    second = True
                    s_p2 = tags.index("ss")
                    senti_w.append(words[s_p2])
                    tags[s_p2] = "o"
                    if "ts" in tags:
                        t_p2 = tags.index("ts")
                        themes.append(words[t_p2])
                        tags[t_p2] = "o"
                    else:
                        t_p2 = "None"
                        themes.append("NULL")
                if "ss" in tags:
                    Third = True
                    s_p3 = tags.index("ss")
                    senti_w.append(words[s_p3])
                    tags[s_p3] = "o"
                    if "ts" in tags:
                        t_p3 = tags.index("ts")
                        themes.append(words[t_p3])
                        tags[t_p3] = "o"
                    else:
                        t_p3 = "None"
                        themes.append("NULL")
                if "ss" in tags:
                    s_ = tags.index("ss")
                    senti_w.append(words[s_])
                    tags[s_] = "o"
                    if "ts" in tags:
                        t_ = tags.index("ts")
                        themes.append(words[t_])
                        tags[t_] = "o"
                    else:
                        t_ = "None"
                        themes.append("NULL")
                    
                if "s" in "".join(tags):
                    if Third:
                        print("after extracted:", t_p1, s_p1, t_p2, s_p2, t_p3, s_p3,tags, words)
                    elif Second:
                        print("after extracted:", t_p1, s_p1, t_p2, s_p2,tags, words)
                    else:
                        print("after extracted:", t_p1, s_p1, tags, words)
                else:
                    continue
            elif "ts" in is_exist:
                _ = classifier.predict_proba(words ,k=2)
                r = []
                for __ in _:
                    r.extend(__)
                r = [x[1] for x in r if x[0]=="1"]
                p__ = np.array(r).argmax()
                if tags[p__] != "ts":
                    themes.append(words[tags.index("ts")])
                    tags[tags.index("ts")] = "o"
                    senti_w.append(words[p__])
                if "s" in "".join(tags):
                    print("ts not extracted", tags, words)
            else:
                print("No extracted", tags, words)
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



## 词向量数据集和判断是否是情感词的fasttext训练文件

past_train_df = pd.read_excel("./data/泰一指尚训练集.xlsx", encoding="gbk")
past_train_df.iloc[:,1] = past_train_df.iloc[:,1].astype("str")
past_train_df = past_train_df.fillna("")
past_test_df = pd.read_csv("./data/泰一指尚-评测集.csv", encoding="gbk")
past_test_df = past_test_df.astype("str")
tmp = list(past_test_df.iloc[:,1])
test = [past_test_df.columns[1]]
test.extend(tmp)
past_test_df = pd.DataFrame(test)
past_test_df["row_id"] = range(1,past_test_df.shape[0]+1)
past_test_df.columns = ["content", "row_id"]
past_test_df = past_test_df.loc[: , ["row_id","content"]]

past_train_df.iloc[:, 1] = past_train_df.apply(content_filter, axis=1)
past_train_df.iloc[:, 1] = past_train_df.apply(content_cut, axis=1)
past_train_df = past_train_df.apply(tokenized_sub_sents, axis=1)

past_test_df.iloc[:, 1] = past_test_df.apply(content_filter, axis=1)
past_test_df.iloc[:, 1] = past_test_df.apply(content_cut, axis=1)
past_test_df = past_test_df.apply(tokenized_sub_sents, axis=1)

for_fast = []
for _ in train_df.sub_sents_tokenized:
    for_fast.extend(_)
for _ in test_df.sub_sents_tokenized:
    for_fast.extend(_)
for _ in past_train_df.sub_sents_tokenized:
    for_fast.extend(_)
for _ in past_test_df.sub_sents_tokenized:
    for_fast.extend(_)
with open("./data/fast_text_corpus.txt", "w") as f:
    for line in for_fast:
        f.writelines(" ".join(line)+"\n")

fast_themes = []
fast_sentis = []
for _ in train_df.loc[:, "theme-主题"]:
    fast_themes.extend(_[:-1].split(";"))
for _ in past_train_df.loc[:, "theme-主题"]:
    fast_themes.extend(_[:-1].split(";"))    
fast_themes = [(_, 0) for _ in fast_themes if _ != "NULL"]

for _ in train_df.loc[:, "sentiment_word-情感关键词"]:
    fast_sentis.extend(_[:-1].split(";"))
for _ in past_train_df.loc[:, "sentiment_word-情感关键词"]:
    fast_sentis.extend(_[:-1].split(";"))    
fast_sentis = [(_, 1) for _ in fast_sentis]


fast_ = []
fast_.extend(fast_themes)
fast_.extend(fast_sentis)
random_index = np.random.permutation(len(fast_)).tolist()
with open("./data/fast_supervised_corpus.txt", "w") as f:
    for _ in random_index:
        f.writelines(fast_[_][0] + " __label__{}\n".format(fast_[_][1]))



# model = fasttext.skipgram("./fast_text_corpus.txt", "./data/skip_model_50", dim=50)
# model = fasttext.cbow("./fast_text_corpus.txt", "./data/cbow_model_100", dim=100)
skip_model_50 = fasttext.load_model("./data/skip_model_50_v2.bin")
# 判断是否是情感词
classifier = fasttext.supervised('./data/fast_supervised_corpus.txt', './data/classify_model', label_prefix='__label__', pretrained_vectors="./data/cbow_model_100.vec")
result = classifier.test('./data/fast_supervised_corpus.txt')
# Properties
print(result.precision) # Precision at one
print(result.recall)    # Recall at one
print(result.nexamples) # Number of test examples



def fasttext_senti_supervised(train_df):
    # 分类情感词的极性
    train_X = []
    train_y = []
    for i in range(train_df.shape[0]):
        content = train_df.iloc[i,1]
        senti_v = train_df.iloc[i,4][:-1].split(";")
        senti_word = train_df.iloc[i,3][:-1].split(";")
        theme_word = train_df.iloc[i,2][:-1].split(";")
        content = train_df.iloc[i,1]
        if senti_v == senti_word:
            continue
        for t, w, v in zip(theme_word, senti_word, senti_v):
            if t == "NULL":
                train_X.append("NULL"+ " " + w)
                train_y.append(v)
            else:
                train_X.append(t + " " + w)
                train_y.append(v)
    assert len(train_X) == len(train_y)
    with open("./data/fasttext_senti_classify_supervised_corpus.txt", "w") as f:
        for x, y in zip(train_X, train_y):
            f.writelines(x + " " + "__label__{}".format(y) + "\n")
    return True
# fasttext_senti_supervised(train_df)
senti_value_classifier = fasttext.supervised('./data/fasttext_senti_classify_supervised_corpus.txt', './data/senti_value_classify_model_', dim=50,label_prefix='__label__', pretrained_vectors="./data/skip_model_50.vec")
# SKIP训练出来的vec似乎会让分类器效果的方差大一点，也就是有可能更好，也有可能更差，但是总的来说变化都不大
result = senti_value_classifier.test('./data/fasttext_senti_classify_supervised_corpus.txt')
# Properties
print(result.precision) # Precision at one
print(result.recall)    # Recall at one
print(result.nexamples) # Number of test examples

senti_value_classifier = fasttext.load_model('./data/senti_value_classify_model.bin', label_prefix='__label__')
result = senti_value_classifier.test('./data/fasttext_senti_classify_supervised_corpus.txt')
# Properties
print(result.precision) # Precision at one
print(result.recall)    # Recall at one
print(result.nexamples) # Number of test examples


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



train_df["sub_sents_features"] = train_df.apply(features_gen, axis=1)


def char2features(sent, i):
    char = sent[i]
    l = len(sent)
    is_senti_prob_1 = dict(classifier.predict_proba("去" ,k=2)[0])["1"]
    is_senti_prob_2 = 0 if total_char_counter[char]==0 else senti_char_counter[char]/total_char_counter[char]
    is_theme_prob = 0 if total_char_counter[char]==0 else theme_char_counter[char]/total_char_counter[char]
    features = {
        "word_location": i,
        "char": char,
        "sent_length": l,
        "is_senti_prob_1": is_senti_prob_1,
        "is_senti_prob_2": is_senti_prob_2,
        "is_theme_char_prob": is_theme_prob
    }
        
    if i > 0:
        char1eft = sent[i-1]
        left_de = True if char1eft == "的" else False
        c_sim = cosine_similarity(np.array(skip_model_50[char1eft]).reshape(1, -1), np.array(skip_model_50[char]).reshape(1, -1))[0,0]
        features.update({
            "left_is_de": left_de,
            "left_char":char1eft,
            "2gram_left": char1eft + "|" + char,
#             "left_char_sim": c_sim
        })
    else:
        features['BOS'] = True

    if i < len(sent)-1:
        charright = sent[i+1]
        right_de = True if charright == "的" else False
        features.update({
            "right_is_de": right_de,
            "rightchar":charright,
            "2gram_right": char + "|" + charright
        })
    else:
        features['EOS'] = True
    
    if i > 1:
        charleft2 = sent[i-2]
        features.update({
            "left_char2": charleft2,
            "3gram_left": charleft2 + "|" + char1eft + "|" + char
        })
    
    if i < len(sent)-2:
        charright2 = sent[i+2]
        features.update({
            "rightchar2":charright2,
            "3gram_right": char + "|" + charright + "|" + charright2
        })
    
    
    features["num_right"] = l-i-1
    
    return features

def sent2features_for_char(sent):
    sent = "".join(sent)
    sent = list(sent)
    return [char2features(sent, i) for i in range(len(sent))]

def features_gen_for_char(row):
    sub_sents_features_for_char = []
    for sub_sents in row.loc["sub_sents_tokenized"]:
        sub_sents_features_for_char.append(sent2features_for_char(sub_sents))
    return sub_sents_features_for_char



train_df["sub_sents_features_for_char"] = train_df.apply(features_gen_for_char, axis=1)



test_df["sub_sents_features"] = test_df.apply(features_gen, axis=1)
test_df["sub_sents_features_for_char"] = ""
test_df.loc[:, "sub_sents_features_for_char"] = test_df.apply(features_gen_for_char, axis=1)




senti_words = set()
for sentis in train_df.loc[:, "sentiment_word-情感关键词"]:
    if sentis != ";":
        senti_words.update(sentis[:-1].split(";"))
theme_words = set()
for themes in train_df.loc[:, "theme-主题"]:
    if themes != ";":
        theme_words.update(themes[:-1].split(";"))
        
senti_sent_dict = defaultdict(set)
theme_sent_dict = defaultdict(set)
senti_theme_dict = defaultdict(list)
theme_senti_dict = defaultdict(list)

for sents, sentis, themes in zip(train_df.loc[:, "sub_sents_tokenized"], train_df.loc[:, "sentiment_word-情感关键词"], train_df.loc[:, "theme-主题"]):
	if sentis:
		for w, t in zip(sentis[:-1].split(";"), themes[:-1].split(";")):
			for s in sents:
				s = "".join(s)
				if w in s:
					senti_sent_dict[w].add(s)
					senti_theme_dict[w].append(t)
				if t in s:
					theme_sent_dict[t].add(s)
					theme_senti_dict[t].append(w)

sent_theme_dict = defaultdict(set)
sent_senti_dict = defaultdict(set)

for sents, sentis, themes in zip(train_df.loc[:, "sub_sents_tokenized"], train_df.loc[:, "sentiment_word-情感关键词"], train_df.loc[:, "theme-主题"]):
	if themes != sentis:
		for w in themes[:-1].split(";"):
			for s in sents:
				s = "".join(s)
				if w in s:
					sent_theme_dict[s].add(w)
		for w in sentis[:-1].split(";"):
			for s in sents:
				s = "".join(s)
				if w in s:
					sent_senti_dict[s].add(w)



c_q = []
for i in range(test_df.shape[0]):
	for q in test_df.iloc[i, 2]:
		x = set(q)
		q = "".join(q)
		x = set(list(jieba.cut_for_search(q)))
		x = x.union(set(jieba.lcut(q, cut_all=True)))
		inter = x.intersection(senti_words)
		if len(inter) == 1 and len(x.intersection(theme_words)) == 0:
			inter = list(inter)[0]
			for s in senti_sent_dict[inter]:
				if len(q) > len(inter):
					if s in sent_theme_dict and "NULL" not in senti_theme_dict[inter]:
						if [y for x, y in list(jieba.posseg.cut(s))] == [y for x, y in list(jieba.posseg.cut(q))]:
							print("查询句: ", q)
							print("中间情感词：", inter)
							print("目标带主题的句子:", s, sent_theme_dict[s])
							print("\n")
							c_q.append(q + " " + inter + " " + s + " " + list(sent_theme_dict[s])[0])
							break

c_q_counter = Counter(c_q)


candidate_themes_semi_test = []
for w, v in c_q_counter.most_common():
    _ = w.split(" ")
    q = jieba.lcut(_[0])
    s = _[1]
    r = jieba.lcut(_[2])
    t = _[3]
    q1 = copy.deepcopy(q)
    try:
        q1.remove(s)
    except ValueError:
        for i, j in zip(_[0].replace(s, " ").split(" "), _[2].replace(s, " ").split(" ")):
            if j == t:
                print(i, w)
                candidate_themes_semi_test.append(i)
        continue
    if len(q1) == 1:
        print(q1[0], w)
        candidate_themes_semi_test.append(q1[0])
    else:
        print(q[r.index(t)], w)
        candidate_themes_semi_test.append(q[r.index(t)])

with open("./data/candidate_themes_semi_test.json", "w") as f:
    json.dump(fp=f, obj=candidate_themes_semi_test)


# 用于输出特征 for chandi
print('feature extract!!')
past_train_df = pd.read_excel("./data/泰一指尚训练集.xlsx", encoding="gbk")
past_train_df.iloc[:,1] = past_train_df.iloc[:,1].astype("str")
past_train_df = past_train_df.fillna("")
past_test_df = pd.read_csv("./data/泰一指尚-评测集.csv", encoding="gbk")
past_test_df = past_test_df.astype("str")
tmp = list(past_test_df.iloc[:,1])
test = [past_test_df.columns[1]]
test.extend(tmp)
past_test_df = pd.DataFrame(test)
past_test_df["row_id"] = range(1,past_test_df.shape[0]+1)
past_test_df.columns = ["content", "row_id"]
past_test_df = past_test_df.loc[:, ["row_id", "content"]]


past_train_df.iloc[:, 1] = past_train_df.apply(content_filter, axis=1)
past_train_df.iloc[:, 1] = past_train_df.apply(content_cut, axis=1)
past_train_df = past_train_df.apply(tokenized_sub_sents, axis=1)

past_test_df.iloc[:, 1] = past_test_df.apply(content_filter, axis=1)
past_test_df.iloc[:, 1] = past_test_df.apply(content_cut, axis=1)
past_test_df = past_test_df.apply(tokenized_sub_sents, axis=1)

# bie char
X_train_df = []
y_train_df = []
for i in range(train_df.shape[0]):
    for sub_feature, sub_tags in zip(train_df.loc[i, "sub_sents_features_for_char"], train_df.loc[i, "bies_tags_based_char"]):
        X_train_df.append(sub_feature)
        y_train_df.append(sub_tags)

crf = sklearn_crfsuite.CRF(
    algorithm='lbfgs',
    c1=0.5,
    c2=2,
    max_iterations=200,
    all_possible_transitions=True
)

crf.fit(X_train_df, y_train_df)
pred_prob = crf.predict_marginals(X_train_df)
train_df_for_chandi = tags_predprob2tags_sample(df=train_df, pred_prob=pred_prob, target_column="bie_prob_char")

X_test = []

for i in range(test_df.shape[0]):
    for sub_feature in test_df.loc[i, "sub_sents_features_for_char"]:
        X_test.append(sub_feature)
        
test_pred_prob = crf.predict_marginals(X_test)
test_df_for_chandi = tags_predprob2tags_sample(df=test_df, pred_prob=test_pred_prob, target_column="bie_prob_char")

"""tsn char"""
X_train_df = []
y_train_df = []
for i in range(train_df.shape[0]):
    for sub_feature, sub_tags in zip(train_df.loc[i, "sub_sents_features_for_char"], train_df.loc[i, "sub_sents_tags_based_char"]):
        X_train_df.append(sub_feature)
        y_train_df.append(sub_tags)

crf = sklearn_crfsuite.CRF(
    algorithm='lbfgs',
    c1=0.5,
    c2=2,
    max_iterations=200,
    all_possible_transitions=True
)

crf.fit(X_train_df, y_train_df)
pred_prob = crf.predict_marginals(X_train_df)
train_df_for_chandi = tags_predprob2tags_sample(df=train_df_for_chandi, pred_prob=pred_prob, target_column="tsn_prob_char")

        
test_pred_prob = crf.predict_marginals(X_test)
test_df_for_chandi = tags_predprob2tags_sample(df=test_df_for_chandi, pred_prob=test_pred_prob, target_column="tsn_prob_char")



# bie word
X_train_df = []
y_train_df = []
for i in range(train_df.shape[0]):
    for sub_feature, sub_tags in zip(train_df.loc[i, "sub_sents_features"], train_df.loc[i, "bies_tags_based_word"]):
        X_train_df.append(sub_feature)
        y_train_df.append(sub_tags)

crf = sklearn_crfsuite.CRF(
    algorithm='lbfgs',
    c1=1,
    c2=5,
    max_iterations=200,
    all_possible_transitions=True
)

crf.fit(X_train_df, y_train_df)
pred_prob = crf.predict_marginals(X_train_df)
train_df_for_chandi = tags_predprob2tags_sample(df=train_df_for_chandi, pred_prob=pred_prob, target_column="bie_prob_word")

X_test = []

for i in range(test_df.shape[0]):
    for sub_feature in test_df.loc[i, "sub_sents_features"]:
        X_test.append(sub_feature)
        
test_pred_prob = crf.predict_marginals(X_test)
test_df_for_chandi = tags_predprob2tags_sample(df=test_df_for_chandi, pred_prob=test_pred_prob, target_column="bie_prob_word")



"""tsn word"""
X_train_df = []
y_train_df = []
for i in range(train_df.shape[0]):
    for sub_feature, sub_tags in zip(train_df.loc[i, "sub_sents_features"], train_df.loc[i, "sub_sents_tags_based_word"]):
        X_train_df.append(sub_feature)
        y_train_df.append(sub_tags)

crf = sklearn_crfsuite.CRF(
    algorithm='lbfgs',
    c1=1,
    c2=5,
    max_iterations=100,
    all_possible_transitions=True
)

crf.fit(X_train_df, y_train_df)
pred_prob = crf.predict_marginals(X_train_df)
train_df_for_chandi = tags_predprob2tags_sample(df=train_df_for_chandi, pred_prob=pred_prob, target_column="tsn_prob_word")

        
test_pred_prob = crf.predict_marginals(X_test)
test_df_for_chandi = tags_predprob2tags_sample(df=test_df_for_chandi, pred_prob=test_pred_prob, target_column="tsn_prob_word")

with open("./chandi/data/features_chandi_1210.pkl", "wb") as f:
    pickle.dump(file=f, obj=past_train_df.loc[:, ["row_id", "sub_sents_tokenized"]], protocol=0)
    pickle.dump(file=f, obj=past_test_df.loc[:, ["row_id", "sub_sents_tokenized"]], protocol=0)
    pickle.dump(file=f, obj=train_df_for_chandi.loc[:, ["row_id", "sub_sents_tokenized", "tsn_prob_word", "bie_prob_word", "tsn_prob_char", "bie_prob_char"]], protocol=0)
    pickle.dump(file=f, obj=test_df_for_chandi.loc[:, ["row_id", "sub_sents_tokenized", "tsn_prob_word", "bie_prob_word", "tsn_prob_char", "bie_prob_char"]], protocol=0)

print('feature write done!!')