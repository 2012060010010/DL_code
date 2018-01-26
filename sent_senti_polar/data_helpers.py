import numpy as np
# import re
import pandas as pd
# import itertools
# from collections import Counter
import codecs
import pickle
import re
import jieba
jieba.set_dictionary('../BDCI2017-taiyi/word4seg1.txt')
with codecs.open("../BDCI2017-taiyi/theme.txt", encoding='utf-8') as f:
    theme_list = [line.strip() for line in f]

with codecs.open("../BDCI2017-taiyi/sentiment.txt", encoding='utf-8') as f:
    sentiment_list = [line.strip() for line in f]

theme_set = set(theme_list)
sentiment_set = set(sentiment_list)

# import random
punc_pattern = re.compile(r"[^\u4e00-\u9fa5\w\d]+")
from pyltp import SentenceSplitter
def tokenized_sub_sents(content):
    sub_sents = []
    sents = SentenceSplitter.split(content)
    for sent in sents:
        subs = [x for x in re.split(punc_pattern, sent) if x]
        sub_sents.extend(subs)
    return sub_sents

def make_test_data(row):
    data4model = []
    # content = row['content_t']
    content = row
    sent_tmp = tokenized_sub_sents(content)
    for sent in sent_tmp:
        word_seg = jieba.lcut(sent.lower())
        sent = ' '.join(word_seg)
        candidate_theme = set(word_seg) & theme_set
        candidate_theme.add('NULL')
        candidate_sentiment = set(word_seg) & sentiment_set
        sample_tmp = []
        if len(candidate_sentiment) > 0:
            pair_tmp = set()
            for t in candidate_theme:
                for s in candidate_sentiment:
                    if t != s:
                        pair_tmp.add((t, s))
            for it in pair_tmp:
                sample_tmp.append([sent, it[0], it[1]])
        data4model.extend(sample_tmp)
    return data4model
# 导入数据
def load_train_datalabels(train_file, max_seq_length):
    train_df = pd.read_csv(train_file)
    print('train data number: ', train_df.shape[0])
    sent_data = []
    position_data = []
    pair_data = []
    label_data = []
    polar_data = []
    for i in range(train_df.shape[0]):
    # for i in range(10000):
        sent = ' '.join(train_df.loc[i, 'sent'].replace(' ', ''))
        sub_sent = train_df.loc[i, 'sent'].replace(' ', '')
        theme = str(train_df.loc[i, 'theme'])
        if theme == 'nan':
            theme_l = 'PAD'
            theme_s = 0
            theme_e = 0
        else:
            theme_l = ' '.join(theme)
            theme_s = sub_sent.index(theme)
            theme_e = theme_s + len(theme) - 1
        theme_pos = []
        for ind in range(min(len(sub_sent),max_seq_length)):
            if ind < theme_s:
                theme_pos.append(ind - theme_s)
            if ind >= theme_s and ind <= theme_e:
                theme_pos.append(0)
            if ind > theme_e:
                theme_pos.append(ind - theme_e)

        sentiment = train_df.loc[i, 'sentiment']
        sentiment_l = ' '.join(sentiment)
        senti_s = sub_sent.index(sentiment)
        senti_e = senti_s + len(sentiment) - 1
        # print(sub_sent, sentiment, senti_s, senti_e)
        senti_pos = []
        for ind in range(min(len(sub_sent),max_seq_length)):
            if ind < senti_s:
                senti_pos.append(ind - senti_s)
            if ind >= senti_s and ind <= senti_e:
                senti_pos.append(0)
            if ind > senti_e:
                senti_pos.append(ind - senti_e)
        position_data.append([[it[0], it[1]] for it in zip(theme_pos, senti_pos)])
        label = int(train_df.loc[i, 'ans'])
        polar = int(train_df.loc[i, 'polar'])

        pair = [theme_l, sentiment_l]
        sent_data.append(sent)
        pair_data.append(pair)
        if label == 1:
            label_data.append([0, 1])
        else:
            label_data.append([1, 0])
        p = [0] * 4
        p[polar + 1] = 1
        polar_data.append(p)
    assert len(sent_data) == len(pair_data)
    return sent_data, position_data, pair_data, np.array(label_data), np.array(polar_data)

def load_test_data(test_file):
    with codecs.open(test_file, 'rb') as f:
        # train_1 = pickle.load(f)
        # test_1 = pickle.load(f)
        train_2 = pickle.load(f)
        test_2 = pickle.load(f)
    print(train_2.columns)
    print(test_2.columns)
    return train_2, test_2


def batch_iter(data, batch_size, shuffle=True):
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_data = data[shuffle_indices]
    else:
        shuffled_data = data
    for batch_num in range(num_batches_per_epoch):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        yield shuffled_data[start_index:end_index]


# 导入初始词向量
def load_wordvector(wordvectorPath):
    import gensim
    model = gensim.models.Word2Vec.load(wordvectorPath)
    vocab = list(model.wv.vocab.keys())
    vocab.append('UNK')
    vocab2index = {}
    word_vector = []
    vector_size = model.vector_size
    for ind, w in enumerate(vocab[:-1]):
        vocab2index[w] = ind
        word_vector.append(model[w])
    vocab2index['UNK'] = len(vocab)-1
    vocab2index['PAD'] = len(vocab)
    vocab.append('PAD')
    word_vector.append(np.random.uniform(-1.0, 1.0, vector_size))
    word_vector.append(np.array([0] * vector_size))
    print("finished loaded word2vec!!")
    return vocab,  vocab2index, np.array(word_vector)


def sent2index(sentence, max_sentence_length, vocab2index):
    word_list = sentence.split()
    index_list = []
    for w in word_list[:max_sentence_length]:
        try:
            index_list.append(vocab2index[w])
        except KeyError:
            index_list.append(vocab2index['UNK'])
    if len(word_list) < max_sentence_length:
        for i in range(max_sentence_length-len(word_list)):
            index_list.append(vocab2index['PAD'])
    return index_list

def padding_pos(pos_list, max_seq_length):
    if len(pos_list) >= max_seq_length:
        return np.array(pos_list)
    else:
        for i in range(max_seq_length - len(pos_list)):
            pos_list.append([0, 0])
        return np.array(pos_list)

def get_fasttext(vecfile):
    with codecs.open(vecfile,encoding='utf-8') as f:
        word2vec={}
        lines = f.readlines()
        vector_size = int(lines[0].strip().split()[1])
        for line in lines[1:]:
            item = line.strip().split()
            word = item[0]
            vec = np.array([float(it) for it in item[1:]])
            if len(vec) == vector_size:
                word2vec[word] = vec
    vocab = list(word2vec.keys())
    vocab.append('UNK')
    vocab2index = {}
    word_vector = []
    vector_size = len(word2vec[vocab[0]])
    for ind, w in enumerate(vocab[:-1]):
        vocab2index[w] = ind
        word_vector.append(word2vec[w])
    # print(np.array(word_vector).shape)
    vocab2index['UNK'] = len(vocab)-1
    vocab2index['PAD'] = len(vocab)
    vocab.append('PAD')
    word_vector.append(np.random.uniform(-1.0, 1.0, vector_size))
    word_vector.append([0.1] * vector_size)
    print("finished loaded word2vec!!")
    return vocab, vocab2index, np.array(word_vector)

if __name__ == "__main__":
    pass
    train_file = '../BDCI2017-taiyi/data4model.csv'
    wordvector_path = '../w2v/fasttext_char_vec/fasttext_cbow_char.model.vec'
    word2vec_path = '../w2v/word2vec_review.model'
    # sent_data, pair_data, label_data = load_train_datalabels(train_file)
    # print([it for it in zip(sent_data[:10],pair_data[:10],label_data[:10])])
    vocab, vocab2index, word_vector = get_fasttext(wordvector_path)
    # vocab, vocab2index,word_vector=load_wordvector(word2vec_path)
    # print(word_vector.shape)
    # print(type(word_vector))

