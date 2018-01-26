import numpy as np
import re
import pandas as pd
import itertools
from collections import Counter
import codecs
import random
import pickle
# 导入数据
def load_train_datalabels(train_file):

    sent_data = []
    label_data = []
    train_data = []
    with codecs.open(train_file,encoding='utf-8')as f:
        for line in f:
            train_data.append(line.strip().split('\t'))

    for ind,it in enumerate(train_data):
        sent = ' '.join(it[0].replace(' ', ''))
        label = [0]*8
        try:
            label[int(it[1])] = 1
        except:
            print(ind,sent)
        sent_data.append(sent)
        label_data.append(label)
    assert len(sent_data) == len(label_data)
    print('train data number: ', len(sent_data))
    return sent_data,np.array(label_data)

def load_test_data(test_file):
    with codecs.open(test_file, 'rb') as f:
        train_1 = pickle.load(f)
        test_1 = pickle.load(f)
        train_2 = pickle.load(f)
        test_2 = pickle.load(f)
    # print(train_2.columns)
    # print(test_2.columns)
    return train_2, test_2
#
# load_test_data('features_chandi.pkl')
# with codecs.open('train_test.pkl', 'rb') as f:
#     train_1 = pickle.load(f)
#     test_1 = pickle.load(f)

# del test_1['bie_prob']
# del test_1['tsn_prob']
# del train_1['bie_prob']
# del train_1['tsn_prob']
# print(train_1[:2])
# print(test_1[:2])
# with codecs.open('train_test_1.pkl','wb') as f:
#     pickle.dump(train_1,f,0)
#     pickle.dump(test_1,f,0)

# load_train_datalabels('../BDCI2017-taiyi/data4sent_has_ans.txt')
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
    model=gensim.models.Word2Vec.load(wordvectorPath)
    vocab=list(model.wv.vocab.keys())
    vocab.append('UNK')
    vocab2index={}
    word_vector=[]
    vector_size=model.vector_size
    for ind,w in enumerate(vocab[:-1]):
        vocab2index[w]=ind
        word_vector.append(model[w])
    vocab2index['UNK']=len(vocab)-1
    vocab2index['PAD']=len(vocab)
    vocab.append('PAD')
    word_vector.append(np.random.uniform(-1.0, 1.0,vector_size))
    word_vector.append(np.array([0] * vector_size))
    print("finished loaded word2vec!!")
    return vocab,vocab2index,np.array(word_vector)

def sent2index(sentence,max_sentence_length,vocab2index):
    word_list=sentence.split()
    index_list=[]
    for w in word_list[:max_sentence_length]:
        try:
            index_list.append(vocab2index[w])
        except:
            index_list.append(vocab2index['UNK'])
    if len(word_list)<max_sentence_length:
        for i in range(max_sentence_length-len(word_list)):
            index_list.append(vocab2index['PAD'])
    return index_list

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
    vocab=list(word2vec.keys())
    vocab.append('UNK')
    vocab2index = {}
    word_vector = []
    vector_size = len(word2vec[vocab[0]])
    for ind,w in enumerate(vocab[:-1]):
        vocab2index[w]=ind
        word_vector.append(word2vec[w])
    # print(np.array(word_vector).shape)
    vocab2index['UNK']=len(vocab)-1
    vocab2index['PAD']=len(vocab)
    vocab.append('PAD')
    word_vector.append(np.random.uniform(-1.0,1.0,vector_size))
    word_vector.append([0.1] * vector_size)
    print("finished loaded word2vec!!")
    return vocab,vocab2index,np.array(word_vector)

if __name__ == "__main__":
    pass
    train_file = '../BDCI2017-taiyi/data4model.csv'
    wordvector_path='../w2v/fasttext_char_vec/fasttext_cbow_char.model.vec'
    word2vec_path='../w2v/word2vec_review.model'
    # sent_data, pair_data, label_data = load_train_datalabels(train_file)
    # print([it for it in zip(sent_data[:10],pair_data[:10],label_data[:10])])
    # vocab, vocab2index,word_vector=get_fasttext(wordvector_path)
    # vocab, vocab2index,word_vector=load_wordvector(word2vec_path)
    # print(word_vector.shape)
    # print(type(word_vector))

