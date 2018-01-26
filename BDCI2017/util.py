# coding=utf-8

import pandas as pd
import re
import collections
import codecs

def getPreTrainDF():
    df = pd.read_excel('./data/预赛训练集.xlsx')
    df.columns = ['id', 'content', 'theme', 'keyword', 'polarity']
    return df

def getPreTestDF():
    df = pd.read_csv('./data/预赛评测集.csv', header=None)
    df.columns = ['id', 'content']
    return df

def getSemiTrainDF():
    df = pd.read_csv('./data/复赛训练集.csv', header=None)
    df.columns = ['id', 'content', 'theme', 'keyword', 'polarity']
    return df

def getSemiTestDF():
    df = pd.read_csv('./data/复赛评测集.csv', header=None)
    df.columns = ['id', 'content']
    return df

def cleanContent(content):
    pattern = u'[^\u4e00-\u9fa5a-z]+'
    content = unicode(content).lower()
    content = re.sub(pattern, u' ', content)
    content = re.sub(u'\s+', u'，', content)
    textList = [text for text in content.split(u'，') if len(text) > 0]
    return textList

def getTruthWordList():

    themeList = []
    kywrdList = []

    semiTrainDF = getSemiTrainDF()
    contentList = semiTrainDF['content'].values
    for index, content in enumerate(contentList):
        if type(semiTrainDF.iloc[index]['theme']) != float:
            tmp = [word.decode('utf-8') for word in semiTrainDF.iloc[index]['theme'].split(';')[:-1]]
            tmp = [word.strip().lower() for word in tmp]
            themeList.extend(tmp)
        if type(semiTrainDF.iloc[index]['keyword']) != float:
            tmp = [word.decode('utf-8') for word in semiTrainDF.iloc[index]['keyword'].split(';')[:-1]]
            tmp = [word.strip().lower() for word in tmp]
            kywrdList.extend(tmp)

    themeList = collections.Counter(themeList).most_common()
    with codecs.open('./data/themeTruthList-semiTrain-1206.txt', 'w', 'utf-8') as fw:
        for (theme, count) in themeList:
            fw.write(theme + ' ' + str(count) + '\n')

    kywrdList = collections.Counter(kywrdList).most_common()
    with codecs.open('./data/kywrdTruthList-semiTrain-1206.txt', 'w', 'utf-8') as fw:
        for (kywrd, count) in kywrdList:
            fw.write(kywrd + ' ' + str(count) + '\n')

def readWordList(fileName):
    wordCount = []
    with codecs.open(fileName, 'r', 'utf-8') as fr:
        lines = fr.readlines()
        for line in lines:
            line = line.strip().split(u' ')
            word = line[0]
            count = line[1]
            wordCount.append((word, count))
    return wordCount

def lcs_str(x, y):
    def LCS_length(x, y):
        xlen = len(x)
        ylen = len(y)
        tmp_1 = [[0 for i in range(ylen + 1)] for j in range(2)]
        tmp_2 = [[0 for i in range(ylen + 1)] for j in range(xlen + 1)]
        for i in range(1, xlen + 1):
            for j in range(1, ylen + 1):
                if x[i - 1] == y[j - 1]:
                    tmp_1[i % 2][j] = tmp_1[(i - 1) % 2][j - 1] + 1
                    tmp_2[i][j] = 0
                else:
                    if tmp_1[i % 2][j - 1] >= tmp_1[(i - 1) % 2][j]:
                        tmp_1[i % 2][j] = tmp_1[i % 2][j - 1]
                        tmp_2[i][j] = 1
                    else:
                        tmp_1[i % 2][j] = tmp_1[(i - 1) % 2][j]
                        tmp_2[i][j] = -1
        return tmp_1, tmp_2

    def LCS_print(x, y, tmp_2):
        result = []
        i = len(x)
        j = len(y)
        k = 0
        while (i > 0 and j > 0):
            if (tmp_2[i][j] == 0):
                result.append(x[i - 1])
                k = k + 1
                i = i - 1
                j = j - 1
            elif tmp_2[i][j] == 1:
                j = j - 1
            elif tmp_2[i][j] == -1:
                i = i - 1
        return result

    _, tmp_2 = LCS_length(x, y)
    result = LCS_print(x, y, tmp_2)
    r_len=len(result)
    out=[result[r_len-1-i] for i in range(r_len)]
    return ''.join(out)

import os
import pyltp

def ltpSetup():
    LTP_DATA_DIR = './ltp_data_v3.4.0/'
    cws_model_path = os.path.join(LTP_DATA_DIR, 'cws.model')
    pos_model_path = os.path.join(LTP_DATA_DIR, 'pos.model')
    segmentor = pyltp.Segmentor()
    segmentor.load(cws_model_path)
    postagger = pyltp.Postagger()
    postagger.load(pos_model_path)
    return segmentor, postagger

def wordSeg(segmentor, sentence):
    sentence = sentence.encode('utf-8')
    wordList = segmentor.segment(sentence)
    return wordList

def wordPos(postagger, wordList):
    posList = postagger.postag(wordList)
    return posList

def ltpRelease(segmentor, postagger):
    segmentor.release()
    postagger.release()

if __name__ == '__main__':
    getTruthWordList()