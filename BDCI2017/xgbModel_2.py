# coding=utf-8

import numpy as np
import re
import gensim
import collections
import util
import codecs
import fasttext
from textrank4zh import TextRank4Keyword
from sklearn.model_selection import train_test_split
import sklearn
import xgboost as xgb
import time
from multiprocessing import Pool
import pickle
from matplotlib import pyplot as plt
import Levenshtein

with open('./data/features_chandi_1210.pkl', 'rb') as fr:
    preTrainDF = pickle.load(fr)
    preTestDF = pickle.load(fr)
    semiTrainDF = pickle.load(fr)
    semiTestDF = pickle.load(fr)

def getTextList(preTrainDF, preTestDF, semiTrainDF, semiTestDF):

    contentList = []
    for df in [preTrainDF, preTestDF, semiTrainDF, semiTestDF]:
        cL = df['sub_sents_tokenized'].values
        for content in cL:
            for text in content:
                contentList.append(u''.join(text))

    textList = []
    for content in contentList:
        t = u' '.join(list(content))
        textList.append(t)
    return textList

def buildGensimModel(textList):
    documents = []
    for text in textList:
        documents.append(text)
    texts = [[word for word in document.split()] for document in documents]

    dictionary = gensim.corpora.Dictionary(texts)
    dictionary.save('./dic&corp/bdci.dict')
    print 'Finished saving dictionary.'

    corpus = [dictionary.doc2bow(text) for text in texts]
    gensim.corpora.MmCorpus.serialize('./dic&corp/bdci.mm', corpus)
    print 'Finished saving corpus.'

    tfidf_model = gensim.models.TfidfModel(corpus)
    tfidf_model.save('./dic&corp/model.tfidf')
    print 'Finished saving tfidf_model.'

    corpus_tfidf = tfidf_model[corpus]

    lsi_model = gensim.models.LsiModel(corpus_tfidf, id2word=dictionary, num_topics=3)
    lsi_model.save('./dic&corp/model.lsi')
    print 'Finished saving lsi_model.'

    lda_model = gensim.models.LdaModel(corpus_tfidf, id2word=dictionary, num_topics=3)
    lda_model.save('./dic&corp/model.lda')
    print 'Finished saving lda_model.'

def getCorpusDictionary():
    corpus = gensim.corpora.MmCorpus('./dic&corp/bdci.mm')
    dictionary = gensim.corpora.Dictionary.load('./dic&corp/bdci.dict')
    return corpus, dictionary

def getTfIdf_model():
    tfidf_model = gensim.models.TfidfModel.load('./dic&corp/model.tfidf')
    return tfidf_model

def getLsi_model():
    lsi_model = gensim.models.LsiModel.load('./dic&corp/model.lsi')
    return lsi_model

def getLda_model():
    lda_model = gensim.models.LdaModel.load('./dic&corp/model.lda')
    return lda_model

def compute_idf(textList):
    docFrequency = collections.defaultdict(float)
    for text in textList:
        charList = text.split(u' ')
        charSet = set(charList)
        for char in charSet:
            docFrequency[char] += 1.0
    iDocFrequency = collections.defaultdict(float)
    for char in docFrequency.keys():
        iDocFrequency[char] = np.log10(float(len(textList)) / docFrequency[char])
    return iDocFrequency

def buildEmbedding():
    textList = getTextList(preTrainDF, preTestDF, semiTrainDF, semiTestDF)
    with codecs.open('./dic&corp/text.txt', 'w', 'utf-8') as fw:
        for text in textList:
            fw.write(text + '\n')
    model = fasttext.cbow('./dic&corp/text.txt', 'model.cbow', dim=128)

def getEmb_model():
    model = fasttext.load_model('./dic&corp/model.cbow.bin')
    return model

def getPosDict():
    posDict = {
        'null': 0,
        'a': 1,
        'b': 2,
        'c': 3,
        'd': 4,
        'e': 5,
        'g': 6,
        'h': 7,
        'i': 8,
        'j': 9,
        'k': 10,
        'm': 11,
        'n': 12,
        'nd': 13,
        'nh': 14,
        'ni': 15,
        'nl': 16,
        'ns': 17,
        'nt': 18,
        'nz': 19,
        'o': 20,
        'p': 21,
        'q': 22,
        'r': 23,
        'u': 24,
        'v': 25,
        'wp': 26,
        'ws': 27,
        'x': 28,
        'z': 29,
    }
    return posDict

class xgbFeatureGenerator():
    def __init__(self, textList, textTagProb):
        self.posDict = getPosDict()
        self.oneENC = sklearn.preprocessing.OneHotEncoder().fit([[i] for i in range(len(self.posDict.keys()))])
        self.iDocFrequency = compute_idf(textList)
        self.textTagProb = textTagProb
        self.corpus, self.dictionary = getCorpusDictionary()
        self.tfidf_model = getTfIdf_model()
        self.lsi_model = getLsi_model()
        self.lda_model = getLda_model()
        self.emb_model = getEmb_model()
        self.segmentor, self.postagger = util.ltpSetup()

    def getPosOfWord(self, text, word):

        startIndex = int(self.getWordStartIndex(text, word))
        wordList = [w for w in util.wordSeg(self.segmentor, text)]
        posList = self.postagger.postag(wordList)
        wordList = [w.decode('utf-8') for w in wordList]
        posCode = self.oneENC.transform([[0]]).toarray().reshape(-1)
        str = u''
        for (w, pos) in zip(wordList, posList):
            str += w
            if word == w and len(str) > startIndex:
                posCode = self.posDict[pos]
                posCode = self.oneENC.transform([[posCode]]).toarray().reshape(-1)
                break
        return posCode

    def getWordStartIndex(self, text, word):
        startIndex = 0.0
        w = re.compile(re.escape(word))
        for seg_sign in w.finditer(text):
            startIndex = seg_sign.start()
            break
        return float(startIndex)

    def getWordTF(self, text, word):
        charSet = set(list(word))
        termFrequency = np.asarray([float(count) for (char, count) in collections.Counter(list(text)).most_common() if char in charSet]).sum() / len(text)
        return termFrequency

    def getWordIDF(self, word):
        idf = np.asarray([self.iDocFrequency[char] for char in list(word)]).sum() / len(word)
        return idf

    def getSimilarity(self, text, word, model):
        word_bow = self.dictionary.doc2bow(list(word))
        word_tfidf = self.tfidf_model[word_bow]
        text_bow = self.dictionary.doc2bow(list(text))
        text_tfidf = self.tfidf_model[text_bow]

        similarity = 0.0

        if model == 'vsm':
            for (i, i_score) in word_tfidf:
                for (j, j_score) in text_tfidf:
                    if i == j:
                        similarity += i_score * j_score
        if model == 'lsi':
            word_lsi = self.lsi_model[word_tfidf]
            text_lsi = self.lsi_model[text_tfidf]
            vec1 = np.asarray([i_score for (i, i_score) in word_lsi])
            vec2 = np.asarray([j_score for (j, j_score) in text_lsi])
            similarity = np.dot(vec1, vec2.T)
        if model == 'lda':
            word_lda = self.lda_model[word_tfidf]
            text_lda = self.lda_model[text_tfidf]
            vec1 = np.asarray([i_score for (i, i_score) in word_lda])
            vec2 = np.asarray([j_score for (j, j_score) in text_lda])
            similarity = np.dot(vec1, vec2.T)
        return similarity

    def getWordEmb(self, word):
        charList = list(word)
        wordEmb = np.asarray([self.emb_model[char] for char in charList]).sum(axis=0) / float(len(charList))
        return wordEmb

    def getEmbSimilarity(self, text, word):
        wordEmb = self.getWordEmb(word)
        textEmb = self.getWordEmb(text)
        similarity = np.dot(wordEmb, textEmb.T) / np.sqrt(np.square(wordEmb).sum()) * np.sqrt(np.square(textEmb).sum())
        return similarity

    def getTextRankWeight(self, text, word):
        tr4w = TextRank4Keyword()
        tr4w.analyze(text=text, lower=True, window=2)
        weight = 0.0
        for item in tr4w.get_keywords(20, word_min_len=1):
            if item.word == word:
                weight = item.weight
        return weight

    def generateFeatureOfWord(self, text, word):
        if word == u'' or word == u'NULL':
            featureDict = {
                'pos': self.oneENC.transform([[0]]).toarray().reshape(-1),
                'startIndex': -1.0,
                'wordLength': 0.0,
                'tf': 0.0,
                'idf': 0.0,
                'tf_idf': 0.0,
                'vsm_sim': 0.0,
                'lsi_sim': 0.0,
                'lda_sim': 0.0,
                'embedding': np.asarray(self.emb_model[u'']),
                'emb_sim': 0.0,
                'tr_weight': 0.0
            }
            return featureDict
        else:
            tf = self.getWordTF(text, word)
            idf = self.getWordIDF(word)
            featureDict = {
                'pos': self.getPosOfWord(text, word),
                'startIndex': self.getWordStartIndex(text, word) / len(text),
                'wordLength': float(len(word)),
                'tf': tf,
                'idf': idf,
                'tf_idf': tf * idf,
                'vsm_sim': self.getSimilarity(text, word, 'vsm'),
                'lsi_sim': self.getSimilarity(text, word, 'lsi'),
                'lda_sim': self.getSimilarity(text, word, 'lda'),
                'embedding': self.getWordEmb(word),
                'emb_sim': self.getEmbSimilarity(text, word),
                'tr_weight': self.getTextRankWeight(text, word)
            }
            return featureDict

    def generateFeatureOfWordPair(self, text, theme, kywrd):
        if kywrd == u'' or theme == u'NULL':
            featureDict = {
                'wordDistance': 0.0,
                'lsi_sim': 0.0,
                'lda_sim': 0.0,
                'emb_sim': 0.0
            }
            return featureDict
        else:
            featureDict = {
                'wordDistance': (self.getWordStartIndex(text, kywrd) - self.getWordStartIndex(text, theme)) / len(text),
                'lsi_sim': self.getSimilarity(theme, kywrd, 'lsi'),
                'lda_sim': self.getSimilarity(theme, kywrd, 'lda'),
                'emb_sim': self.getEmbSimilarity(theme, kywrd)
            }
            return featureDict

    def generateFeatureOfText(self, index, textID, text):
        featureDict = {
            'textLength': np.asarray([float(len(text))]),
            'embedding': self.getWordEmb(text),
            'tagProb': np.asarray([wordTagProb.values() for wordTagProb in self.textTagProb[index][textID]]).sum(axis=0) / float(len(self.textTagProb[index][textID]))
        }
        return featureDict

    def generateFeatureOfResultSet(self, index, textID, text, resultSet):
        textFeature = self.generateFeatureOfText(index, textID, text)
        resultFeatureSet = []
        for (theme, kywrd) in resultSet:
            themeFeature = self.generateFeatureOfWord(text, theme)
            kywrdFeature = self.generateFeatureOfWord(text, kywrd)
            wordPairFeature = self.generateFeatureOfWordPair(text, theme, kywrd)

            resultFeature = np.concatenate((textFeature['textLength'], textFeature['embedding'], textFeature['tagProb'], themeFeature['embedding'], themeFeature['pos'], kywrdFeature['embedding'], kywrdFeature['pos']), axis=0)

            tF = np.asarray([value for (key, value) in themeFeature.items() if key not in ['embedding', 'pos']])
            kF = np.asarray([value for (key, value) in kywrdFeature.items() if key not in ['embedding', 'pos']])
            wpF = np.asarray([value for (key, value) in wordPairFeature.items()])

            resultFeature = np.concatenate((resultFeature, tF, kF, wpF), axis=0)
            resultFeatureSet.append(resultFeature)

        return resultFeatureSet

    def generateLabelOfResultSet(self, resultSet, truthSet):
        labelList = []
        if len(resultSet) == 1:
            labelList.append(1)
        else:
            flag = True
            for result in resultSet:
                if result in truthSet:
                    labelList.append(1)
                    flag = False
                else:
                    labelList.append(0)
            if flag:
                labelList[-1] = 1
        return labelList

def buildCandidatePair(themeTruthList, kywrdTruthList, text):
    candidatePairList = []
    themeCandidate = [theme for theme in themeTruthList if theme in text]
    kywrdCandidate = [kywrd for kywrd in kywrdTruthList if kywrd in text]
    if len(kywrdCandidate) > 0:
        themeCandidate.append(u'NULL')
    for theme in themeCandidate:
        for kywrd in kywrdCandidate:
            candidatePairList.append((theme, kywrd))
    candidatePairList.append((u'', u''))
    return candidatePairList

def getSubstringFeature(wordList):
    substringCount = dict()
    for w1 in wordList:
        substringCount[w1] = 0.0
        for w2 in wordList:
            if w2 != u'' and w1 != w2 and w2 in w1:
                substringCount[w1] += 1.0
    return np.asarray([substringCount[word] for word in wordList]) / float(len(wordList))

def getLevDistance(wordList):
    levDis = dict()
    for w1 in wordList:
        levDis[w1] = 0.0
        for w2 in wordList:
            if w2 != u'' and w1 != w2:
                levDis[w1] += Levenshtein.distance(w1, w2)
    return np.asarray([levDis[word] for word in wordList]) / float(len(wordList))

def getLevRatio(wordList):
    levDis = dict()
    for w1 in wordList:
        levDis[w1] = 0.0
        for w2 in wordList:
            if w2 != u'' and w1 != w2:
                levDis[w1] += Levenshtein.ratio(w1, w2)
    return np.asarray([levDis[word] for word in wordList]) / float(len(wordList))

def multiprocessFeatureData(params):

    index, textList = params
    print 'Processing {}...'.format(index)
    textList = [u''.join(text) for text in textList]
    X = []
    y = []

    themeList = []
    kywrdList = []

    if type(df.iloc[index]['theme']) != float:
        themeList = [theme.decode('utf-8').strip() for theme in df.iloc[index]['theme'].split(';')[:-1]]
    if type(df.iloc[index]['keyword']) != float:
        kywrdList = [kywrd.decode('utf-8').strip() for kywrd in df.iloc[index]['keyword'].split(';')[:-1]]

    truthSet = zip(themeList, kywrdList)

    for textID, text in enumerate(textList):

        resultSet = buildCandidatePair(themeTruthList, kywrdTruthList, text)
        X.extend(xFG.generateFeatureOfResultSet(index, textID, text, resultSet))
        y.extend(xFG.generateLabelOfResultSet(resultSet, truthSet))

    X = np.asarray(X)
    y = np.asarray(y)

    return (X, y)

def addNewFeature(semiTrainDF, data):

    themeTruthList = [theme for (theme, count) in util.readWordList('./data/themeTruthList-semiTrain.txt')]
    kywrdTruthList = [kywrd for (kywrd, count) in util.readWordList('./data/kywrdTruthList-semiTrain.txt')]

    with open('./xgb-data-%s.pkl' % data, 'rb') as fr:
        semiTrain = pickle.load(fr)
    contentList = semiTrainDF['sub_sents_tokenized'].values
    for index, textList in enumerate(contentList):
        print 'Processing {}...'.format(index)
        textList = [u''.join(text) for text in textList]
        all_resultSet = []
        for text in textList:
            resultSet = buildCandidatePair(themeTruthList, kywrdTruthList, text)
            for result in resultSet:
                all_resultSet.append((result, text))
        # To do
        # 传入all_resultSet抽取特征 返回新特征
        # To do
        NEW_FEATURE = np.asarray(range(len(all_resultSet))).reshape(-1, 1)
        X, y = semiTrain[index]
        X = np.concatenate((X, NEW_FEATURE), axis=1)
        semiTrain[index] = (X, y)

    with open('./xgb-data-new-%s.pkl' % data, 'wb') as fw:
        pickle.dump(semiTrain, fw)
    print '** Finished adding NEW Feature.'

def addSubstringTag(df, data):
    themeTruthList = [theme for (theme, count) in util.readWordList('./data/themeTruthList-semiTrain.txt')]
    kywrdTruthList = [kywrd for (kywrd, count) in util.readWordList('./data/kywrdTruthList-semiTrain.txt')]

    with open('./xgb-data-%s.pkl' % data, 'rb') as fr:
        featureData = pickle.load(fr)
    contentList = df['sub_sents_tokenized'].values

    for index, textList in enumerate(contentList):
        print 'Processing {}...'.format(index)
        textList = [u''.join(text) for text in textList]
        NEW_FEATURE = []
        for textID, text in enumerate(textList):
            resultSet = buildCandidatePair(themeTruthList, kywrdTruthList, text)
            tL, kL = zip(*resultSet)
            tmp = np.asarray([getSubstringFeature(tL), getLevDistance(tL), getLevRatio(tL), getSubstringFeature(kL), getLevDistance(kL), getLevRatio(kL)])
            NEW_FEATURE.append(tmp)
        try:
            NEW_FEATURE = np.hstack(NEW_FEATURE).transpose()
            X, y = featureData[index]
            X = np.concatenate((X, NEW_FEATURE), axis=1)
            featureData[index] = (X, y)
        except Exception:
            print '{} is null'.format(index)
    with open('./xgb-data-%s-addSubstringTag.pkl' % data, 'wb') as fw:
        pickle.dump(featureData, fw)
    print '** Finished adding %s NEW Feature.' % data

def addNewTextTagProb(df, data):

    themeTruthList = [theme for (theme, count) in util.readWordList('./data/themeTruthList-semiTrain.txt')]
    kywrdTruthList = [kywrd for (kywrd, count) in util.readWordList('./data/kywrdTruthList-semiTrain.txt')]

    with open('./xgb-data-%s.pkl' % data, 'rb') as fr:
        featureData = pickle.load(fr)
    contentList = df['sub_sents_tokenized'].values

    textTagProb = df['tsn_prob_word'].values

    for index, textList in enumerate(contentList):
        print 'Processing {}...'.format(index)
        textList = [u''.join(text) for text in textList]
        NEW_FEATURE = []
        for textID, text in enumerate(textList):
            tagProb = np.asarray([wordTagProb.values() for wordTagProb in textTagProb[index][textID]]).sum(axis=0) / float(len(textTagProb[index][textID]))
            resultSet = buildCandidatePair(themeTruthList, kywrdTruthList, text)
            for _ in resultSet:
                NEW_FEATURE.append(tagProb)
        NEW_FEATURE = np.asarray(NEW_FEATURE)
        X, y = featureData[index]
        try:
            X = np.concatenate((X, NEW_FEATURE), axis=1)
            featureData[index] = (X, y)
        except Exception:
            print '{} is null'.format(index)

    with open('./xgb-data-%s-addTSN.pkl' % data, 'wb') as fw:
        pickle.dump(featureData, fw)
    print '** Finished adding %s NEW Feature.' % data

def addCharTagProb(df, data):

    themeTruthList = [theme for (theme, count) in util.readWordList('./data/themeTruthList-semiTrain.txt')]
    kywrdTruthList = [kywrd for (kywrd, count) in util.readWordList('./data/kywrdTruthList-semiTrain.txt')]

    with open('./xgb-data-%s.pkl' % data, 'rb') as fr:
        featureData = pickle.load(fr)
    contentList = df['sub_sents_tokenized'].values

    for charTagProb in [df['bie_prob_char'].values, df['tsn_prob_char'].values]:

        for index, textList in enumerate(contentList):
            print 'Processing {}...'.format(index)
            textList = [u''.join(text) for text in textList]
            NEW_FEATURE = []
            for textID, text in enumerate(textList):
                tagProb = np.asarray([wordTagProb.values() for wordTagProb in charTagProb[index][textID]]).sum(axis=0) / float(len(charTagProb[index][textID]))
                resultSet = buildCandidatePair(themeTruthList, kywrdTruthList, text)
                for _ in resultSet:
                    NEW_FEATURE.append(tagProb)
            NEW_FEATURE = np.asarray(NEW_FEATURE)
            X, y = featureData[index]
            try:
                X = np.concatenate((X, NEW_FEATURE), axis=1)
                featureData[index] = (X, y)
            except Exception:
                print '{} is null'.format(index)

    with open('./xgb-data-%s-addCharTagProb.pkl' % data, 'wb') as fw:
        pickle.dump(featureData, fw)
    print '** Finished adding %s NEW Feature.' % data

def addResultCount(df, data):

    themeTruthList = [theme for (theme, count) in util.readWordList('./data/themeTruthList-semiTrain.txt')]
    kywrdTruthList = [kywrd for (kywrd, count) in util.readWordList('./data/kywrdTruthList-semiTrain.txt')]

    with open('./xgb-data-%s.pkl' % data, 'rb') as fr:
        featureData = pickle.load(fr)
    contentList = df['sub_sents_tokenized'].values

    with open('./data/train_test_1210.pkl', 'rb') as fr:
        resultCountDF = pickle.load(fr)
        _ = pickle.load(fr)
    ans_num = resultCountDF['ans_num'].values
    sub_sent_vector = resultCountDF['sub_sent_vector'].values

    for index, textList in enumerate(contentList):
        print 'Processing {}...'.format(index)
        textList = [u''.join(text) for text in textList]
        NEW_FEATURE = []
        for textID, text in enumerate(textList):
            textAnsNum = [ans_num[index][textID]]
            textVec = sub_sent_vector[index][textID]
            tmp = np.concatenate((textAnsNum, textVec), axis=0)
            resultSet = buildCandidatePair(themeTruthList, kywrdTruthList, text)
            for _ in resultSet:
                NEW_FEATURE.append(tmp)
        NEW_FEATURE = np.asarray(NEW_FEATURE)
        X, y = featureData[index]
        try:
            X = np.concatenate((X, NEW_FEATURE), axis=1)
            featureData[index] = (X, y)
        except Exception:
            print '{} is null'.format(index)

    with open('./xgb-data-%s-addResultCount.pkl' % data, 'wb') as fw:
        pickle.dump(featureData, fw)
    print '** Finished adding %s NEW Feature.' % data

def addPairFeature(df, data):
    with open('./xgb-data-%s.pkl' % data, 'rb') as fr:
        featureData = pickle.load(fr)
    contentList = df['sub_sents_tokenized'].values

    with open('./data/pair_features_chandi.pkl', 'rb') as fr:
        pairFeatureDF = pickle.load(fr)
        _ = pickle.load(fr)

    pairFeatureList = pairFeatureDF['pair_features'].values

    for index, textList in enumerate(contentList):
        print 'Processing {}...'.format(index)
        textList = [u''.join(text) for text in textList]
        NEW_FEATURE = []
        for textID, text in enumerate(textList):
            tmp = np.asarray([pairFeature.values() for pairFeature in pairFeatureList[index][textID]])
            NEW_FEATURE.append(tmp)
        try:
            NEW_FEATURE = np.vstack(NEW_FEATURE)
            X, y = featureData[index]
            X = np.concatenate((X, NEW_FEATURE), axis=1)
            featureData[index] = (X, y)
        except Exception:
            print '{} is null'.format(index)
    with open('./xgb-data-%s-addPairFeature.pkl' % data, 'wb') as fw:
        pickle.dump(featureData, fw)
    print '** Finished adding %s NEW Feature.' % data

def multiprocessFeatureDataForTest(params):

    index, textList = params
    print 'Processing {}...'.format(index)
    textList = [u''.join(text) for text in textList]
    X = []

    for textID, text in enumerate(textList):

        resultSet = buildCandidatePair(themeTruthList, kywrdTruthList, text)
        X.extend(xFG.generateFeatureOfResultSet(index, textID, text, resultSet))

    X = np.asarray(X)

    return X

def addSubstringTagForTest(df, data):
    themeTruthList = [theme for (theme, count) in util.readWordList('./data/themeTruthList-semiTrain.txt')]
    kywrdTruthList = [kywrd for (kywrd, count) in util.readWordList('./data/kywrdTruthList-semiTrain.txt')]

    with open('./xgb-data-%s.pkl' % data, 'rb') as fr:
        featureData = pickle.load(fr)
    contentList = df['sub_sents_tokenized'].values

    for index, textList in enumerate(contentList):
        print 'Processing {}...'.format(index)
        textList = [u''.join(text) for text in textList]
        NEW_FEATURE = []
        for textID, text in enumerate(textList):
            resultSet = buildCandidatePair(themeTruthList, kywrdTruthList, text)
            tL, kL = zip(*resultSet)
            tmp = np.asarray([getSubstringFeature(tL), getSubstringFeature(kL)])
            # tmp = np.asarray([getSubstringFeature(tL), getLevDistance(tL), getLevRatio(tL), getSubstringFeature(kL), getLevDistance(kL), getLevRatio(kL)])
            NEW_FEATURE.append(tmp)
        try:
            NEW_FEATURE = np.hstack(NEW_FEATURE).transpose()
            X = featureData[index]
            X = np.concatenate((X, NEW_FEATURE), axis=1)
            featureData[index] = X
        except Exception:
            print '{} is null'.format(index)
    with open('./xgb-data-%s-addSubstringTag.pkl' % data, 'wb') as fw:
        pickle.dump(featureData, fw)
    print '** Finished adding %s NEW Feature.' % data

def addNewTextTagProbForTest(df, data):

    themeTruthList = [theme for (theme, count) in util.readWordList('./data/themeTruthList-semiTrain.txt')]
    kywrdTruthList = [kywrd for (kywrd, count) in util.readWordList('./data/kywrdTruthList-semiTrain.txt')]

    with open('./xgb-data-%s.pkl' % data, 'rb') as fr:
        featureData = pickle.load(fr)
    contentList = df['sub_sents_tokenized'].values

    textTagProb = df['tsn_prob_word'].values

    for index, textList in enumerate(contentList):
        print 'Processing {}...'.format(index)
        textList = [u''.join(text) for text in textList]
        NEW_FEATURE = []
        for textID, text in enumerate(textList):
            tagProb = np.asarray([wordTagProb.values() for wordTagProb in textTagProb[index][textID]]).sum(axis=0) / float(len(textTagProb[index][textID]))
            resultSet = buildCandidatePair(themeTruthList, kywrdTruthList, text)
            for _ in resultSet:
                NEW_FEATURE.append(tagProb)
        NEW_FEATURE = np.asarray(NEW_FEATURE)
        X = featureData[index]
        try:
            X = np.concatenate((X, NEW_FEATURE), axis=1)
            featureData[index] = X
        except Exception:
            print '{} is null'.format(index)

    with open('./xgb-data-%s-addTSN.pkl' % data, 'wb') as fw:
        pickle.dump(featureData, fw)
    print '** Finished adding %s NEW Feature.' % data

def addCharTagProbForTest(df, data):

    themeTruthList = [theme for (theme, count) in util.readWordList('./data/themeTruthList-semiTrain.txt')]
    kywrdTruthList = [kywrd for (kywrd, count) in util.readWordList('./data/kywrdTruthList-semiTrain.txt')]

    with open('./xgb-data-%s.pkl' % data, 'rb') as fr:
        featureData = pickle.load(fr)
    contentList = df['sub_sents_tokenized'].values

    for charTagProb in [df['bie_prob_char'].values, df['tsn_prob_char'].values]:

        for index, textList in enumerate(contentList):
            print 'Processing {}...'.format(index)
            textList = [u''.join(text) for text in textList]
            NEW_FEATURE = []
            for textID, text in enumerate(textList):
                tagProb = np.asarray([wordTagProb.values() for wordTagProb in charTagProb[index][textID]]).sum(axis=0) / float(len(charTagProb[index][textID]))
                resultSet = buildCandidatePair(themeTruthList, kywrdTruthList, text)
                for _ in resultSet:
                    NEW_FEATURE.append(tagProb)
            NEW_FEATURE = np.asarray(NEW_FEATURE)
            X = featureData[index]
            try:
                X = np.concatenate((X, NEW_FEATURE), axis=1)
                featureData[index] = X
            except Exception:
                print '{} is null'.format(index)

    with open('./xgb-data-%s-addCharTagProb.pkl' % data, 'wb') as fw:
        pickle.dump(featureData, fw)
    print '** Finished adding %s NEW Feature.' % data

def addResultCountForTest(df, data):

    themeTruthList = [theme for (theme, count) in util.readWordList('./data/themeTruthList-semiTrain.txt')]
    kywrdTruthList = [kywrd for (kywrd, count) in util.readWordList('./data/kywrdTruthList-semiTrain.txt')]

    with open('./xgb-data-%s.pkl' % data, 'rb') as fr:
        featureData = pickle.load(fr)
    contentList = df['sub_sents_tokenized'].values

    with open('./data/train_test_1210.pkl', 'rb') as fr:
        _ = pickle.load(fr)
        resultCountDF = pickle.load(fr)
    ans_num = resultCountDF['ans_num'].values
    sub_sent_vector = resultCountDF['sub_sent_vector'].values

    for index, textList in enumerate(contentList):
        print 'Processing {}...'.format(index)
        textList = [u''.join(text) for text in textList]
        NEW_FEATURE = []
        for textID, text in enumerate(textList):
            textAnsNum = [ans_num[index][textID]]
            textVec = sub_sent_vector[index][textID]
            tmp = np.concatenate((textAnsNum, textVec), axis=0)
            resultSet = buildCandidatePair(themeTruthList, kywrdTruthList, text)
            for _ in resultSet:
                NEW_FEATURE.append(tmp)
        NEW_FEATURE = np.asarray(NEW_FEATURE)
        X = featureData[index]
        try:
            X = np.concatenate((X, NEW_FEATURE), axis=1)
            featureData[index] = X
        except Exception:
            print '{} is null'.format(index)

    with open('./xgb-data-%s-addResultCount.pkl' % data, 'wb') as fw:
        pickle.dump(featureData, fw)
    print '** Finished adding %s NEW Feature.' % data

def addPairFeatureForTest(df, data):
    with open('./xgb-data-%s.pkl' % data, 'rb') as fr:
        featureData = pickle.load(fr)
    contentList = df['sub_sents_tokenized'].values

    with open('./data/pair_features_chandi.pkl', 'rb') as fr:
        pairFeatureDF = pickle.load(fr)
        _ = pickle.load(fr)

    pairFeatureList = pairFeatureDF['pair_features'].values

    for index, textList in enumerate(contentList):
        print 'Processing {}...'.format(index)
        textList = [u''.join(text) for text in textList]
        NEW_FEATURE = []
        for textID, text in enumerate(textList):
            tmp = np.asarray([pairFeature.values() for pairFeature in pairFeatureList[index][textID]])
            NEW_FEATURE.append(tmp)
        try:
            NEW_FEATURE = np.vstack(NEW_FEATURE)
            X = featureData[index]
            X = np.concatenate((X, NEW_FEATURE), axis=1)
            featureData[index] = X
        except Exception:
            print '{} is null'.format(index)
    with open('./xgb-data-%s-addPairFeature.pkl' % data, 'wb') as fw:
        pickle.dump(featureData, fw)
    print '** Finished adding %s NEW Feature.' % data

def xgbTrainAndValid(data):

    start_time = time.time()
    with open('./xgb-data-%s.pkl' % data, 'rb') as fr:
        semiTrain = pickle.load(fr)

    featureData = semiTrain
    indexList = range(len(featureData))
    tmp = [0] * len(featureData)

    trainFeatureData, validFeatureData, _, _ = train_test_split(zip(indexList, featureData), tmp, test_size=0.2, random_state=42)

    X_train = []
    y_train = []
    for (index, featureData) in trainFeatureData:
        X_train.extend(featureData[0])
        y_train.extend(featureData[1])
    X_train = np.asarray(X_train)
    y_train = np.asarray(y_train)

    X_valid = []
    y_valid = []
    for (index, featureData) in validFeatureData:
        X_valid.extend(featureData[0])
        y_valid.extend(featureData[1])
    X_valid = np.asarray(X_valid)
    y_valid = np.asarray(y_valid)

    xgb_train = xgb.DMatrix(X_train, label=y_train)
    xgb_valid = xgb.DMatrix(X_valid, label=y_valid)

    del semiTrain, trainFeatureData, validFeatureData, X_train, y_train, X_valid

    params = {
        'booster': 'gbtree',
        'silent': 0,  # 设置成1则没有运行信息输出，最好是设置为0.
        # 'nthread': -1,# cpu 线程数 默认最大
        'eta': 0.007,  # 如同学习率
        'min_child_weight': 3,
        # 这个参数默认是 1，是每个叶子里面 h 的和至少是多少，对正负样本不均衡时的 0-1 分类而言
        # ，假设 h 在 0.01 附近，min_child_weight 为 1 意味着叶子节点中最少需要包含 100 个样本。
        # 这个参数非常影响结果，控制叶子节点中二阶导的和的最小值，该参数值越小，越容易 overfitting。
        'max_depth': 6,  # 构建树的深度，越大越容易过拟合 5-15
        'gamma': 0.1,  # 树的叶子节点上作进一步分区所需的最小损失减少,越大越保守，一般0.1、0.2这样子。
        'subsample': 0.85,  # 随机采样训练样本
        'colsample_bytree': 0.7,  # 生成树时进行的列采样
        'lambda': 2,  # 控制模型复杂度的权重值的L2正则化项参数，参数越大，模型越不容易过拟合。
        # 'alpha':0, # L1 正则项参数
        'scale_pos_weight': 1, #如果取值大于0的话，在类别样本不平衡的情况下有助于快速收敛。
        'objective': 'binary:logistic',
        # 'objective': 'multi:softmax', #多分类的问题
        # 'num_class':10, # 类别数，多分类与 multisoftmax 并用
        'seed': 1000,  # 随机种子
        # 'eval_metric': 'auc'
    }

    plst = list(params.items())
    num_rounds = 10000  # 迭代次数
    watchlist = [(xgb_train, 'train'), (xgb_valid, 'valid')]

    # model = xgb.train(plst, xgb_train, num_rounds, watchlist, early_stopping_rounds=100)
    model = xgb.train(plst, xgb_train, num_rounds, watchlist, early_stopping_rounds=1000)
    model.save_model('./dic&corp/xgb-%s.model' % data) # 用于存储训练出的模型
    print "best best_ntree_limit", model.best_ntree_limit
    cost_time = time.time() - start_time
    print "xgboost success!", '\n', "cost time:", cost_time, "(s)......"

    model = xgb.Booster({'nthread': -1})  # init model
    model.load_model('./dic&corp/xgb-%s.model' % data)  # load data
    y_pred = model.predict(xgb_valid)

    for i in range(len(y_pred)):
        if y_pred[i] > 0.5:
            y_pred[i] = 1
        else:
            y_pred[i] = 0
    print "Accuracy : %.4g" % sklearn.metrics.accuracy_score(y_valid, y_pred)
    xgb.plot_importance(model)
    plt.show()

if __name__ == '__main__':

    # Add New Feature
    addPairFeature(semiTrainDF, 'semiTrain-BIEonly-addTSN-addSubstringTag-addResultCount-addCharTagProb')
    # Add New Feature

    # xgbTrain
    xgbTrainAndValid('semiTrain-BIEonly-addTSN-addSubstringTag-addResultCount-addCharTagProb-addPairFeature')
    # xgbTrain

    # Add New Feature
    addPairFeatureForTest(semiTestDF, 'semiTest-BIEonly-addTSN-addSubstringTag-addResultCount-addCharTagProb')
    # Add New Feature

    # Test Data Predict
    data = 'BIEonly-addTSN-addSubstringTag-addResultCount-addCharTagProb-addPairFeature'
    with open('./xgb-data-semiTest-%s.pkl' % data, 'rb') as fr:
        featureDataList = pickle.load(fr)

    X_test = []
    for featureData in featureDataList:
        X_test.extend(featureData)
    X_test = np.asarray(X_test)
    xgb_test = xgb.DMatrix(X_test)

    model = xgb.Booster({'nthread': 4})  # init model
    model.load_model('./dic&corp/xgb-semiTrain-%s.model' % data)  # load data
    y_pred = model.predict(xgb_test)

    with open('./y_pred-semiTest-%s.pkl' % data, 'wb') as fw:
        pickle.dump(y_pred, fw)
    # Test Data Predict