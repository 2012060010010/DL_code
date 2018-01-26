# coding=utf-8

import numpy as np
import pickle
import util

with open('./data/features_chandi_1210.pkl', 'rb') as fr:
    preTrainDF = pickle.load(fr)
    preTestDF = pickle.load(fr)
    semiTrainDF = pickle.load(fr)
    semiTestDF = pickle.load(fr)

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

if __name__ == '__main__':

    # Merge All Model
    data = 'BIEonly-addTSN-addSubstringTag-addResultCount-'
    stackingList = ['0', '1', '2', '3', '4', 'addCharTagProb', 'addCharTagProb-addPairFeature']
    y_pred = np.zeros(shape=(494265))
    for i in stackingList:
        print i
        with open('./stackingResult/y_pred-semiTest-%s.pkl' % (data + i), 'rb') as fr:
            y_pred += pickle.load(fr)
    y_pred = y_pred / float(len(stackingList))
    with open('./stackingResult/y_pred-semiTest-MergeSeven.pkl', 'wb') as fw:
        pickle.dump(y_pred, fw)
    # Merge All Model

    # Merge All Model Except Two
    data = 'BIEonly-addTSN-addSubstringTag-addResultCount-'
    stackingList = ['0', '1', '3', '4', 'addCharTagProb', 'addCharTagProb-addPairFeature']
    y_pred = np.zeros(shape=(494265))
    for i in stackingList:
        print i
        with open('./stackingResult/y_pred-semiTest-%s.pkl' % (data + i), 'rb') as fr:
            y_pred += pickle.load(fr)
    y_pred = y_pred / float(len(stackingList))
    with open('./stackingResult/y_pred-semiTest-MergeSevenExceptTwo.pkl', 'wb') as fw:
        pickle.dump(y_pred, fw)
    # Merge All Model Except Two

    # Test Data Answer
    for data in ['MergeSeven', 'MergeSevenExceptTwo']:
        with open('./stackingResult/y_pred-semiTest-%s.pkl' % data, 'rb') as fr:
            y_pred = pickle.load(fr)

        df = util.getSemiTestDF()

        themeTruthList = [theme for (theme, count) in util.readWordList('./data/themeTruthList-semiTrain.txt')]
        kywrdTruthList = [kywrd for (kywrd, count) in util.readWordList('./data/kywrdTruthList-semiTrain.txt')]

        contentList = semiTestDF['sub_sents_tokenized'].values

        print 'contentList len={}'.format(len(contentList))
        show_num = 0
        match_index = 0

        all_resultList = []

        for textList in contentList:

            textList = [u''.join(text) for text in textList]
            resultSet = []

            for text in textList:

                if show_num % 10 == 0:
                    print 'Processing {}...'.format(show_num)
                show_num += 1

                result = buildCandidatePair(themeTruthList, kywrdTruthList, text)
                score = []
                for r in result:
                    score.append(y_pred[match_index])
                    match_index += 1
                _result = [r for (r, s) in zip(result, score) if s > 0.5]
                if len(_result) == 0:
                    _result.extend([r for (r, s) in zip(result, score) if s == max(score)])
                resultSet.extend(_result)

            if len(resultSet) >= 2 and (u'', u'') in resultSet:
                while len(resultSet) != 0 and (u'', u'') in resultSet:
                    resultSet.remove((u'', u''))
            if len(resultSet) == 0:
                resultSet.append((u'', u''))

            all_resultList.append(resultSet)

        themeStrList = []
        kywrdStrList = []

        for result in all_resultList:
            result = [(theme, kywrd) for (theme, kywrd) in result]
            if len(result) > 0:
                themeList, kywrdList = zip(*result)
            else:
                themeList, kywrdList = [], []
            if u'' not in themeList:
                themeStrList.append(u';'.join(themeList) + u';')
                kywrdStrList.append(u';'.join(kywrdList) + u';')
            else:
                themeStrList.append(u'')
                kywrdStrList.append(u'')

        df['theme'] = themeStrList
        df['sentiment_word'] = kywrdStrList

        import polarityClassification

        CRF = polarityClassification.senti_crf_gen()
        df = df.apply(polarityClassification.senti_gen_with_crf, axis=1, args=(CRF,))
        df = df.apply(polarityClassification.pair_postpreprocess, axis=1)

        df.to_csv('./submission-Test-%s.csv' % data, index=False, encoding='utf-8', columns=['id', 'content', 'theme', 'sentiment_word', 'sentiment_anls'], header=None)
    # Test Data Answer