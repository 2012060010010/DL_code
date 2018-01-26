# coding=utf-8

import codecs
import gensim
import jieba
import jieba.analyse as analyse
import re
from nltk.corpus import stopwords
import collections
import thulac

jieba.load_userdict('./wordDict7.22.txt')
jieba.enable_parallel(4)

thu_seg = thulac.thulac(user_dict='./wordDict7.22.txt', seg_only=True)

stops = list(stopwords.words("chinese"))
stops.extend(list(stopwords.words("english")))
stops = set(stops)

pattern = u"[0-9\s\.\!\/_,$%^*()?；:\-【】\"\']+|[——！，;：。？、~@￥%……&*（）\[\]《》“”‘’]+"

def elementWiseCompare(word):
    result = True
    if len(word) > 2:
        if word[0] == u'第' and word[1] in list(u'一二三四五六七八九十'):
            result = False
        return result
    else:
        if not re.search(u'[a-z\sA-Z]+', word):
            charList = list(word)
            for char in charList:
                if char in stops:
                    result = False
                    break
        return result

def get_blogID2Content(path):
    blogID2Content = dict()
    with codecs.open(path, 'r', 'utf-8') as fr:
        lines = fr.readlines()
        for line in lines:
            try:
                ID = line.strip().split(' ')[0]
                blog = ' '.join(line.strip().split(' ')[1].split('\001'))
                blogID2Content[ID] = blog
            except Exception:
                pass
    return blogID2Content

def get_blogContent(path):
    blogContent = []
    with codecs.open(path, 'r', 'utf-8') as fr:
        lines = fr.read().split('\n')[:-1]
        for line in lines:
            try:
                blog = ' '.join(line.strip().split(' ')[1].split('\001'))
                blogContent.append(blog)
            except Exception:
                pass
    return blogContent

def get_blogIDTitle(path):
    blogIDTitle = []
    with codecs.open(path, 'r', 'utf-8') as fr:
        lines = fr.read().split('\n')[:-1]
        for line in lines:
            try:
                blogID = line.strip().split(' ')[0]
                blogTitle = ' '.join(line.strip().split(' ')[1].split('\001'))
                blogIDTitle.append((blogID, blogTitle))
            except Exception:
                pass
    return blogIDTitle

def build_TitleDictionary(blogTitleList):
    documents = []
    for blogTitle in blogTitleList:
        documents.append(blogTitle)
    texts = [[word for word in document.split()] for document in documents]
    dictionary = gensim.corpora.Dictionary(texts)
    dictionary.save('./dic&corp/title/csdnTitle.dict')

    corpus = [dictionary.doc2bow(text) for text in texts]
    gensim.corpora.MmCorpus.serialize('./dic&corp/title/csdnTitle.mm', corpus)

def segTitle(title):
    title = re.sub(pattern, u' ', title)
    segList = jieba.cut(title.lower(), cut_all=False)
    wordSet = [word.lower() for word in segList if word != u' ']
    if len(wordSet) > 0:
        return wordSet
    else:
        return []

def wordListRefine(wordList, target):
    refineList = []
    shortWordList = [word for word in wordList if len(word) < 3]
    longWordList = [word for word in wordList if len(word) >= 3]
    for word in wordList:
        for longWord in longWordList:
            if word in longWord:
                refineList.append(longWord)
    if target == 'title':
        if len(refineList) == len(longWordList): #1
            refineList.extend(shortWordList)     #1
    elif target == 'blog':
        refineList.extend(shortWordList) #2
    return refineList

def get_trainIDLabel(IDPath):
    blogID2Label = dict()
    with codecs.open(IDPath, 'r', 'utf-8') as fr:
        lines = fr.readlines()
        for line in lines:
            blogID = line.strip().split('\001')[0]
            blogLabel = ' '.join(line.strip().split('\001')[1:])
            blogID2Label[blogID] = blogLabel
    return blogID2Label

def get_validIDList(IDPath):
    blogIDList = []
    with codecs.open(IDPath, 'r', 'utf-8') as fr:
        lines = fr.readlines()
        for line in lines:
            blogID = line.strip()
            blogIDList.append(blogID)
    return blogIDList

def get_blogTitle(blogID, path):
    blogTitle = ''
    with codecs.open(path, 'r', 'utf-8') as fr:
        lines = fr.read().split('\n')[:-1]
        for line in lines:
            _blogID = line.split('\001')[0].strip()
            if _blogID == blogID:
                blogTitle = line.split('\001')[1].strip()
                break
    return blogTitle

def get_validBlogID2Content(path):
    blogID2Content = dict()
    with codecs.open(path, 'r', 'utf-8') as fr:
        lines = fr.read().split('\n')[:-1]
        for line in lines:
            try:
                blogID = line.strip().split('\001')[0]
                blogTitle = line.strip().split('\001')[1]
                blogContent = line.strip().split('\001')[2]
                blog = blogTitle + blogContent
                blogID2Content[blogID] = blog.lower()
            except Exception:
                pass
    return blogID2Content

def get_validBlogID2Result(path):
    blogID2Result = dict()
    with codecs.open(path, 'r', 'utf-8') as fr:
        lines = fr.readlines()
        for line in lines:
            try:
                blogID = line.strip().split(',')[0]
                blogResult = line.strip().split(',')[1:]
                blogID2Result[blogID] = blogResult
            except Exception:
                pass
    return blogID2Result

def get_jieba_topKeywords(wordList, blogContent):

    num = len(collections.Counter(blogContent.split(' ')).most_common())
    seg = analyse.extract_tags(blogContent, topK=num, withWeight=True)
    result = [(word, weight) for (word, weight) in seg if word in wordList]
    result = sorted(result, key=lambda x: x[1], reverse=True)
    return [word for (word, weight) in result]

def get_tf_topKeywords(wordList, blogContent):

    wordSet = [word for (word, tf) in collections.Counter(blogContent.split(' ')).most_common() if word in wordList]
    return wordSet

def relatedLabelFilter(sourcePath, targetPath):

    validBlogID2Content = get_validBlogID2Content('./tmp/allBlogContent-Test.txt')
    validBlogID2Result = get_validBlogID2Result('./tmp/result-test-ensemble.txt')

    with codecs.open(sourcePath, 'r', 'utf-8') as fr, codecs.open(targetPath, 'w', 'utf-8') as fw:
        lines = fr.readlines()
        show_num = 0
        for line in lines:
            relatedTitle = line.strip().split(' ')[1]
            blogID = line.strip().split(' ')[0]
            print '{} Processing {}'.format(show_num, blogID)
            validBlogResult = validBlogID2Result[blogID]
            if relatedTitle != u'NULL':
                labelList = line.strip().split(' ')[2].split('/')
                validBlogContent = validBlogID2Content[blogID]

                validBlogContent = re.sub(pattern, u' ', validBlogContent)
                segList = thu_seg.cut(validBlogContent.encode('utf-8').lower(), text=True).decode('utf-8').split(u' ')

                wordList = [word.lower() for word in segList if word not in stops and len(word) > 1 and elementWiseCompare(word)]

                candidateList = [word for word in labelList if word in wordList and word not in validBlogResult]
                if len(candidateList) > 0:
                    crucialList = [word for word in labelList if word in wordList and word in validBlogResult]
                    result = []
                    result.extend(crucialList)
                    padNum = 3 - len(result)
                    if padNum > 0:
                        validBlogContent = ' '.join(wordList)
                        result.extend(get_tf_topKeywords(candidateList, validBlogContent)[:padNum])
                        padNum = 3 - len(result)
                        tmp = [word for word in validBlogResult if word not in result]
                        result.extend(tmp[:padNum])
                    fw.write(blogID + ',' + ','.join(result) + '\n')
                else:
                    fw.write(blogID + ',' + ','.join(validBlogResult) + '\n')
            else:
                fw.write(blogID + ',' + ','.join(validBlogResult) + '\n')
            show_num += 1

if __name__ == '__main__':

    # 需要手动设置的 两个 路径
    trainIDLabelPath = './tmp/SMPCUP2017任务1训练集/SMPCUP2017_TrainingData_Task1.txt'  # 训练集ID及对应keywords路径
    testIDPath = './tmp/SMPCUP2017评测集/SMPCUP2017_TestSet_Task1.txt'  # 测试集ID路径
    # 需要手动设置的 两个 路径

    # Compute Title Similarity
    allBlogTitle = get_blogContent('./tmp/dic&corp/segTitle.txt')
    build_TitleDictionary(allBlogTitle)
    print 'Finish!'

    dictionary = gensim.corpora.Dictionary.load('./tmp/dic&corp/title/csdnTitle.dict')
    corpus = gensim.corpora.MmCorpus('./tmp/dic&corp/title/csdnTitle.mm')

    tfidf_model = gensim.models.TfidfModel(corpus)
    tfidf_model.save('./tmp/dic&corp/title/model.tfidf')
    tfidf_model = gensim.models.TfidfModel.load('./tmp/dic&corp/title/model.tfidf')
    corpus_tfidf = tfidf_model[corpus]

    indexSim = gensim.similarities.Similarity('./tmp/dic&corp/title/', corpus_tfidf, len(dictionary))
    indexSim.save('./tmp/dic&corp/title/tfidfsim.index')
    indexSim = gensim.similarities.Similarity.load('./tmp/dic&corp/title/tfidfsim.index')

    trainID2Label = get_trainIDLabel(trainIDLabelPath)
    trainIDList = trainID2Label.keys()
    testIDList = get_validIDList(testIDPath)
    validIDList = testIDList

    allIDTitle = get_blogIDTitle('./tmp/dic&corp/segTitle.txt')
    show_num = 0
    with codecs.open('./tmp/test-relatedTitle.txt', 'w', 'utf-8') as fw:
        for validID in validIDList:
            blogTitle = get_blogTitle(validID, './tmp/allBlogContent-Test.txt')
            query = segTitle(blogTitle)
            query_bow = dictionary.doc2bow(query)
            query_tfidf = tfidf_model[query_bow]
            simList = enumerate(indexSim[query_tfidf])
            simIndexList = [id for (id, sim) in simList if sim > 0.95]
            blogIDList = [allIDTitle[index] for index in simIndexList if allIDTitle[index][0] != validID and allIDTitle[index][0] in trainIDList]
            print '{} Processing {} ...'.format(show_num, validID)
            if len(blogIDList) > 0:
                labelList = []
                blogIDList = [blogID for (blogID, blogTitle) in sorted(blogIDList, key=lambda x:x[0])]
                for blogID in blogIDList:
                    labelList.extend([word for word in trainID2Label[blogID].split(' ') if len(word) > 1])
                labelList = [word for (word, count) in collections.Counter(labelList).most_common()]
                fw.write(validID + u' ' + u'/'.join(blogIDList) + u' ' + u'/'.join(labelList) + '\n')
            else:
                fw.write(validID + u' ' + u'NULL' + '\n')
            show_num += 1
    # Compute Title Similarity

    relatedLabelFilter('./tmp/test-relatedTitle.txt', './tmp/result-test-Refine.txt')
