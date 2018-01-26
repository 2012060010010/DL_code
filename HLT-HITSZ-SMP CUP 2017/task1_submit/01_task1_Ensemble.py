# coding=utf-8

import codecs
import re
import jieba
import jieba.analyse as analyse
import collections
from nltk.corpus import stopwords
import thulac

jieba.load_userdict('./tmp/wordDict7.22.txt')
jieba.enable_parallel(4)

thu_seg = thulac.thulac(user_dict='./tmp/wordDict7.22.txt', seg_only=True)

stops = list(stopwords.words("chinese"))
stops.extend(list(stopwords.words("english")))
stops = set(stops)

def get_blogIDList(IDPath):
    blogIDList = []
    with codecs.open(IDPath, 'r', 'utf-8') as fr:
        lines = fr.readlines()
        for line in lines:
            blogID = line.strip()
            blogIDList.append(blogID)
    return blogIDList

def write_allContent(blogIDList, sourcePath, targetPath):
    with codecs.open(sourcePath, 'r', 'utf-8') as fr, codecs.open(targetPath, 'w', 'utf-8') as fw:
        lines = fr.read().split('\r\n')[:-1]
        for line in lines:
            blogID = line.split('\001')[0].strip()
            if blogID in blogIDList:
                fw.write(line + '\n')

def get_blogContent(blogID, path):
    blogTitle = ''
    blogContent = ''
    with codecs.open(path, 'r', 'utf-8') as fr:
        lines = fr.read().split('\n')[:-1]
        for line in lines:
            _blogID = line.split('\001')[0].strip()
            if _blogID == blogID:
                blogTitle = line.split('\001')[1].strip()
                blogContent = line.split('\001')[2].strip()
                break
    return (blogTitle, blogContent)

def clean_blogContent(blogIDList):
    contentPath = './tmp/allBlogContent-Test.txt'
    pattern = u"[0-9\s\.\!\/_,$%^*()?；:\-【】\"\']+|[——！，;：。？、~@￥%……&*（）《》“”‘’]+"
    num = 0
    print 'Start Cleaning Blog ...'
    with codecs.open('./tmp/task1TestCleanBlog.txt', 'w', 'utf-8') as fw:
        for blogID in blogIDList:
            blogContent = get_blogContent(blogID, contentPath)
            blogContent = ''.join(blogContent)
            blog = re.sub(pattern, u' ', blogContent)
            try:
                segList = thu_seg.cut(blog.encode('utf-8').lower(), text=True).decode('utf-8').split(u' ')
            except Exception:
                segList = jieba.cut(blog.lower(), cut_all=False)
            wordList = [word.lower() for word in segList if word not in stops and len(word) > 1 and elementWiseCompare(word)]
            fw.write(blogID + ' ' + ' '.join(wordList) + '\n')
            if num % 10 == 0:
                print 'Finish {}'.format(num)
            num += 1

def jieba_keywords(blogContent):

    seg = analyse.extract_tags(blogContent, topK=10, withWeight=True)
    return [word for (word, weight) in seg]

def get_jieba_padKeywords(wordList, blogContent):

    seg = analyse.extract_tags(blogContent, withWeight=True)
    result = [(word, weight) for (word, weight) in seg if word not in wordList]
    result = sorted(result, key=lambda x: x[1], reverse=True)
    return [word for (word, weight) in result]

def get_jieba_topKeywords(wordList, blogContent):

    num = len(collections.Counter(blogContent.split(' ')).most_common())
    seg = analyse.extract_tags(blogContent, topK=num, withWeight=True)
    result = [(word, weight) for (word, weight) in seg if word in wordList]
    result = sorted(result, key=lambda x: x[1], reverse=True)
    return [word for (word, weight) in result]

def result_sort(resultList, smpLabelList):

    preferResult = [word for word in resultList if word in smpLabelList]
    candidateResult = [word for word in resultList if word not in smpLabelList]
    preferResult.extend(candidateResult)
    return preferResult

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
    else:
        refineList.extend(shortWordList) #2
    return refineList

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

if __name__ == '__main__':

    # Step 1 - Pre Process

    # 需要手动设置的 两个 路径
    blogIDPath = './tmp/SMPCUP2017_TestSet_Task1.txt' # 测试集ID路径
    sourcePath = './tmp/SMPCUP2017数据集/1_BlogContent.txt' # 所有博客内容路径
    # 需要手动设置的 两个 路径

    blogIDList = get_blogIDList(blogIDPath)
    targetPath = './tmp/allBlogContent-Test.txt'
    write_allContent(blogIDList, sourcePath, targetPath)
    clean_blogContent(blogIDList)
    # Step 1 - Pre Process

    # Step 2 - Get keywords from blogTitle
    pattern = u"[0-9\s\.\!\/_,$%^*()?；:\-【】\"\']+|[——！，;：。？、~@￥%……&*（）《》“”‘’]+"
    blogIDindex = 0
    with codecs.open('./tmp/result-test-Title.txt', 'w', 'utf-8') as fw:
        for blogID in blogIDList:
            print u'{} Processing {}'.format(blogIDindex, blogIDList[blogIDindex])
            blogTitle = get_blogContent(blogID, './tmp/allBlogContent-Test.txt')[0]
            blogTitle = re.sub(pattern, u' ', blogTitle)
            segList = jieba.cut(blogTitle.lower(), cut_all=True)
            wordSet = set([word.lower() for word in segList if word not in stops and len(word) > 1 and elementWiseCompare(word)])
            wordSet = set(wordListRefine(wordSet, 'title'))
            fw.write(blogID + ' ' + '/'.join(wordSet) + '\n')
            blogIDindex += 1
    # Step 2 - Get keywords from blogTitle

    # Step 3 - Ensemble
    blogID2Content = dict()
    with codecs.open('./tmp/task1TestCleanBlog.txt', 'r', 'utf-8') as fr:
        lines = fr.readlines()
        for line in lines:
            ID = line.strip().split(' ')[0]
            blog = ' '.join(line.strip().split(' ')[1:])
            blogID2Content[ID] = blog

    smpLabelList = [word.lower() for word in codecs.open('./tmp/smpLabel.txt', 'r', 'utf-8').read().split('\n')[:-1]]

    with codecs.open('./tmp/result-test-Title.txt', 'r', 'utf-8') as fr, codecs.open('./tmp/result-test-ensemble.txt', 'w', 'utf-8') as fw:
        lines = fr.readlines()
        blogIDindex = 0
        for line in lines:
            blogID = line.strip().split(' ')[0]
            if len(line.strip().split(' ')) == 2:
                blogTagList = (line.strip().split(' ')[1]).split('/')
                print u'{} Processing {}'.format(blogIDindex, blogID)
                blogContent = blogID2Content[blogID]
                keywords = jieba_keywords(blogContent)
                blogTagList = [word for word in blogTagList if word in keywords or word in smpLabelList]
                if len(blogTagList) > 2:
                    topKeywords = get_jieba_topKeywords(blogTagList, blogContent)
                    result = result_sort(topKeywords, smpLabelList)[:3]
                    if len(result) == 3:
                        fw.write(blogID + ',' + ','.join(result) + '\n')
                    if len(result) < 3:
                        padNum = 3 - len(result)
                        padKeywords = get_jieba_padKeywords(result, blogContent)
                        tmp = result_sort(padKeywords, smpLabelList)[:padNum]
                        result.extend(tmp)
                        fw.write(blogID + ',' + ','.join(result) + '\n')
                if len(blogTagList) == 2:
                    padKeywords = get_jieba_padKeywords(blogTagList, blogContent)
                    result = result_sort(padKeywords, smpLabelList)[:1]
                    blogTagList.extend(result)
                    resultStr = blogID + ',' + ','.join(blogTagList) + '\n'
                    fw.write(resultStr)
                if len(blogTagList) < 2:
                    padNum = 3 - len(blogTagList)
                    padKeywords = get_jieba_padKeywords(blogTagList, blogContent)
                    result = result_sort(padKeywords, smpLabelList)[:padNum]
                    blogTagList.extend(result)
                    resultStr = blogID + ',' + ','.join(blogTagList) + '\n'
                    fw.write(resultStr)
            else:
                blogContent = blogID2Content[blogID]
                padKeywords = get_jieba_padKeywords([], blogContent)
                resultStr = blogID + ',' + ','.join(padKeywords[:3]) + '\n'
                fw.write(resultStr)
            blogIDindex += 1
    # Step 3 - Ensemble