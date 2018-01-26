# coding=utf-8

import codecs
import re
import jieba
from nltk.corpus import stopwords
import thulac

jieba.load_userdict('./wordDict7.22.txt')
jieba.enable_parallel(4)

thu_seg = thulac.thulac(user_dict='./wordDict7.22.txt', seg_only=True)

stops = list(stopwords.words("chinese"))
stops.extend(list(stopwords.words("english")))
stops = set(stops)

pattern = u"[0-9\s\.\!\/_,$%^*()?；:\-【】\"\']+|[——！，;：。？、~@￥%……&*（）\[\]《》“”‘’]+"

def segTitle(sourcePath, targetPath):
    lineNum = 0
    with codecs.open(sourcePath, 'r', 'utf-8') as fr, codecs.open(targetPath, 'w', 'utf-8') as fw:
        lines = fr.read().split('\n')[:-1]
        for line in lines:
            try:
                blogID = line.split('\001')[0].strip()
                blogTitle = line.split('\001')[1].strip()
                blog = blogTitle
                blog = re.sub(pattern, u' ', blog)
                segList = jieba.cut(blog.lower(), cut_all=False)
                wordSet = [word.lower() for word in segList if word != u' ']
                if len(wordSet) > 0:
                    fw.write(blogID + ' ' + '\001'.join(wordSet) + '\n')
            except Exception:
                print lineNum
            lineNum += 1
            if lineNum % 100 == 0:
                print 'Finish {}'.format(lineNum)

if __name__ == '__main__':

    # 需要手动设置的路径
    sourcePath = './tmp/SMPCUP2017数据集/1_BlogContent.txt' # 所有博客内容路径
    # 需要手动设置的路径

    targetPath = './tmp/dic&corp/segTitle.txt'
    segTitle(sourcePath, targetPath)