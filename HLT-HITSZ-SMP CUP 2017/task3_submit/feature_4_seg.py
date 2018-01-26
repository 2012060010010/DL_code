#encoding:utf-8
import jieba
import codecs
'''在服务器上执行多线程分词'''
jieba.enable_parallel(24)
jieba.set_dictionary('../data/seg_Dict_new.txt')


def load_stopwords():
    stopwords = []
    file_read = codecs.open('../data/stop_words_new.txt', 'r', 'utf-8')
    for line in file_read.readlines():
        stopwords.append(line.strip())
    return set(stopwords)


file_write = open('seg_word.txt', 'w')
stopwords = load_stopwords()
with open('../SMP2017/SMPCUP2017数据集/1_BlogContent.txt', 'r') as f:
    count = 0
    for line in f:
        # print(line)
        if count % 1000 == 0:
            print(count)
        line = (line.strip()).split('\001')
        text = [word for word in jieba.cut(line[1] + line[2]) if word not in stopwords]
        out = line[0] + '\001' + ' '.join(text) + '\n'
        file_write.write(out.encode('utf-8'))
        count += 1