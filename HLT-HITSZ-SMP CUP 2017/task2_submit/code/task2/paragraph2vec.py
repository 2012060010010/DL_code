#coding:utf8
"""
利用所有分词后的文档训练文档向量，结果为每个文档一个128维的向量，存入pkl文件。
"""
import time
import gensim
import pickle
LabeledSentence = gensim.models.doc2vec.LabeledSentence

dirpath='../../SMP2017_data/SMPCUP2017'
blog_cut_file='/Blogs_Content_cut_34.txt'
import gensim
doc2vec_dbow='dbow/dbow_doc2vec.model'

reader = open(dirpath+blog_cut_file,'r')
count = 0
doc_content=[]
for line in reader:
    count+=1
    if count%10000==0:
        print count
    items = line.strip().split('\001')
    doc_ID = items[0]
    title = items[1].split()
    content = items[3].split()[:100]
    title.extend(content)
    if len(title)>10:
        doc_content.append(LabeledSentence(title,[doc_ID]))
print len(doc_content)
#train
start=time.localtime()
model_dbow = gensim.models.Doc2Vec(min_count=3,max_vocab_size=40000, window=10, size=128, sample=1e-3, negative=5, dm=0, workers=3)
model_dbow.build_vocab(doc_content)
model_dbow.train(doc_content, total_examples=model_dbow.corpus_count, epochs=10, start_alpha=0.025, end_alpha=0.015)
model_dbow.save(doc2vec_dbow)
end=time.localtime()
print start
print end

#merge to dict

model_dbow=gensim.models.Doc2Vec.load(doc2vec_dbow)
doc2vec_dbow_dict={}
for it in doc_content:
    doc2vec_dbow_dict[it.tags[0]] = model_dbow.docvecs[it.tags[0]]
print len(doc2vec_dbow_dict)
with open(dirpath+'_2/feature_doc/doc2vec_dbow_128.pkl','w') as f:
    pickle.dump(doc2vec_dbow_dict,f)
