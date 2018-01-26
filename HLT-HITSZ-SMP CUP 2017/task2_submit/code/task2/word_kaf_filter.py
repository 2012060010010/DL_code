#coding:utf-8
"""
利用训练集用户文档，对词表进行卡方检验，过滤掉无效专业词语
"""
import pickle
from collections import Counter
import os
dirpath='../../SMP2017_data/SMPCUP2017'
wordpath = dirpath+'_2/cs_word_kaf/'
outpath = dirpath+'_2/cs_word_out/'
user_blog_match=dirpath+'_2/user_blog_match/usr_blogs_match_priority.pkl'
task2_train_file=dirpath+'_2/SMPCUP2017_TrainingData_Task2.txt'
cs_word_file=dirpath+"_2/cs_word_out/label2cs_word_nodup_34.pkl"
blog_content_dict_file=dirpath+"_2/task_blogs/blogID_content_cut_dict_34.pkl"

with open(user_blog_match,'r') as f:
    user_blog_dict=pickle.load(f)
#用户标签对应词典
user_label_dict = {}
with open(task2_train_file,'r') as f:
    for line in f:
        tmp = line.strip().split("\001")
        user_label_dict[tmp[0]] = tmp[1:]
user_list=user_blog_dict.keys()

with open(cs_word_file,'r') as f:
    cs_word_dict=pickle.load(f)
with open(blog_content_dict_file,'r') as f:
    blog_content_dict=pickle.load(f)
label_list=cs_word_dict.keys()
user_content_dict={}
for usr in user_list:
    blog_content_merge=[]
    for blog in user_blog_dict[usr]:
        blog_content_merge.extend(blog_content_dict[blog][0])
        blog_content_merge.extend(blog_content_dict[blog][2])
    user_content_dict[usr]=blog_content_merge
user_content_count_dict={}
for u in user_list:
    content=user_content_dict[u]
    word_count=Counter(content)
    word_set=set(content)
    user_content_count_dict[u]=(word_count,word_set)
label2user_po_ne={}
for lb in label_list:
    label2user_po_ne[lb]=([],[])
for u in user_list:
    ul=user_label_dict[u]
    for lb in label_list:
        if lb in ul:
            label2user_po_ne[lb][0].append(u)
        else:
            label2user_po_ne[lb][1].append(u)
cs_word_new_dict={}
for lb in label_list:
    word2kaf={}
    count=0
    for word in cs_word_dict[lb]:
        A=0
        B=0
        C=0
        D=0
        kaf=0
        for u in label2user_po_ne[lb][0]:
            if word in user_content_count_dict[u][1]:
                A+=1
        C=len(label2user_po_ne[lb][0])-A
        for u in label2user_po_ne[lb][1]:
            if word in user_content_count_dict[u][1]:
                B+=1
        D=len(label2user_po_ne[lb][1])-B
        if (A+C)*(A+B)*(B+D)*(B+C)==0:
            pass
        else:
            kaf=1055.0*(A*D-B*C)*(A*D-B*C)/((A+C)*(A+B)*(B+D)*(C+D))
        if A==0 and B==0:
            count+=1
        word2kaf[word]=kaf
    word2k=sorted(word2kaf.items(),key=lambda x:x[1],reverse=True)
    cs_word_new_dict[lb]=[it[0] for it in word2k if it[1]>0]
for lb in label_list:
    print lb,len(cs_word_new_dict[lb])
for lb in label_list:
    with open(wordpath+lb+'.txt','w') as f:
        f.write('\n'.join(cs_word_new_dict[lb]))
with open(outpath+"label2word_kaf_filter_34.pkl",'w') as f:
    pickle.dump(cs_word_new_dict,f)

def make_word_dict(dirpath,outpath):
    cs_word_dict={}
    for root, dirs, files in os.walk(dirpath):
        for f in files:
            print f
            label=f.split('.')[0]
            file = open(root + f, 'r')
            lines = file.readlines()
            file.close()
            word_list = []
            for line in lines:
                wo=line.strip()
                word_list.append(wo)
            cs_word_dict[label]=word_list
    with open(outpath + 'cs_word_34_simple.pkl', 'w') as ff:
        pickle.dump(cs_word_dict,ff)
make_word_dict(wordpath, outpath)