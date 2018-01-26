#coding: utf8
import pickle
import numpy as np
import time
dirpath="../../SMP2017_data/SMPCUP2017"
user_blog_match_file=dirpath+"_2/user_blog_match/usr_blogs_match_priority_forall.pkl"
task2_train_file=dirpath+"_2/SMPCUP2017_TrainingData_Task2.txt"
cs_word_file=dirpath+"_2/cs_word_out/cs_word_34_simple.pkl"
doc_content_cut_all=dirpath+"/Blogs_Content_cut_34.txt"

doc_content_cut_file=dirpath+"_2/task_blogs/follow_user_blogs_content_cut_34.txt"
doc_content_file=dirpath+"_2/task_blogs/followed_user_blogs_task2_for_3ops.txt"

blog_content_cut_dict=dirpath+"_2/task_blogs/blogID_content_cut_dict_34.pkl"

label_space_file=dirpath+'_2/LabelSpace_Task2.txt'
label_space_file_34=dirpath+'_2/part_LabelSpace_Task2.txt'
valid_file="../../SMP2017_data/SMPCUP2017_valid/SMPCUP2017_ValidationSet_Task2.txt"
usr_all_file='_2/user_blog_match/usr_ID_tvt.pkl'
doc2vec_file='_2/feature_doc/doc2vec_dbow_128.pkl'


def get_doc2content_cut_dict(doc_content_cut_file,doc_content_dict_file):
    """
    构建文档ID和文档内容的词典，doc_ID: （title,english,content）
    :param doc_content_file: 任务2分词后的文档集合
    :return: 对应文档词典
    """
    with open(doc_content_cut_file,'r') as f:
        lines=f.readlines()
    doc2content_dict={}
    for line in lines:
        items=line.strip().split('\001')
        doc_ID=items[0]
        title=items[1].split()
        english=items[2].split('\002')
        content=items[3].split()
        doc2content_dict[doc_ID]=(title,english,content)
    del lines
    # with open(doc_content_dict_file,'w') as f:
    #     pickle.dump(doc2content_dict,f)
    return doc2content_dict
doc2content_dict=get_doc2content_cut_dict(doc_content_cut_all,blog_content_cut_dict)

start=time.localtime()
def feature_usr2labels_34_102(doc2content_list,label_csword,usr_blog_match_file):
    """
    构建用户特征对应词典，usr_ID : feature
    :param doc2content_list:文档ID和内容对应词典
    :param label_csword: 标签与关键词对应词典
    :param 用户与博客对应词典，usr_ID : blog_list
    :return: 用户和标签对应关系,usrID:labels
    """
    # with open(doc2content_list,'r') as f:
    #     doc2content_dict=pickle.load(f)
    with open(label_csword,'r') as f:
        label_csword_dict=pickle.load(f)
    with open(usr_blog_match_file,'r') as f:
        usr_blog_match=pickle.load(f)
    with open(label_space_file_34,'r') as f:
        label_set=[line.split()[0] for line in f.readlines()]
    print ' '.join(label_set)
    usr_set=usr_blog_match.keys()
    usr2label_dict = {}
    for usr in usr_set:
        print usr
        blogs = usr_blog_match[usr]
        title = []
        english = []
        content = []
        for blog in blogs:
            title.extend(doc2content_dict[blog][0])
            english.extend(doc2content_dict[blog][1])
            content.extend(doc2content_dict[blog][2])
        english_set = set(english)
        content_set = set(content)
        label_num = []
        for l in range(len(label_set)):
            label_word = label_csword_dict[label_set[l]]
            title_num = 0
            english_num = 0
            content_num = 0
            for lb in range(len(label_word)):
                if label_word[lb] in title:
                    title_num += 1
                if label_word[lb] in english_set:
                    english_num += 1
                if label_word[lb] in content_set:
                    content_num += 1
            label_num.append(title_num)
            label_num.append(english_num)
            label_num.append(content_num)
        usr2label_dict[usr] = label_num
    with open(dirpath+"_2/feature_doc/usr_word_feature_102_all.pkl",'w') as f:
        pickle.dump(usr2label_dict,f)
    return usr2label_dict
feature_usr2labels_34_102(blog_content_cut_dict,cs_word_file,user_blog_match_file)
end=time.localtime()
print start
print end

def usr_doc2vec_avg_feature(doc2vec_file,usr_all_file):
    """
    获取每个用户的文档向量表示
    :param doc2vec_file: 文档向量文件，内容为doc_ID:docvec
    :param usr_all_file: 所有用户全集
    :return: 用户和文档向量对应词典，每个用户对应一个向量
    """
    usr_blog_match_file=dirpath + '_2/user_blog_match/all_usr_blogs_match_merge.pkl'
    with open(usr_blog_match_file,'r') as f:
        usr_blog_match=pickle.load(f)
    with open(dirpath+usr_all_file,'r') as f:
        usr_all_list=pickle.load(f)
    with open(dirpath+doc2vec_file,'r') as f:
        doc2vec=pickle.load(f)
    usr2docvec_dict={}
    print len(usr_all_list)
    for u in usr_all_list:
        blog_list=[]
        for bl in usr_blog_match[u]:
            blog_list.extend(bl)
        doc_Vector=[]
        for blog in blog_list:
            try:
                doc_Vector.append(doc2vec[blog])
            except KeyError:
                pass
        usr2docvec_dict[u]=np.mean(np.array(doc_Vector),0)
    with open(dirpath+'_2/feature_doc/user2docvec_dbow.pkl','w') as f:
        pickle.dump(usr2docvec_dict,f)
usr_doc2vec_avg_feature(doc2vec_file,usr_all_file)