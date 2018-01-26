#coding:utf-8
import pickle
import random
from collections import Counter
dirpath='../../SMP2017_data/SMPCUP2017'
blog_file='/1_BlogContent.txt'
blog_cut_file='/Blogs_Content_cut_34.txt'
post_file='/2_Post.txt'
browse_file='/3_Browse.txt'
comment_file='/4_Comment.txt'
voteup_file='/5_Vote-up.txt'
votedown_file='/6_Vote-down.txt'
favorite_file='/7_Favorite.txt'
follow_file='/8_Follow.txt'
letter_file='/9_Letter.txt'
task2_train_file='_2/SMPCUP2017_TrainingData_Task2.txt'
task2_label_file='_2/LabelSpace_Task2.txt'
task2_valid_file='_valid/SMPCUP2017_ValidationSet_Task2.txt'
usr_all_file='_2/user_blog_match/usr_ID_all.pkl'
follow_who_file='_2/user_blog_match/followed_who.pkl'
letter_who_file='_2/user_blog_match/lettered_who.pkl'

def get_all_user_list(usr_all_file):
    """
    获取所有可能存在的用户，返回用户ID列表
    :param usr_all_file: 输出文件名
    :return: 所有用户列表
    """
    usr_list=[]
    file_list=[post_file,
               browse_file,
               comment_file,
               voteup_file,
               votedown_file,
               favorite_file,
               follow_file,
               letter_file]
    for f in file_list[:6]:
        with open(dirpath+f,'r') as f:
            lines=f.readlines()
        for line in lines:
            usr=line.split('\001')[0]
            usr_list.append(usr)
    for f in file_list[6:]:
        with open(dirpath+f,'r') as f:
            lines=f.readlines()
        for line in lines:
            usr=line.strip().split('\001')[0]
            usr_list.append(usr[0])
            usr_list.append(usr[1])
    usr_list=list(set(usr_list))
    print len(usr_list)
    with open(dirpath+usr_all_file,'w') as f:
        pickle.dump(usr_list,f)
get_all_user_list(usr_all_file)

def ops_doc_ID_all_usr(list_ops_file):
    """
    获取所有用户对应各种操作（发布，收藏，评论等）文档列表，存入pkl
    :param list_ops_file:   用户对应的各种操作记录文件列表
    :return: 一个用户每种操作的对应字典，格式为 用户：文档编号list
    """
    for f in list_ops_file:
        print f
        ops_name=f.split('.')[0]
        #用户某种操作对应文档字典，比如post
        user_blogs_dict = {}
        post_ops = open(dirpath+f)
        lines = post_ops.readlines()
        for line in lines:
            item = line.strip().split('\001')
            try:
                user_blogs_dict[item[0]].append(item[1])
            except KeyError:
                user_blogs_dict[item[0]]=[item[1]]
        with open(dirpath+'_2/user_blog_match/'+ops_name+'_all.pkl','w') as f:
            pickle.dump(user_blogs_dict,f)
        post_ops.close()
ops_doc_ID_all_usr([post_file,browse_file,comment_file,favorite_file,votedown_file,voteup_file])
def follow_who():
    follow = open(dirpath + '/8_Follow.txt', 'r')
    letter = open(dirpath + '/9_Letter.txt', 'r')
    follow_who = {}
    letter_who = {}
    letter_lines = letter.readlines()
    follow_lines = follow.readlines()
    for line in follow_lines:
        items = line.split('\001')
        follow_who[items[0]] = []
        follow_who[items[1]] = []
    for line in follow_lines:
        items = line.split('\001')
        follow_who[items[1]].append(items[0])

    for line in letter_lines:
        items = line.split('\001')
        letter_who[items[0]] = []
        letter_who[items[1]] = []
    for line in letter_lines:
        items = line.split('\001')
        letter_who[items[0]].append(items[1])


    follow.close()
    letter.close()

    print len(follow_who.keys())
    print len(letter_who.keys())

    with open(dirpath + '_2/user_blog_match/follow_who.pkl', 'w') as f:
        pickle.dump(follow_who, f)
    with open(dirpath + '_2/user_blog_match/letter_who.pkl', 'w') as f:
        pickle.dump(letter_who, f)
follow_who()

def followed_by_who():
    follow = open(dirpath + '/8_Follow.txt', 'r')
    letter = open(dirpath + '/9_Letter.txt', 'r')
    followed_who = {}
    lettered_who = {}
    letter_lines = letter.readlines()
    follow_lines = follow.readlines()
    for line in follow_lines:
        items = line.split('\001')
        followed_who[items[0]] = []
        followed_who[items[1]] = []
    for line in follow_lines:
        items = line.split('\001')
        followed_who[items[0]].append(items[1])

    for line in letter_lines:
        items = line.split('\001')
        lettered_who[items[0]] = []
        lettered_who[items[1]] = []
    for line in letter_lines:
        items = line.split('\001')
        lettered_who[items[1]].append(items[0])


    follow.close()
    letter.close()

    print len(followed_who.keys())
    print len(lettered_who.keys())

    with open(dirpath + '_2/user_blog_match/followed_who.pkl', 'w') as f:
        pickle.dump(followed_who, f)
    with open(dirpath + '_2/user_blog_match/lettered_who.pkl', 'w') as f:
        pickle.dump(lettered_who, f)
followed_by_who()

def usr_blog_represent_priority(task2_train_file,list_ops_file):
    """
    获取用户表示，利用操作文档表示用户。优先级为
    发布（post），评论（comment），收藏（favorite），点赞（voteup），点踩（voteup），浏览（browse）
    对于没有记录的用户，利用他私信，关注的用户发布的文档替代
    :param task2_train_file: 训练集文件
    :param list_ops_file: 所有训练集用户各种操作的pkl字典
    :return: 结合各种操作的用户文本对应字典，加入了关注信息补全文档缺失用户
    """
    task2_train=open(dirpath+task2_train_file)
    task2_train_lines = task2_train.readlines()
    task2_user_ID = []
    # 问题2中训练集的user_ID
    for line in task2_train_lines:
        item = line.strip().split('\001')
        task2_user_ID.append(item[0])
    task2_train.close()
    usr_ops_dict={}   #用户操作记录，按照优先级排序
    for user in task2_user_ID:
        usr_ops_dict[user] = []
    for f in list_ops_file[:6]:
        print f
        with open(dirpath+'_2/user_blog_match/'+f,'r') as f:
           op_dict=pickle.load(f)
        for u in task2_user_ID:
            if len(usr_ops_dict[u])<20:
                try:
                    usr_ops_dict[u].extend(op_dict[u])
                except KeyError:
                    pass
    with open(dirpath + '_2/user_blog_match/follow_who.pkl', 'r') as f:
        follow_who = pickle.load(f)
    with open(dirpath + '_2/user_blog_match/letter_who.pkl', 'r') as f:
        letter_who = pickle.load(f)
    with open(dirpath + '_2/user_blog_match/2_Post_all.pkl', 'r') as f:
        usr_post = pickle.load(f)
    for u in task2_user_ID:
        if len(usr_ops_dict[u])<10:
            doc_list=[]
            # print u
            try:
                letter_list=letter_who[u]
                # print 'letter'
                for lu in letter_list:
                    try:
                        doc_list.extend(usr_post[lu])
                    except KeyError:
                        pass
            except KeyError:
                try:
                    follow_list=follow_who[u]
                    # print 'follow'
                    for fu in follow_list:
                        try:
                            doc_list.extend(usr_post[fu])
                        except KeyError:
                            pass
                except KeyError:
                    # print 'except'
                    pass
            doc_num_add=10-len(usr_ops_dict[u])
            random.shuffle(doc_list)
            usr_ops_dict[u].extend(doc_list[:doc_num_add+1])

    with open(dirpath+'_2/user_blog_match/usr_blogs_match_priority.pkl','w') as f:
        pickle.dump(usr_ops_dict,f)
    doc_len=[]
    for l in usr_ops_dict.values():
        doc_len.append(len(l))
    deg_count = sorted(Counter(doc_len).items(), key=lambda x: x[0])
    print deg_count

usr_blog_represent_priority(task2_train_file,['2_Post_all.pkl','4_Comment_all.pkl',
                                   '7_Favorite_all.pkl','5_Vote-up_all.pkl',
                                   '6_Vote-down_all.pkl','3_Browse_all.pkl'])
def usr_blog_represent_priority_forall(all_user_file,list_ops_file):
    """
    获取用户表示，利用操作文档表示用户。优先级为
    发布（post），评论（comment），收藏（favorite），点赞（voteup），点踩（voteup），浏览（browse）
    对于没有记录的用户，利用他私信，关注的用户发布的文档替代
    :param task2_train_file: 训练集文件
    :param list_ops_file: 所有可能用户各种操作的pkl字典
    :return: 结合各种操作的用户文本对应字典，加入了关注信息补全文档缺失用户
    """
    with open(dirpath+all_user_file,'r') as f:
        user_all_list=pickle.load(f)

    usr_ops_dict={}   #用户操作记录，按照优先级排序
    for user in user_all_list:
        usr_ops_dict[user] = []
    for f in list_ops_file[:6]:
        print f
        # ops_name=f.split('.')[0]
        with open(dirpath+'_2/user_blog_match/'+f,'r') as f:
           op_dict=pickle.load(f)
        for u in user_all_list:
            if len(usr_ops_dict[u])<20:
                try:
                    usr_ops_dict[u].extend(op_dict[u])
                except KeyError:
                    pass
    with open(dirpath + '_2/user_blog_match/follow_who.pkl', 'r') as f:
        follow_who = pickle.load(f)
    with open(dirpath + '_2/user_blog_match/letter_who.pkl', 'r') as f:
        letter_who = pickle.load(f)
    with open(dirpath + '_2/user_blog_match/2_Post_all.pkl', 'r') as f:
        usr_post = pickle.load(f)
    for ind,u in enumerate(user_all_list):
        if ind%10000==1:
            print ind
        if len(usr_ops_dict[u])<10:
            doc_list=[]
            # print u
            try:
                letter_list=letter_who[u]
                # print 'letter'
                for lu in letter_list:
                    try:
                        doc_list.extend(usr_post[lu])
                    except KeyError:
                        pass
            except KeyError:
                try:
                    follow_list=follow_who[u]
                    # print 'follow'
                    for fu in follow_list:
                        try:
                            doc_list.extend(usr_post[fu])
                        except KeyError:
                            pass
                except KeyError:
                    # print 'except'
                    pass
            doc_num_add=10-len(usr_ops_dict[u])
            random.shuffle(doc_list)
            usr_ops_dict[u].extend(doc_list[:doc_num_add+1])

    with open(dirpath+'_2/user_blog_match/usr_blogs_match_priority_forall.pkl','w') as f:
        pickle.dump(usr_ops_dict,f)
    doc_len=[]
    for l in usr_ops_dict.values():
        doc_len.append(len(l))
    deg_count = sorted(Counter(doc_len).items(), key=lambda x: x[0])
    print deg_count
usr_blog_represent_priority_forall(usr_all_file,['2_Post_all.pkl','4_Comment_all.pkl',
                                   '7_Favorite_all.pkl','5_Vote-up_all.pkl',
                                   '6_Vote-down_all.pkl','3_Browse_all.pkl'])

def usr_blog_for_allops(usr_list_file, list_ops_file):
    """
    获取所有用户所有各种操作，利用操作文档表示用户。优先级为
    发布（post），评论（comment），收藏（favorite），点赞（voteup），点踩（voteup），浏览（browse）
    对于没有记录的用户，利用他私信，关注的用户发布的文档替代
    :param usr_list_file: 用户列表文件
    :param list_ops_file: 各种操作的pkl字典
    :return: 结合各种操作的用户文本对应字典,一共6种，每个用户对应6个列表，不存在的操作列表为空。
    """
    with open(dirpath+usr_list_file,'r') as f:
        user_ID=pickle.load(f)

    usr_ops_dict = {}  # 用户操作记录，按照优先级排序
    for user in user_ID:
        usr_ops_dict[user] = []
    for f in list_ops_file[:6]:
        print f
        # ops_name=f.split('.')[0]
        with open(dirpath + '_2/user_blog_match/' + f, 'r') as f:
            op_dict = pickle.load(f)
        for u in user_ID:
            try:
                usr_ops_dict[u].append(op_dict[u])
            except KeyError:
                usr_ops_dict[u].append([])

    with open(dirpath + '_2/user_blog_match/all_usr_blogs_match_merge.pkl', 'w') as f:
        pickle.dump(usr_ops_dict, f)
usr_blog_for_allops(usr_all_file,['2_Post_all.pkl','4_Comment_all.pkl',
                                   '7_Favorite_all.pkl','5_Vote-up_all.pkl',
                                   '6_Vote-down_all.pkl','3_Browse_all.pkl'])

def get_blog_content_cut(outfile,usr_blog_dict_file):
    """
    从分词文档全集中提取记录对应的文档,写入一个小文件
    :param outfile: 输出文件名，存储任务用户相关切分后文档
    :param usr_blog_dict: 用户和文档列表(各操作列表的列表)记录，user_ID: [ops_list]
    :return:任务二用户相关切分文档全集
    """
    blog_cut=open(dirpath+blog_cut_file,'r')
    outfile = open(outfile, "w")
    blogs_all = blog_cut.readlines()
    with open(dirpath+usr_blog_dict_file,'r') as f:
        usr_blog_dict=pickle.load(f)
    train_user=[]
    with open(task2_train_file) as f:
        for line in f:
            train_user.append(line.strip().split('\001')[0])
    post_blog=[]
    for usr in train_user:
        post_blog.extend(usr_blog_dict[usr])
    post_blog=list(set(post_blog))
    print len(post_blog)
    for ind,line in enumerate(blogs_all):
        if ind%10000==0:
            print ind
        blog_ID, title,english, content = line.strip().split('\001')
        if blog_ID in post_blog:
            outfile.write(line)
    outfile.close()
    blog_cut.close()

usr_ops_dict_file='_2/user_blog_match/usr_blogs_match_priority.pkl'
get_blog_content_cut(dirpath+"_2/task_blogs/user_blogs_content_cut_34.txt",usr_ops_dict_file)

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
    with open(doc_content_dict_file,'w') as f:
        pickle.dump(doc2content_dict,f)
    return doc2content_dict
get_doc2content_cut_dict(dirpath+"_2/task_blogs/user_blogs_content_cut_34.txt",
                         dirpath+"_2/task_blogs/blogID_content_cut_dict_34.pkl")