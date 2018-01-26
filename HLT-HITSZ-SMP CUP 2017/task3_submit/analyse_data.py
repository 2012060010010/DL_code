#coding:utf-8
import pandas as pd
import codecs
from collections import defaultdict
import networkx as nx
import pickle
import jieba
from task3_code.LR import load


# 发信人和收信人互换（feature4  做准备）
def change_letter():
    file_read = codecs.open('../SMP2017/SMPCUP2017数据集/9_Letter.txt','r','utf-8')
    file_write = codecs.open('../task3_feature_first/letter(reverse).txt','w','utf-8')
    for line in file_read:
        line = line.split(u'\001')
        file_write.writelines(line[1]+u'\001'+line[0]+'\001'+line[2])
    file_write.close()
    file_read.close()

# 统计每一用户每月的发稿数量
def analyse_post(filename,filewrite):
    df_post = pd.read_csv(filename,sep=u'\001',header=None,names=['userid','blogid','time'],index_col='userid')
    df_post['count'] = 1            #统计某用户一年发稿数量
    df_post = df_post.sort_values(by='time',ascending=True)   # sort by time
    df_post['time'] = pd.to_datetime(df_post['time'])

    f = lambda x : x.strftime('%B')

    df_post['month'] = df_post['time'].apply(f)
    df_group_month_id = df_post.groupby(['userid','month']).sum()

    df_group_month_id.to_csv(filewrite)

# 将每一用户一年发稿数量 转化为  1*12 向量
def trans_vector(filename,filewrite):
    df_read = codecs.open(filename,'r','utf-8')
    file_write = codecs.open(filewrite,'w','utf-8')
    # 保存用户 向量
    user_dict = dict()
    month_dict = {'January':0, 'February':1, 'March':2, 'April':3, 'May':4, 'June':5, 'July':6, 'August':7,
                  'September':8, 'October':9, 'November':10, 'December':11}
    for line in df_read:
        line = line.strip()
        line = line.split(',')
        if line[0] not in user_dict:
            user_dict[line[0]] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        user_dict[line[0]][month_dict[line[1]]] = line[2]

    for key,value in user_dict.items():
        value = list(map(str,value))
        file_write.writelines(key+'\001'+'\001'.join(value)+'\n')
    file_write.close()
    df_read.close()

# 将 文件读取进来， 转化为 user：vector 的字典
def get_dict(filename):
    file_read =  codecs.open(filename,'r','utf-8')
    user_dict = dict()
    for line in file_read:
        line = line.strip()
        line = line.split(u'\001')
        user_dict[line[0]] = line[1:]
    file_read.close()
    return user_dict

# 把所有特征连接起来，某一特征不存在的用户 补0
def concat_vector(filewrite):
    file_write = codecs.open(filewrite,'w','utf-8')
    post_dict = get_dict('../task3_feature_first/post_vector.txt')
    browse_dict = get_dict('../task3_feature_first/browse_vector.txt')
    comment_dict = get_dict('../task3_feature_first/comment_vector.txt')
    voteup_dict = get_dict('../task3_feature_first/voteup_vector.txt')
    votedown_dict = get_dict('../task3_feature_first/votedown_vector.txt')
    favorite_dict = get_dict('../task3_feature_first/favorite_vector.txt')
    letter_dict = get_dict('../task3_feature_first/letter_vector.txt')
    user_id = list(set(list(post_dict.keys())+list(browse_dict.keys())+list(comment_dict.keys())+list(voteup_dict.keys())+list(votedown_dict.keys())+list(favorite_dict.keys())+list(letter_dict.keys())))
    print(len(user_id))
    user_dict = dict()
    for userid in user_id:
        if userid in post_dict:
            user_dict[userid] = post_dict[userid]
        else: user_dict[userid] = [0,0,0,0,0,0,0,0,0,0,0,0]
        if userid in browse_dict:
            user_dict[userid] += browse_dict[userid]
        else:user_dict[userid] += [0,0,0,0,0,0,0,0,0,0,0,0]
        if userid in comment_dict:
            user_dict[userid] += comment_dict[userid]
        else:user_dict[userid] += [0,0,0,0,0,0,0,0,0,0,0,0]
        if userid in voteup_dict:
            user_dict[userid] += voteup_dict[userid]
        else:user_dict[userid] += [0,0,0,0,0,0,0,0,0,0,0,0]
        if userid in votedown_dict:
            user_dict[userid] += votedown_dict[userid]
        else:user_dict[userid] += [0,0,0,0,0,0,0,0,0,0,0,0]
        if userid in favorite_dict:
            user_dict[userid] += favorite_dict[userid]
        else:user_dict[userid] += [0,0,0,0,0,0,0,0,0,0,0,0]
        if userid in letter_dict:
            user_dict[userid] += letter_dict[userid]
        else:user_dict[userid] += [0,0,0,0,0,0,0,0,0,0,0,0]
    #print(user_dict['U0045528'])
    user_dict = sorted(user_dict.items(),key=lambda x:x[0])
    print(len(user_dict))
    count = 0
    for value in user_dict:
        user_list = value[1]
        user_list = list(map(str,user_list))
        count += 1
        file_write.writelines(value[0]+','+','.join(user_list)+'\n')
    file_write.close()

# 统计每一用户的  粉丝数 + 关注人数
def count_fans(filename):
    file_follow = codecs.open(filename,'r','utf-8')
    followed = defaultdict(int)
    follow = defaultdict(int)
    user_dict = dict()
    for line in file_follow:
        line = line.strip()
        line = line.split(u'\001')
        followed[line[0]] += 1
        follow[line[1]] +=1
    for user,value in followed.items():
        user_list = []
        user_list.append(value)
        if user in follow:
            user_list.append(follow[user])
        else:user_list.append(0)
        user_dict[user] = user_list
    for user,value in follow.items():
        if user not in user_dict:
            user_dict[user] = [0,user]
    return user_dict


def user_follow_vector():
    follow_dict = count_fans('../SMP2017/SMPCUP2017数据集/8_Follow.txt')
    letter_dict = count_fans('../SMP2017/SMPCUP2017数据集/9_Letter.txt')
    file_write = codecs.open('../task3_feature_medium/user_follow_vector.txt', 'w', 'utf-8')
    user_dict = dict()
    for user, value in follow_dict.items():
        if user in letter_dict:
            user_dict[user] = [value[0] + letter_dict[user][0] + letter_dict[user][1],
                               value[1] + letter_dict[user][0] + letter_dict[user][1]]
        else:
            user_dict[user] = [value[0], value[1]]
    for key, value in user_dict.items():
        file_write.writelines(key + u'\001' + str(value[0]) + u'\001' + str(value[1]) + '\n')



# 统计每一用户粉丝的最大粉丝数（基于社交网络分析中 被影响力大的人关注的人影响力也比较大，此特征用来表述用户的重要程度）
def extract_fans_second_fans(filename,filewrite):
    file_read = codecs.open(filename,'r','utf-8')
    file_write = codecs.open(filewrite,'w','utf-8')   #-------------1
    #file_write = codecs.open('../task3_feature_medium/fans_mean_fans_num.txt','w','utf-8') #-------------------2
    node_file = codecs.open('../task3_feature_medium/all_user_vector.txt','r','utf-8')
    user_id = []
    D = nx.DiGraph()
    for line in node_file:
        line = line.strip()
        line = line.split(',')
        D.add_node(line[0])
        user_id.append(line[0])
    edges_list = []
    # line[0] be followed   line[1] following
    for line in file_read:
        line = line.strip()
        line = line.split(u'\001')
        edges_list.append([line[1],line[0],1])
    D.add_weighted_edges_from(edges_list)

    for user in user_id:
        fan_fans_num = 0
        neighbors = list(nx.all_neighbors(D,user))
        out_neighbors = list(D.neighbors(user))
        fans = [u for u in neighbors if u not in out_neighbors]   # 用户的一阶粉丝
        if len(fans)==0:    #如果用户没有入度 ： 粉丝
            file_write.writelines(user + '\001' + str(fan_fans_num) + '\n')
            continue
        fans_fansnumber = []
        for f in fans:
            fans_fansnumber.append(D.in_degree(f))
        #mean_fans_fansnum = sum(fans_fansnumber)/len(fans_fansnumber)#-----------2
        fan_fans_num = max(fans_fansnumber)
        file_write.writelines(user+'\001'+str(fan_fans_num)+'\n')# -------------1

        #file_write.writelines(user+'\001'+str(mean_fans_fansnum)+'\n')#-------2


# 统计 博客 被浏览或者评论等 某种操作的次数
def blog_analyse(filename):
    file_read = codecs.open(filename,'r','utf-8')
    blog_dict = defaultdict(int)
    for line in file_read:
        line = line.strip()
        line = line.split(u'\001')
        blog_dict[line[1]] += 1
    return blog_dict

# 统计每一用户的发表博客的：（总、最大、平均）浏览数，评论数，点赞数，点踩数，被收藏数
def write_feature3_to_file():
    browse_dict = blog_analyse('../SMP2017/SMPCUP2017数据集/3_Browse.txt')
    comment_dict = blog_analyse('../SMP2017/SMPCUP2017数据集/4_Comment.txt')
    voteup_dict = blog_analyse('../SMP2017/SMPCUP2017数据集/5_Vote-up.txt')
    votedown_dict = blog_analyse('../SMP2017/SMPCUP2017数据集/6_Vote-down.txt')
    favorite_dict = blog_analyse('../SMP2017/SMPCUP2017数据集/7_Favorite.txt')
    user_post_blog_dict = pickle.load(open('../task3_statistic_data/user_blog_match/user_blog_post_match.pkl','rb'))
    '''
    user_post_blog_dict
    {user_id:[[blog_id,browse_num,comment_num,voteup_num,votedown_num,favorite num],[blog_id,browse_num,comment_num,voteup_num,votedown_num,favorite num]]}
    '''
    #  key: userid    value: blog id(list)
    for key,value in user_post_blog_dict.items():
        user_vector = []
        for blogid in value:
            blog_list = []
            blog_list.append(blogid)
            if blogid in browse_dict:
                blog_list.append(browse_dict[blogid])
            else:blog_list.append(0)
            if blogid in comment_dict:
                blog_list.append(comment_dict[blogid])
            else:blog_list.append(0)
            if blogid in voteup_dict:
                blog_list.append(voteup_dict[blogid])
            else:blog_list.append(0)
            if blogid in votedown_dict:
                blog_list.append(votedown_dict[blogid])
            else:blog_list.append(0)
            user_vector.append(blog_list)
            if blogid in favorite_dict:
                blog_list.append(favorite_dict[blogid])
            else:blog_list.append(0)
        user_post_blog_dict[key] = user_vector
    user_final_dict = dict()
    for key,value in user_post_blog_dict.items():
        sum_browse_num = 0
        max_browse_num = 0
        mean_browse_num = 0
        sum_comment_num = 0
        max_comment_num = 0
        mean_comment_num = 0
        sum_vote_updown_num = 0
        sum_voteup_num = 0
        max_voteup_num= 0
        mean_voteup_num = 0
        sum_votedown_num = 0
        max_votedown_num = 0
        mean_votedown_num = 0
        sum_vote_updown_num = 0
        sum_favorite_num = 0
        max_favorite_num = 0
        mean_favorite_num = 0
        for blog_vector in value:
            sum_browse_num = sum_browse_num + blog_vector[1]
            sum_comment_num = sum_comment_num + blog_vector[2]
            sum_voteup_num = sum_voteup_num + blog_vector[3]
            sum_votedown_num = sum_votedown_num + blog_vector[4]
            sum_favorite_num = sum_favorite_num + blog_vector[5]
            if blog_vector[1] > max_browse_num:
                max_browse_num = blog_vector[1]
            if blog_vector[2] > max_comment_num:
                max_comment_num = blog_vector[2]
            if blog_vector[3] > max_voteup_num:
                max_voteup_num = blog_vector[3]
            if blog_vector[4] > max_votedown_num:
                max_votedown_num = blog_vector[4]
            if blog_vector[5] > max_favorite_num:
                max_favorite_num = blog_vector[5]
        mean_browse_num = sum_browse_num/(len(value)+1)
        mean_comment_num = sum_comment_num/(len(value)+1)
        mean_voteup_num = sum_voteup_num/(len(value)+1)
        mean_votedown_num = sum_votedown_num/(len(value)+1)
        mean_vote_updown_num = (sum_votedown_num+sum_voteup_num)/(len(value)+1)
        mean_favorite_num = sum_favorite_num / (len(value) + 1)
        user_final_dict[key] = [sum_browse_num,max_browse_num,mean_browse_num,sum_comment_num,max_comment_num,mean_comment_num,
                                sum_voteup_num,max_voteup_num,mean_voteup_num,sum_votedown_num,max_votedown_num,mean_votedown_num,
                                mean_vote_updown_num,sum_favorite_num,max_favorite_num,mean_favorite_num]


    file_write = codecs.open('../task3_feature_first/task3_feature_3.txt','w','utf-8')
    for key,value in user_final_dict.items():
        file_write.writelines(key+u'\001'+'\001'.join(list(map(str,value)))+'\n')

# 每个用户博客的最大/最小/平均  长度
#粉丝博客的最大/最小/平均长度
# 关注用户博客的最大/最小/平均 长度
# 私信用户博客的最大/最小/平均长度

def get_useful_id():
    edge_read = codecs.open('../SMP2017/SMPCUP2017Data/8_Follow.txt', 'r', 'utf-8')
    node_file = codecs.open('../task3_feature_first/all_user_vector.txt', 'r', 'utf-8')
    # train 和valid 的 user id
    task3_train_user_file = codecs.open('../SMP2017/Task3Train/SMPCUP2017_TrainingData_Task3.txt', 'r', 'utf-8')
    task3_valid_user_file = codecs.open('../valid/SMPCUP2017_ValidationSet_Task3.txt', 'r', 'utf-8')

    users = []

    for line in task3_train_user_file:
        line = line.strip()
        line = line.split(u'\001')
        users.append(line[0])

    for line in task3_valid_user_file:
        line = line.strip()
        line = line.split(u'\001')
        users.append(line[0])
    # 根据关注关系  建好图
    D = nx.DiGraph()
    for line in node_file:
        line = line.strip()
        line = line.split(',')
        D.add_node(line[0])
    edges_list = []
    # line[0] be followed   line[1] following
    for line in edge_read:
        line = line.strip()
        line = line.split(u'\001')
        edges_list.append([line[1], line[0], 1])
    D.add_weighted_edges_from(edges_list)
    for user in users:
        neighbors = list(nx.all_neighbors(D, user))
        users.append(neighbors)
    return set(users)


def load_stopwords(filename):
    stop_words = codecs.open(filename, 'r', 'utf-8')
    stopwords = []
    for line in stop_words:
        line = line.strip()
        stopwords.append(line)
    stopwords = set(stopwords)
    stop_words.close()
    return stopwords


def load_alluser(filename):
    # user_id_file = codecs.open('../task3_feature_first/all_user_vector.txt', 'r', 'utf-8')
    user_id_file = open(filename, 'r')
    all_user = []
    for line in user_id_file:
        line = (line.strip()).split(',')
        all_user.append(line[0])
    return all_user


def make_document_dict(filename):
    document_dict = dict()  # id,分词后的原文
    # all_text_seg = codecs.open('../data/ContentWords_2.txt', 'r', 'utf-8')
    all_text_seg = codecs.open(filename, 'r')
    i = 0
    for line in all_text_seg:
        line = (line.strip()).split('\001')
        words = line[1].split()
        document_dict[line[0]] = words
        i += 1
    all_text_seg.close()
    return document_dict


def extract_all(stopwords, all_user, document_dict):  # load stop words ,加载所有用户i

    # 私信关系
    letter_file = codecs.open('../SMP2017/SMPCUP2017Data/9_Letter.txt', 'r', 'utf-8')
    letter_dict = dict()
    for line in letter_file:
        line = line.strip()
        line = line.split(u'\001')
        if line[0] not in letter_dict:
            letter_dict[line[0]] = [line[1]]
        else:
            letter_dict[line[0]].append(line[1])
    print('********\nletter', len(letter_dict))
    letter_file.close()
    # 关注关系
    edge_read = codecs.open('../SMP2017/SMPCUP2017数据集/8_Follow.txt', 'r', 'utf-8')
    node_file = codecs.open('../task3_feature_first/all_user_vector.txt', 'r', 'utf-8')
    # train 和valid 的 user id
    task3_train_user_file = codecs.open('../SMP2017/任务3训练集/SMPCUP2017_TrainingData_Task3.txt', 'r', 'utf-8')
    task3_valid_user_file = codecs.open('../valid/SMPCUP2017_ValidationSet_Task3.txt', 'r', 'utf-8')

    task3_train_user = []
    task3_valid_user = []

    for line in task3_train_user_file:
        line = line.strip()
        line = line.split('\001')
        task3_train_user.append(line[0])

    for line in task3_valid_user_file:
        line = line.strip()
        line = line.split('\001')
        task3_valid_user.append(line[0])
    # 根据关注关系  建好图
    user_id = []
    D = nx.DiGraph()
    for line in node_file:
        line = line.strip()
        line = line.split(',')
        D.add_node(line[0])
        user_id.append(line[0])
    edges_list = []
    # line[0] be followed   line[1] following
    for line in edge_read:
        line = line.strip()
        line = line.split('\001')
        edges_list.append([line[1], line[0], 1])
    D.add_weighted_edges_from(edges_list)

    # 用户+post的文档id
    user_post_dict = pickle.load(open('../task3_statistic_data/user_blog_match/user_blog_post_match.pkl', 'rb'))
    # 粉丝 + post文档id
    user_fans_dict = dict()
    user_follow_dict = dict()
    for user in task3_train_user:
        neighbors = list(nx.all_neighbors(D, user))
        out_neighbors = list(D.neighbors(user))
        fans = [u for u in neighbors if u not in out_neighbors]  # 可能为0
        user_fans_dict[user] = fans
        user_follow_dict[user] = out_neighbors

    print('user follow dict', len(user_follow_dict))
    # 字典全部建好，开始  map 用户与粉丝等的blog id的关系
    user_fans_post_id = dict()  # {user_id: [[fans post id],[],[]]}
    for key, value in user_fans_dict.items():
        if len(value) != 0:
            blog_vector = []
            for user in value:
                if user in user_post_dict:
                    blog_vector.extend(user_post_dict[user])
            user_fans_post_id[key] = blog_vector
        else:
            continue
    user_follow_post_id = dict()
    for key, value in user_follow_dict.items():
        if len(value) != 0:
            blog_vector = []
            for user in value:
                if user in user_post_dict:
                    blog_vector.extend(user_post_dict[user])
            user_follow_dict[key] = blog_vector
        else:
            continue
    user_letter_post_id = dict()
    for key, value in letter_dict.items():
        if len(value) != 0:
            blog_vector = []
            for user in value:
                if user in user_post_dict:
                    blog_vector.extend(user_post_dict[user])
            user_letter_post_id[key] = blog_vector
        else:
            continue

    # 开始计算文档长度
    user_final_dict = defaultdict(int)
    for user in all_user:
        user_vector = []
        if user in user_post_dict:
            if len(user_post_dict[user]) == 0:
                user_vector.extend([0, 0, 0])  # max,min,mean ,max(del stop),min(del stop),mean(del stop)
            else:
                max_len, min_len, mean_len = 0, 0, 0
                count = 0
                for blog in user_post_dict[user]:
                    count += 1
                    document_len = len(document_dict[blog])
                    # document_del_len = len(document_del_stop_dict[blog])
                    if document_len > max_len:
                        max_len = document_len
                    if document_len < min_len:
                        min_len = document_len
                    mean_len += document_len
                # if document_del_len > max_del_stop_len:
                #                         max_del_stop_len = document_del_len
                #                     if document_del_len < min_del_stop_len:
                #                         min_del_stop_len = document_del_len
                #                     mean_del_stop_len += document_del_len
                mean_len = mean_len / (count + 1)
                #                 mean_del_stop_len = mean_del_stop_len/(count + 1)
                user_vector.extend([max_len, min_len, mean_len])
        else:
            user_vector.extend([0, 0, 0])
        if user in user_fans_post_id:
            if len(user_fans_post_id[user]) == 0:
                user_vector.extend([0, 0, 0])
            else:
                max_len, min_len, mean_len = 0, 0, 0
                count = 0
                for blog in user_fans_post_id[user]:
                    count += 1
                    document_len = len(document_dict[blog])
                    # document_del_len = len(document_del_stop_dict[blog])
                    if document_len > max_len:
                        max_len = document_len
                    if document_len < min_len:
                        min_len = document_len
                    mean_len += document_len
                # if document_del_len > max_del_stop_len:
                #                         max_del_stop_len = document_del_len
                #                     if document_del_len < min_del_stop_len:
                #                         min_del_stop_len = document_del_len
                #                     mean_del_stop_len += document_del_len
                mean_len = mean_len / (count + 1)
                #                 mean_del_stop_len = mean_del_stop_len / (count + 1)
                user_vector.extend([max_len, min_len, mean_len])
        else:
            user_vector.extend([0, 0, 0])
        if user in user_follow_post_id:
            if len(user_follow_post_id[user]) == 0:
                user_vector.extend([0, 0, 0])
            else:
                max_len, min_len, mean_len = 0, 0, 0
                count = 0
                for blog in user_follow_dict[user]:
                    count += 1
                    document_len = len(document_dict[blog])
                    # document_del_len = len(document_del_stop_dict[blog])
                    if document_len > max_len:
                        max_len = document_len
                    if document_len < min_len:
                        min_len = document_len
                    mean_len += document_len
                # if document_del_len > max_del_stop_len:
                #                         max_del_stop_len = document_del_len
                #                     if document_del_len < min_del_stop_len:
                #                         min_del_stop_len = document_del_len
                #                     mean_del_stop_len += document_del_len
                mean_len = mean_len / (count + 1)
                #                 mean_del_stop_len = mean_del_stop_len / (count + 1)
                user_vector.extend([max_len, min_len, mean_len])
        else:
            user_vector.extend([0, 0, 0])
        if user in user_letter_post_id:
            if len(user_letter_post_id[user]) == 0:
                user_vector.extend([0, 0, 0])
            else:
                max_len, min_len, mean_len = 0, 0, 0
                count = 0
                for blog in user_letter_post_id[user]:
                    print('blog', blog)
                    print(blog in document_dict)
                    count += 1
                    document_len = len(document_dict[blog])
                    # document_del_len = len(document_del_stop_dict[blog])
                    if document_len > max_len:
                        max_len = document_len
                    if document_len < min_len:
                        min_len = document_len
                    mean_len += document_len
                # if document_del_len > max_del_stop_len:
                #                         max_del_stop_len = document_del_len
                #                     if document_del_len < min_del_stop_len:
                #                         min_del_stop_len = document_del_len
                #                     mean_del_stop_len += document_del_len
                mean_len = mean_len / (count + 1)
                #                 mean_del_stop_len = mean_del_stop_len / (count + 1)
                user_vector.extend([max_len, min_len, mean_len])
        else:
            user_vector.extend([0, 0, 0])
        user_final_dict[user] = user_vector

    file_write = codecs.open('../task3_feature_first/task3_feature_4.txt', 'w', 'utf-8')
    for key, value in user_final_dict.items():
        print(key, value)
        file_write.writelines(key + '\001' + '\001'.join(list(map(str, value))) + '\n')
    file_write.close()

# 每个用户 赞成票与反对票之比  *12
#         浏览数与评论数之比  *12
#         私信数与粉丝数之比
#         私信数与关注数之比
def extract_ratio():
    file_original = load('../task3_feature_final/all_user_dict.pkl')
    letter = codecs.open('../task3_feature_first/letter(reverse)_vector.txt','r','utf-8')
    file_280 = pickle.load(open('../task3_feature_final/all_user_dict.pkl','rb'))
    file_280_add = open('../task3_feature_final/all_user_dict(280+).pkl','wb')        #  318维度


    # 统计用户收到的私信数
    letter_dict = dict()
    for line in letter:
        line = line.split(u'\001')
        letter_dict[line[0]] = list(map(float,line[1:]))

    user_dict = dict()
    for key,value in file_original.items():
        value = list(map(float,value))
       # print(len(value))
        user_dict[key] = value
    for key,value in user_dict.items():
        vector = []
        for i in range(36,48):
            vector.append(value[i]/(value[i+12]+1))
        for i in range(12,24):
            vector.append(value[i]/(value[i+12]+1))
        sum_letter = 0

        # 私信 与  关注数 之比
        for i in range(72,84):
            sum_letter += value[i]
        vector.append(sum_letter/(value[85]+1))
        # 收到的私信 与 粉丝数 之比
        sum_lettered = 0
        if key in letter_dict:
            for i in letter_dict[key]:
                sum_lettered += i
        vector.append(sum_lettered/(value[84]+1))

        user_dict[key] = vector
        print(len(vector))
    for key ,value in file_280.items():
        if key in user_dict:
            file_280[key] = file_280[key] + user_dict[key]
        else:file_280[key] = file_280[key] + [0]*26
       # print(len(file_280[key]))

    pickle.dump(file_280,file_280_add)


#统计趋势特征  0-83维  task3_feature_5.txt
def trend_information():
    all_user_dict = pickle.load(open('../task3_feature_final/all_user_dict(280+).pkl','rb'))
    file_write = open('../Task3_feature_final/all_user_dict(395).pkl','wb')

    for key,value in all_user_dict.items():
        value = list(map(float,value))
        #print(len(value))
        i = 0
        vector = []
        while(i*12 <= 72):
            for j in range(i*12,(i+1)*12-1):
               # print(j)
                vector.append(value[j+1] - value[j])
            i += 1
        #print(len(vector))
        all_user_dict[key].extend(vector)
    # for key,value in all_user_dict.items():
    #     file_write.writelines(key+u'\001'+u'\001'.join(list(map(str,value)))+'\n')  # 395维度
    pickle.dump(all_user_dict,file_write)





#***********first step************************
#change_letter()  # feature4

#***********second step ***********************
#analyse_post('../task3_feature_first/letter(reverse).txt','../task3_feature_first/letter(reverse)_raw_group_data.csv')
#trans_vector('../task3_feature_first/letter(reverse)_raw_group_data.csv','../task3_feature_first/letter(reverse)_vector.txt')
#concat_vector('../task3_feature_first/all_user_vector.txt')

#***********third step*************************
#user_follow_vector()

#***********fourth step*************************
#add feature: fans' max fans number    -----------------1
#extract_fans_second_fans('../SMP2017/SMPCUP2017数据集/8_Follow.txt','../task3_feature_first/follow_vector.txt')

#************fifth step*************************
#feature3
#write_feature3_to_file()

#************sixth step*****************************************
#feature4
# stopwords = load_stopwords('../data/stop_words_new.txt')
# all_user = load_alluser('../task3_feature_first/all_user_vector.txt')
# document_dict = make_document_dict('../task3_code/seg_word.txt')
# extract_all(stopwords,all_user,document_dict)


#*************ninth step************************
#****执行完LR.py文件中的write_all_feature_to_file()函数后执行此步******************************
#extract_ratio()

#**************tenth step************************
#trend_information()






