#coding: utf8
#对少数样例进行过采样，采样权重由标签数量决定
import numpy as np
import pickle
from collections import Counter
import math
import heapq
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
dirpath='../../SMP2017_data/SMPCUP2017'
train_feature_file=dirpath+"_2/feature_doc/usr_word_feature_102_all.pkl"
train_label_file=dirpath+"_2/SMPCUP2017_TrainingData_Task2.txt"
label_space_file=dirpath+"_2/LabelSpace_Task2.txt"
usr_docvec_file=dirpath+'_2/feature_doc/user2docvec_dbow.pkl'

def get_label_dict(label_file):
    '''获取文本label和位置编号的对应关系，用于转换label到数字'''
    label_space=open(label_file)
    lines=label_space.readlines()
    label_num=0
    label_dict={}
    for label in lines:
        label_dict[label.strip()]=label_num
        label_num+=1
    label_space.close()
    return label_dict
label_dict= get_label_dict(label_space_file)
def get_label(train_file,label_dict=label_dict):
    """
    将训练集label转换index列表，作为最终label
    :param train_file: 训练集数据
    :param label_dict: 标签和位置对应关系
    :return: 用户：标签（3维）
    """
    task2_train=open(train_file,'r')
    lines=task2_train.readlines()
    user_label={}
    for line in lines:
        multi_label=[]
        items=line.strip().split('\001')
        for i in [1,2,3]:
            index=label_dict[items[i]]
            multi_label.append(index)
        user_label[items[0]]=multi_label
    task2_train.close()
    return user_label
def load_data_and_labels(train_feature_file,usr_docvec_file,train_label_file,lb_num):
    user_label=get_label(train_label_file)
    # Load data from files
    with open(train_feature_file, "r") as ff:
        user_feature_dict=pickle.load(ff)
    with open(usr_docvec_file, "r") as ff:
        user_docvec_dict=pickle.load(ff)
    with open(train_label_file,'r') as f:
        lines=f.readlines()
    usr_list=[]
    for line in lines:
        usr_list.append(line.strip().split('\001')[0])
    del lines

    x_text=[]
    y_label=[]
    for usr in usr_list:
        usr_feature34 = user_feature_dict[usr][:lb_num * 3]
        usr_feature_34 = np.array([0] * 34)
        for i in range(34):
            usr_feature_34[i] = sum(user_feature_dict[usr][i * 3:i * 3 + 3])
        usr_feature_34 = list(usr_feature_34)
        max_f1 = max(usr_feature_34) + 0.0001
        usr_feature_34 = [it / max_f1 for it in usr_feature_34]
        usr_feature34.extend(usr_feature_34)
        try:
            usr_feature34.extend(list(user_docvec_dict[usr]))
        except KeyError:
            usr_feature34.extend([0]*128)
        x_text.append(usr_feature34)
        y_label.append(user_label[usr][0])
        x_text.append(usr_feature34)
        y_label.append(user_label[usr][1])
        x_text.append(usr_feature34)
        y_label.append(user_label[usr][2])
        del usr_feature34

    return [np.array(x_text), np.array(y_label)]
x, y = load_data_and_labels(train_feature_file, usr_docvec_file,
                                                train_label_file, lb_num=22)
sort_cnt=sorted(Counter(y).items(),key=lambda x:x[0])
print sort_cnt
lb_num_dict={}
sum_num=0
for it in sort_cnt:
    cnt=3*int(math.log10(it[1])/math.log10(2))
    lb_num_dict[it[0]]=cnt
    sum_num+=cnt
print sorted(lb_num_dict.items(),key=lambda x:x[0])
print sum_num
label_user_dict={}
with open(train_label_file) as f:
    for line in f:
        items=line.strip().split('\001')
        user=items[0]
        for lb in items[1:]:
            try:
                label_user_dict[label_dict[lb]].append(user)
            except KeyError:
                label_user_dict[label_dict[lb]]=[user]
sample_user_list=[]
for i in range(42):
    sample_ind=np.random.permutation(sort_cnt[i][1])
    for ind in sample_ind[:lb_num_dict[i]]:
        sample_user_list.append((label_user_dict[i][ind],i))
print len(sample_user_list)
with open(dirpath+'_2/feature_doc/sample_user_list.pkl','w') as f:
    pickle.dump(sample_user_list,f)

#获得重复抽样的用户列表,内容为usrID:label对应关系
with open(dirpath+'_2/feature_doc/sample_user_list.pkl') as f:
    sample_user=pickle.load(f)
with open(train_feature_file, "r") as ff:
    user_feature_dict=pickle.load(ff)
with open(usr_docvec_file, "r") as ff:
    user_docvec_dict=pickle.load(ff)

def load_data_and_multilabels(train_data_file1,usr_docvec_file,train_label_file,lb_num):
    user_label=get_label(train_label_file)
    with open(train_data_file1, "r") as ff:
        user_feature_dict=pickle.load(ff)
    with open(usr_docvec_file, "r") as ff:
        user_docvec_dict=pickle.load(ff)
    with open(train_label_file,'r') as f:
        lines=f.readlines()
    usr_list=[]
    for line in lines:
        usr_list.append(line.strip().split('\001')[0])
    del lines

    x_text=[]
    y_label=[]
    # random.seed(15)
    # random.shuffle(usr_list)
    for usr in usr_list:
        usr_feature34 = user_feature_dict[usr][:lb_num * 3]
        usr_feature_34 = np.array([0] * 34)
        for i in range(34):
            usr_feature_34[i] = sum(user_feature_dict[usr][i * 3:i * 3 + 3])
        usr_feature_34 = list(usr_feature_34)
        max_f1 = max(usr_feature_34) + 0.0001
        usr_feature_34 = [it / max_f1 for it in usr_feature_34]
        usr_feature34.extend(usr_feature_34)
        try:
            usr_feature34.extend(list(user_docvec_dict[usr]))
        except KeyError:
            usr_feature34.extend([0]*128)

        x_text.append(usr_feature34)
        y_label.append(user_label[usr])
        del usr_feature34
    return [np.array(x_text), np.array(y_label)]
def accuracy(pred,y_data):
    hit_num=0
    for i in range(pred.shape[0]):
        pred_label=heapq.nlargest(3, range(len(pred[i])), pred[i].take)
        hit_num+=len(set(pred_label)&set(y_data[i]))/float(len(y_data[i]))
    acc=hit_num/pred.shape[0]
    return acc
x_samp=[]
y_samp=[]
for it in sample_user:
    user=it[0]
    label=it[1]
    usr_feature34 = user_feature_dict[user][:22 * 3]
    usr_feature_34 = np.array([0] * 34)
    for i in range(34):
        usr_feature_34[i] = sum(user_feature_dict[user][i * 3:i * 3 + 3])
    usr_feature_34 = list(usr_feature_34)
    max_f1 = max(usr_feature_34) + 0.0001
    usr_feature_34 = [it / max_f1 for it in usr_feature_34]
    usr_feature34.extend(usr_feature_34)
    usr_feature34.extend(user_docvec_dict[user])
    x_samp.append(usr_feature34)
    y_samp.append(label)
    del usr_feature34
    del usr_feature_34
X_data = pd.DataFrame(x)
Y_data = pd.DataFrame(y)
X_data.columns = [i for i in range(X_data.shape[1])]
Y_data.columns = [j for j in range(Y_data.shape[1])]
X_samp= pd.DataFrame(x_samp)
y_samp = pd.DataFrame(y_samp)
X_samp.columns=X_data.columns
y_samp.columns=Y_data.columns
X_semi=pd.concat([X_samp,X_data],axis=0,join='inner')
Y_semi=pd.concat([y_samp,Y_data],axis=0,join='inner')
X_train, X_test, y_train, y_test = train_test_split(X_semi, Y_semi, test_size=0.1)

print X_train.shape
print X_test.shape
test_x, test_y = load_data_and_multilabels(train_feature_file, usr_docvec_file,train_label_file,lb_num=22)
x_test = pd.DataFrame(test_x)
x_test.columns = X_data.columns
acc=[]
for i in range(10):
    param = {
        'eta': 0.01,
        'silent': 1,
        'objective': 'multi:softprob',
        'booster': 'gblinear',
        'max_depth': i,
        'nthread': 4, 'num_class': 42,
        'lambda': 0.7,
        'alpha': 1.0,
    }
    X_train, X_test, y_train, y_test = train_test_split(X_semi, Y_semi, test_size=0.1)
    num_round = 600
    early_stop =20
    semi_train=xgb.DMatrix(X_train, y_train)
    semi_test=xgb.DMatrix(X_test, y_test)
    test = xgb.DMatrix(x_test)
    watchlist_semi = [(semi_test, 'eval'), (semi_train, 'train')]
    bst2 = xgb.train(param, semi_train, num_round,watchlist_semi, early_stopping_rounds=early_stop)
    semi_preds = bst2.predict(test)
    accu = accuracy(semi_preds, test_y)
    bst2.save_model('models/' + ''.join(str(accu).split('.')[1][:3]) + '.model')
    del bst2
    acc.append(accu)
print acc