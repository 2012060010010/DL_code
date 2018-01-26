#coding: utf8
import pandas as pd
import xgboost as xgb
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
import heapq
import random
import os
from sklearn.model_selection import KFold
#训练
train_data_file1="../../SMP2017_data/SMPCUP2017_2/feature_doc/usr_labels_feature_34_102.pkl"
train_feature_file="../../SMP2017_data/SMPCUP2017_2/feature_doc/usr_word_feature_102_all.pkl"
train_label_file="../../SMP2017_data/SMPCUP2017_2/SMPCUP2017_TrainingData_Task2.txt"
label_space_file="../../SMP2017_data/SMPCUP2017_2/LabelSpace_Task2.txt"
usr_docvec_file1='../../SMP2017_data/SMPCUP2017_2/feature_doc/user2docvec_dbow.pkl'
#测试
valid_data_file="../../SMP2017_data/SMPCUP2017_valid/feature_doc/usr_labels_feature_34_102.pkl"
valid_file="../../SMP2017_data/SMPCUP2017_valid/SMPCUP2017_ValidationSet_Task2.txt"

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
def label_transform(train_file,label_dict=label_dict):
    """
    将训练集label转换成01列表，作为最终label
    :param train_file: 训练集数据
    :param label_dict: 标签和位置对应关系
    :return: 用户：标签（42维）
    """
    task2_train=open(train_file,'r')
    lines=task2_train.readlines()
    user_label={}
    for line in lines:
        multi_label=[0]*42
        items=line.strip().split('\001')
        for i in [1,2,3]:
            index=label_dict[items[i]]
            multi_label[index]=1
        user_label[items[0]]=multi_label
    task2_train.close()
    return user_label

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

def load_data_and_labels(train_data_file1,usr_docvec_file,train_label_file,lb_num):
    user_label=get_label(train_label_file)
    # Load data from files
    with open(train_data_file1, "r") as ff:
        user_feature_dict=pickle.load(ff)
    with open(usr_docvec_file, "r") as ff:
        user_docvec_dict=pickle.load(ff)
    # with open(usr_embedding_doc_file, "r") as ff:
    #     user_embed_dict=pickle.load(ff)
    # with open(train_followed_file, "r") as ff:
    #     user_followed_dict=pickle.load(ff)
    x_text=[]
    y_label=[]
    test_x=[]
    test_y=[]
    usr_set=user_feature_dict.keys()
    random.seed(15)
    random.shuffle(usr_set)
    test_num=int(0.10*len(usr_set))
    # lb_num=22
    # ind=[0,1,2,3,4,5,6,7,8,9,11,13,14,15,16,17,18,20,25,29,32,33]
    # ind=range(lb_num)
    # index=[]
    # for i in ind:
    #     index.append(3*i)
    #     index.append(3*i+1)
    #     index.append(3*i+2)
    # count=0
    for usr in usr_set[:-test_num]:
        usr_feature34 = user_feature_dict[usr][:lb_num * 3]
        usr_feature_34 = np.array([0] * 34)
        # usr_feature_34_f=np.array([0]*34)

        for i in range(34):
            usr_feature_34[i] = sum(user_feature_dict[usr][i * 3:i * 3 + 3])

        # if sum(user_follow_dict[usr]) > 0:
        #     # print 'hehe'
        #     for j in range(3):
        #         for i in range(34):
        #             usr_feature_34[i] = sum(user_follow_dict[usr][102 * j + i * 3:102 * j + i * 3 + 3])
        # if sum(user_followed_dict[usr]) > 0:
        #     # print 'haha'
        #     for j in range(3):
        #         for i in range(34):
        #             usr_feature_34[i] = sum(user_followed_dict[usr][102 * j + i * 3:102 * j + i * 3 + 3])

        usr_feature_34 = list(usr_feature_34)
        max_f1 = max(usr_feature_34) + 0.0001
        usr_feature_34 = [it / max_f1 for it in usr_feature_34]
        usr_feature34.extend(usr_feature_34)
        try:
            usr_feature34.extend(list(user_docvec_dict[usr]))
        except KeyError:
            # count+=1
            usr_feature34.extend([0]*128)

        x_text.append(usr_feature34)
        y_label.append(user_label[usr][0])
        x_text.append(usr_feature34)
        y_label.append(user_label[usr][1])
        x_text.append(usr_feature34)
        y_label.append(user_label[usr][2])
        del usr_feature34
    # print count

    for usr in usr_set[-test_num:]:
        usr_feature34 = user_feature_dict[usr][:lb_num*3]
        usr_feature_34=np.array([0]*34)
        # usr_feature_34_f=np.array([0]*34)

        for i in range(34):
            usr_feature_34[i]=sum(user_feature_dict[usr][i*3:i*3+3])
        usr_feature_34=list(usr_feature_34)
        max_f1 = max(usr_feature_34) + 0.0001
        usr_feature_34 = [it / max_f1 for it in usr_feature_34]
        usr_feature34.extend(usr_feature_34)

        try:
            usr_feature34.extend(list(user_docvec_dict[usr]))
        except KeyError:
            # count += 1
            usr_feature34.extend([0]*128)

        test_x.append(usr_feature34)
        del usr_feature34
        test_y.append(user_label[usr])
    # print count
    return [np.array(x_text), np.array(y_label),np.array(test_x),np.array(test_y)]

def load_data_and_multilabels(train_data_file1,usr_docvec_file,train_label_file,lb_num):
    user_label=get_label(train_label_file)
    with open(train_data_file1, "r") as ff:
        user_feature_dict=pickle.load(ff)
    with open(usr_docvec_file, "r") as ff:
        user_docvec_dict=pickle.load(ff)
    with open('../../SMP2017_data/SMPCUP2017_2/feature_doc/sample_user_list.pkl') as f:
        sample_user = pickle.load(f)
    x_text=[]
    y_label=[]
    with open(train_label_file,'r') as f:
        lines=f.readlines()
    usr_set=[]
    for line in lines:
        usr_set.append(line.strip().split('\001')[0])
    del lines

    # random.seed(15)
    random.shuffle(usr_set)
    for usr in usr_set:
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
    #分界线
    x_samp = []
    y_samp = []
    for it in sample_user:
        user = it[0]
        label = it[1]
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
    return [np.array(x_text), np.array(y_label),x_samp,y_samp]

def load_valid_data(valid_feature_file,usr_docvec_file,valid_file,lb_num):
    # Load data from files
    with open(valid_feature_file, "r") as ff:
        user_feature_dict=pickle.load(ff)
    with open(usr_docvec_file, "r") as ff:
        user_docvec_dict=pickle.load(ff)
    with open(valid_file,'r') as f:
        lines=f.readlines()
    usr_list=[]
    for line in lines:
        usr_list.append(line.strip().split('\001')[0])
    del lines
    x_text=[]
    for usr in usr_list:
        usr_feature34=user_feature_dict[usr][:lb_num*3]
        usr_feature_34=np.array([0]*34)

        for i in range(34):
            usr_feature_34[i]=sum(user_feature_dict[usr][i*3:i*3+3])
        usr_feature_34=list(usr_feature_34)
        max_f1 = max(usr_feature_34) + 0.0001
        usr_feature_34 = [it / max_f1 for it in usr_feature_34]
        usr_feature34.extend(usr_feature_34)
        try:
            usr_feature34.extend(list(user_docvec_dict[usr]))
        except KeyError:
            usr_feature34.extend([0]*128)
        x_text.append(usr_feature34)
        del usr_feature34
    return np.array(x_text),usr_list

def accuracy(pred1,y_data):
    hit_num=0
    for i in range(pred1.shape[0]):
        pred=pred1[i]
        pred_label=heapq.nlargest(3, range(len(pred)), pred.take)
        hit_num+=len(set(pred_label)&set(y_data[i]))/float(len(y_data[i]))
    acc=hit_num/pred1.shape[0]
    return acc

def make_sample_data():
    with open('SMP2017_data/SMPCUP2017_2/feature_doc/sample_user_list.pkl') as f:
        sample_user = pickle.load(f)
    with open(train_feature_file, "r") as ff:
        user_feature_dict = pickle.load(ff)
    with open(usr_docvec_file1, "r") as ff:
        user_docvec_dict = pickle.load(ff)
    x_samp = []
    y_samp = []
    for it in sample_user:
        user = it[0]
        label = it[1]
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
    return [np.array(x_samp),np.array(y_samp)]

def makedir(path):
    import os
    path = path.strip()
    isExists = os.path.exists(path)
    if not isExists:
        os.makedirs(path)
        return True
    else:
        return False

def train_cv_10(k):
    acc = []
    x, y,x_samp,y_samp= load_data_and_multilabels(train_feature_file, usr_docvec_file1,train_label_file, lb_num=22)
    kf=KFold(k)
    model_num=0
    for train_ind,test_ind in kf.split(x):
        model_num+=1
        train_x=[]
        test_x=[]
        train_y=[]
        test_y=[]
        for ind in train_ind:
            train_x.append(x[ind])
            train_y.append(y[ind,0])
            train_x.append(x[ind])
            train_y.append(y[ind,1])
            train_x.append(x[ind])
            train_y.append(y[ind,2])
        for ind in test_ind:
            test_x.append(x[ind])
            test_y.append(y[ind])
        # shuffle_ind=np.random.permutation(len(train_x))
        X_data = pd.DataFrame(np.array(train_x))
        Y_data = pd.DataFrame(np.array(train_y))
        x_test = pd.DataFrame(np.array(test_x))
        y_test = np.array(test_y)

        X_data.columns = [i for i in range(X_data.shape[1])]
        x_test.columns = X_data.columns
        Y_data.columns = [j for j in range(Y_data.shape[1])]

        X_train, X_dev, y_train, y_dev = train_test_split(X_data, Y_data, test_size=0.1,random_state=2)
        #
        print X_data.shape, Y_data.shape
        print x_test.shape,y_test.shape
        dtrain = xgb.DMatrix(X_train, y_train)
        ddev = xgb.DMatrix(X_dev, y_dev)
        test = xgb.DMatrix(x_test)
        param = {
            'eta': 0.01,  # 0.01
            'silent': 1,
            'objective': 'multi:softprob',
            'booster': 'gblinear',
            'max_depth': 6,
            'nthread': 4, 'num_class': 42,
            'lambda': 0.7,  # 0.7
            'alpha': 1.0,  # 1.0
        }
        watchlist = [(ddev, 'eval'), (dtrain, 'train')]
        num_round = 600
        early_stop = 20
        bst = xgb.train(param, dtrain, num_round, watchlist, early_stopping_rounds=early_stop)
        preds = bst.predict(test)
        accu = accuracy(preds, y_test)
        acc.append(accu)
        model_path='models/cv_xgb/' + str(k)
        makedir(model_path)
        bst.save_model(model_path+'/'+str(model_num)+'_'+''.join(str(accu).split('.')[1][:3]) + '.model')
        del bst
    print np.mean(acc),np.max(acc),np.min(acc)
    print acc
def train_cv_sample(k):
    acc = []
    x, y,x_samp,y_samp= load_data_and_multilabels(train_feature_file, usr_docvec_file1,train_label_file, lb_num=22)
    kf=KFold(k)
    model_num=0
    for train_ind,test_ind in kf.split(x):
        model_num+=1
        train_x=[]
        test_x=[]
        train_y=[]
        test_y=[]
        for ind in train_ind:
            train_x.append(x[ind])
            train_y.append(y[ind,0])
            train_x.append(x[ind])
            train_y.append(y[ind,1])
            train_x.append(x[ind])
            train_y.append(y[ind,2])
        for ind in test_ind:
            test_x.append(x[ind])
            test_y.append(y[ind])

        train_x.extend(x_samp)
        train_y.extend(y_samp)
        shuffle_ind=np.random.permutation(len(train_x))
        X_data = pd.DataFrame(np.array(train_x)[shuffle_ind])
        Y_data = pd.DataFrame(np.array(train_y)[shuffle_ind])
        x_test = pd.DataFrame(np.array(test_x))
        y_test = np.array(test_y)

        X_data.columns = [i for i in range(X_data.shape[1])]
        x_test.columns = X_data.columns
        Y_data.columns = [j for j in range(Y_data.shape[1])]

        X_train, X_dev, y_train, y_dev = train_test_split(X_data, Y_data, test_size=0.1)
        #
        print X_data.shape, Y_data.shape
        print x_test.shape,y_test.shape
        dtrain = xgb.DMatrix(X_train, y_train)
        ddev = xgb.DMatrix(X_dev, y_dev)
        test = xgb.DMatrix(x_test)
        param = {
            'eta': 0.01,  # 0.01
            'silent': 1,
            'objective': 'multi:softprob',
            'booster': 'gblinear',
            'max_depth': 6,
            'nthread': 4, 'num_class': 42,
            'lambda': 0.7,  # 0.7
            'alpha': 1.0,  # 1.0
        }
        watchlist = [(ddev, 'eval'), (dtrain, 'train')]
        num_round = 600
        early_stop = 20
        bst = xgb.train(param, dtrain, num_round, watchlist, early_stopping_rounds=early_stop)
        preds = bst.predict(test)
        accu = accuracy(preds, y_test)
        acc.append(accu)
        model_path='models/cv_xgb/' + str(k)
        makedir(model_path)
        bst.save_model(model_path+'/'+str(model_num)+'_'+''.join(str(accu).split('.')[1][:3]) + '.model')
        del bst
    print np.mean(acc),np.max(acc),np.min(acc)
    print acc
for k in [10]:
    train_cv_sample(k)
def pred(valid_file,label_space_file,model_list):
    test_x,usr_list=load_valid_data(train_feature_file,usr_docvec_file1, valid_file,lb_num=22)
    test_x = pd.DataFrame(test_x)
    test_x.columns = [i for i in range(test_x.shape[1])]
    test_x = xgb.DMatrix(test_x)

    pred_tmp=np.zeros([len(usr_list),42])
    for model in model_list:
        bst = xgb.Booster({'nthread': 4})  # init model
        bst.load_model('models/cv_xgb/'+model)  # load model
        preds=bst.predict(test_x)
        pred_tmp+=preds
        del bst
    with open(label_space_file,'r') as f:
        lines=f.readlines()
    label_list=[]
    for label in lines:
        label_list.append(label.strip())
    del lines
    valid_pred_file = open('preds_dir/cv_xgb_valid_pred.txt', 'w')
    for i in range(pred_tmp.shape[0]):
        preds=pred_tmp[i]
        pred_label=heapq.nlargest(3, range(len(preds)), preds.take)
        label_text=[]
        for lb in pred_label:
            label_text.append(label_list[lb])
        valid_pred_file.write(str(usr_list[i])+","+','.join(label_text)+'\n')
# model_list=os.listdir('models/cv_cgb/')
# pred(valid_file,label_space_file,model_list)

def train_acc(model_path):
    model_list = os.listdir(model_path)
    test_x, test_y = load_data_and_multilabels(train_feature_file, usr_docvec_file1, train_label_file, lb_num=22)
    test_x = pd.DataFrame(test_x)
    test_x.columns = [i for i in range(test_x.shape[1])]
    test_x = xgb.DMatrix(test_x)
    pred_tmp=np.zeros([1055,42])
    for model in model_list:
        bst = xgb.Booster({'nthread': 4})  # init model
        bst.load_model(model_path+model)  # load model
        preds=bst.predict(test_x)
        pred_tmp+=preds
        del bst
    acc = accuracy(pred_tmp, test_y)
    print acc

# model_path='models/cv_xgb/7/'
# train_acc(model_path)

