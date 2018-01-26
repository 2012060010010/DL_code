#coding: utf8
import pandas as pd
import xgboost as xgb
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
import heapq
import random
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
    test_x=[]
    test_y=[]
    random.seed(15)
    random.shuffle(usr_list)
    test_num=int(0.10*len(usr_list))
    for usr in usr_list[:-test_num]:
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

    for usr in usr_list[-test_num:]:
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
            usr_feature34.extend([0]*128)

        test_x.append(usr_feature34)
        del usr_feature34
        test_y.append(user_label[usr])
    return [np.array(x_text), np.array(y_label),np.array(test_x),np.array(test_y)]

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
    return x_text,usr_list
def accuracy(pred,y_data):
    hit_num=0
    for i in range(pred.shape[0]):
        pred_label=heapq.nlargest(3, range(len(pred[i])), pred[i].take)
        hit_num+=len(set(pred_label)&set(y_data[i]))/float(len(y_data[i]))
    acc=hit_num/pred.shape[0]
    return acc

def train():
    acc = []
    x, y, test_x, test_y = load_data_and_labels(train_feature_file, usr_docvec_file1,
                                                train_label_file, lb_num=22)
    for lb in range(22,23):
        # x, y, test_x, test_y = load_data_and_labels(train_feature_file, usr_docvec_file1,
        #                                             train_label_file, lb_num=22)
        X_data = pd.DataFrame(x)
        Y_data = pd.DataFrame(y)
        test_x = pd.DataFrame(test_x)

        X_data.columns = [i for i in range(X_data.shape[1])]
        test_x.columns = X_data.columns
        Y_data.columns = [j for j in range(Y_data.shape[1])]
        X = X_data[X_data.columns]
        Y = Y_data[Y_data.columns]

        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.1)

        print X_train.shape, X_test.shape
        print y_train.shape, y_test.shape

        dtrain = xgb.DMatrix(X_train, y_train)
        dtest = xgb.DMatrix(X_test, y_test)
        test = xgb.DMatrix(test_x)
        lr = [0.1]
        for l in lr:
            param = {
                'eta': 0.01, #0.01
                'silent': 1,
                'objective': 'multi:softprob',
                'booster': 'gblinear',
                'max_depth': 6,
                'nthread': 4, 'num_class': 42,
                'lambda':0.7,#0.7
                'alpha':1.0,#1.0
            }
            watchlist = [(dtest, 'eval'), (dtrain, 'train')]
            num_round = 600
            early_stop = 15
            bst = xgb.train(param, dtrain, num_round, watchlist, early_stopping_rounds=early_stop)
            preds = bst.predict(test)
            accu = accuracy(preds, test_y)
            acc.append(accu)
            bst.save_model('models/' + ''.join(str(accu).split('.')[1][:3]) + '.model')
            del bst
    print np.mean(acc)
    print acc
train()
def pred(valid_feature_file, valid_file,label_space_file,model_name):
    test_x,usr_list=load_valid_data(valid_feature_file,usr_docvec_file1,valid_file,lb_num=22)
    test_x = pd.DataFrame(test_x)
    test_x.columns = [i for i in range(test_x.shape[1])]
    test_x = xgb.DMatrix(test_x)
    bst = xgb.Booster({'nthread': 4})  # init model
    bst.load_model(model_name)  # load model

    preds=bst.predict(test_x)
    # with open('pred_prob.pkl','w') as f:
    #     pickle.dump(preds,f)
    with open(label_space_file,'r') as f:
        lines=f.readlines()
    label_list=[]
    for label in lines:
        label_list.append(label.strip())
    del lines
    valid_pred_file = open('preds_dir/'+model_name.split('.')[0].split('/')[-1]+'valid_pred.txt', 'w')
    for i in range(preds.shape[0]):
        pred_label=heapq.nlargest(3, range(len(preds[i])), preds[i].take)
        label_text=[]
        for lb in pred_label:
            label_text.append(label_list[lb])
        valid_pred_file.write(str(usr_list[i])+","+','.join(label_text)+'\n')
# pred(train_feature_file, valid_file,label_space_file,model_name='models/569.model')
# pred(train_data_file1, train_label_file,label_space_file,model_name='models/453.model')

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
    random.shuffle(usr_list)
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
def train_acc(model_list):
    test_x,test_y=load_data_and_multilabels(train_feature_file,usr_docvec_file1,train_label_file,lb_num=22)
    test_x = pd.DataFrame(test_x)
    test_x.columns = [i for i in range(test_x.shape[1])]
    test_x = xgb.DMatrix(test_x)
    for model in model_list:
        bst = xgb.Booster({'nthread': 4})  # init model
        bst.load_model('models/train_model/'+model)  # load model
        preds=bst.predict(test_x)
        acc=accuracy(preds,test_y)
        print model,acc
# model_list=os.listdir('models/train_model/')
# train_acc(model_list)

