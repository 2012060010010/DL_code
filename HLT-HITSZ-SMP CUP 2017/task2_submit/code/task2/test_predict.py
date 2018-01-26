#coding:utf8
import pandas as pd
import xgboost as xgb
import numpy as np
import pickle
import os
import heapq
import random
#训练
dirpath='../../SMP2017_data/SMPCUP2017'
train_label_file=dirpath+"_2/SMPCUP2017_TrainingData_Task2.txt"
label_space_file=dirpath+"_2/LabelSpace_Task2.txt"
usr_docvec_file=dirpath+'_2/feature_doc/user2docvec_dbow_tvt.pkl'
train_feature_file=dirpath+"_2/feature_doc/usr_word_feature_102_tvt.pkl"
#验证
valid_file=dirpath+"_valid/SMPCUP2017_ValidationSet_Task2.txt"
# 测试
test_file=dirpath+"_valid/SMPCUP2017_TestSet_Task2.txt"
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
def load_data_and_labels(train_data_file1,usr_docvec_file,train_label_file,lb_num):
    user_label=get_label(train_label_file)
    # Load data from files
    with open(train_data_file1, "r") as ff:
        user_feature_dict=pickle.load(ff)
    with open(usr_docvec_file, "r") as ff:
        user_docvec_dict=pickle.load(ff)
    x_text=[]
    y_label=[]
    usr_set=user_feature_dict.keys()
    # random.seed(15)
    random.shuffle(usr_set)
    # test_num=int(0.10*len(usr_set))
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
        y_label.append(user_label[usr][0])
        x_text.append(usr_feature34)
        y_label.append(user_label[usr][1])
        x_text.append(usr_feature34)
        y_label.append(user_label[usr][2])
        del usr_feature34
    return [np.array(x_text), np.array(y_label)]
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
def load_data_and_multilabels(train_data_file1,usr_docvec_file,train_label_file,lb_num):
    user_label=get_label(train_label_file)
    with open(train_data_file1, "r") as ff:
        user_feature_dict=pickle.load(ff)
    with open(usr_docvec_file, "r") as ff:
        user_docvec_dict=pickle.load(ff)
    x_text=[]
    y_label=[]
    usr_set=user_feature_dict.keys()
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
    return [np.array(x_text), np.array(y_label)]
def accuracy(pred,y_data):
    hit_num=0
    for i in range(pred.shape[0]):
        pred_label=heapq.nlargest(3, range(len(pred[i])), pred[i].take)
        hit_num+=len(set(pred_label)&set(y_data[i]))/float(len(y_data[i]))
    acc=hit_num/pred.shape[0]
    return acc

# #词表匹配特征
# with open(user_all_feature_file) as f:
#     user_all_feature_34=pickle.load(f)
# #文档向量特征
# with open(usr_docvec_file) as f:
#     user_docvec_dict=pickle.load(f)
#验证集用户列表
# valid_user_list=[]
# with open(valid_file) as f:
#     for line in f:
#         valid_user_list.append(line.strip())
# #训练集用户列表
# train_user_list=[]
# with open(train_label_file) as f:
#     for line in f:
#         train_user_list.append(line.strip().split('\001')[0])
# #验证集用户标签词典
# train_label_dict={}
# with open(train_label_file) as f:
#     for line in f:
#         items=line.strip().split('\001')
#         train_label_dict[items[0]]=items[1:]

def train_pred(model_path):
    test_x, test_y = load_data_and_multilabels(train_feature_file, usr_docvec_file, train_label_file, lb_num=22)
    test_x = pd.DataFrame(test_x)
    test_x.columns = [i for i in range(test_x.shape[1])]
    test_x = xgb.DMatrix(test_x)
    #训练集
    model_list = os.listdir(model_path)
    pred_tmp=np.zeros([1055,42])
    for model in model_list:
        bst = xgb.Booster({'nthread': 4})  # init model
        bst.load_model(model_path+model)  # load model
        preds=bst.predict(test_x)
        print accuracy(preds, test_y)
        pred_tmp+=preds
        del bst
    acc = accuracy(pred_tmp, test_y)
    print acc
# train_pred('models/train_model/')
def valid_pred(model_path):
    valid_x,usr_list=load_valid_data(train_feature_file,usr_docvec_file, valid_file,lb_num=22)
    valid_x = pd.DataFrame(valid_x)
    valid_x.columns = [i for i in range(valid_x.shape[1])]
    valid_x = xgb.DMatrix(valid_x)
    #验证集
    model_list = os.listdir(model_path)
    pred_tmp=np.zeros([len(usr_list),42])
    for model in model_list:
        bst = xgb.Booster({'nthread': 4})  # init model
        bst.load_model(model_path+model)  # load model
        preds=bst.predict(valid_x)
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
    valid_pred_file.close()

# valid_pred('models/cv_xgb/12/')

def test_pred(model_path):
    Test_x,Test_usr_list=load_valid_data(train_feature_file,usr_docvec_file, test_file,lb_num=22)
    # print Test_x.shape
    Test_x = pd.DataFrame(Test_x)
    Test_x.columns = [i for i in range(Test_x.shape[1])]
    Test_x = xgb.DMatrix(Test_x)
    #测试集
    model_list = os.listdir(model_path)
    pred_tmp=np.zeros([len(Test_usr_list),42])
    for model in model_list:
        bst = xgb.Booster({'nthread': 4})  # init model
        bst.load_model(model_path+model)  # load model
        preds=bst.predict(Test_x)
        pred_tmp+=preds
        del bst
    with open(label_space_file,'r') as f:
        lines=f.readlines()
    label_list=[]
    for label in lines:
        label_list.append(label.strip())
    del lines
    Test_pred_file = open('preds_dir/cv_xgb_Test_pred201.txt', 'w')
    for i in range(pred_tmp.shape[0]):
        preds=pred_tmp[i]
        pred_label=heapq.nlargest(3, range(len(preds)), preds.take)
        label_text=[]
        for lb in pred_label:
            label_text.append(label_list[lb])
        Test_pred_file.write(str(Test_usr_list[i])+","+','.join(label_text)+'\n')
    Test_pred_file.close()
test_pred('models/cv_xgb/20/')