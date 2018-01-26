#coding:utf-8
from __future__ import division
import codecs
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import PolynomialFeatures
import numpy as np
import math
import xgboost as xgb
from sklearn.cross_validation import KFold
from joblib import Parallel,delayed
import pickle
import time



# 把所有的特征文件加载进来，进行连接，最后return一个  user：feature vector 字典
def load_data():
    user_vector_1 = codecs.open('../task3_feature_medium/all_user_vector.txt', 'r', 'utf-8')
    user_vector_2 = codecs.open('../task3_feature_medium/user_follow_vector.txt', 'r', 'utf-8')
    user_vector_3 = codecs.open('../task3_feature_medium/follow_vector.txt', 'r', 'utf-8')
    user_vector_4 = codecs.open('../task3_feature_medium/task3_feature_3.txt', 'r', 'utf-8')
    user_vector_5 = codecs.open('../line/embedding_file_128.txt', 'r', 'utf-8')  # use line(four version)
    user_vector_6 = codecs.open('../task3_feature_medium/task3_feature_4.txt', 'r', 'utf-8')

    vector_1_dict = dict()
    vector_2_dict = dict()
    vector_3_dict = dict()
    vector_4_dict = dict()
    vector_5_dict = dict()
    vector_6_dict = dict()
    all_user_dict = dict()

    for line in user_vector_1.readlines():
        line = line.strip()
        line = line.split(',')
        vector_1_dict[line[0]] = line[1:]
        all_user_dict[line[0]] = line[1:]

    for line in user_vector_2.readlines():
        line = line.strip()
        line = line.split(u'\001')
        vector_2_dict[line[0]] = line[1:]

    for line in user_vector_3.readlines():
        l = []
        line = line.strip()
        line = line.split(u'\001')
        line[1] = int(line[1])
        l.append(line[1])
        vector_3_dict[line[0]] = l

    for line in user_vector_4.readlines():
        line = line.strip()
        line = line.split(u'\001')
        vector_4_dict[line[0]] = list(map(float, line[1:]))

    for line in user_vector_5.readlines():
        line = line.strip()
        line = line.split()
        vector_5_dict[line[0]] = list(map(float, line[1:]))

    for line in user_vector_6.readlines():
        line = line.strip()
        line = line.split()
        vector_5_dict[line[0]] = list(map(float, line[1:]))

    for key, value in all_user_dict.items():
        if key in vector_2_dict:
            all_user_dict[key] = value + vector_2_dict[key]
        else:
            all_user_dict[key] = value + [0, 0]
        if key in vector_3_dict:
            all_user_dict[key] = all_user_dict[key] + vector_3_dict[key]
        else:
            all_user_dict[key] = all_user_dict[key] + [0]
        if key in vector_4_dict:
            all_user_dict[key] = all_user_dict[key] + vector_4_dict[key]
        else:
            all_user_dict[key] = all_user_dict[key] + [0] * 16
        if key in vector_5_dict:
            all_user_dict[key] = all_user_dict[key] + vector_5_dict[key]
        else:
            all_user_dict[key] = all_user_dict[key] + [0] * 128
        if key in vector_6_dict:
            all_user_dict[key] = all_user_dict[key] + vector_6_dict[key]
        else:
            all_user_dict[key] = all_user_dict[key] + [0] * 12
    return all_user_dict


# 把用户的7中行为的12个月份的行为，分别统计总数，最大数，平均数
def statistic_feature(all_user_dict):
    for key, value in all_user_dict.items():
        i = 0
        new_cloumn = []
        while (i * 12 <= 72):
            data = value[i * 12:(i + 1) * 12]
            data = list(map(int, data))
            sum_value = np.sum(np.array(data))
            max_value = max(data)
            min_value = min(data)
            mean_value = np.mean(np.array(data))
            media_value = np.median(np.array(data))
            var_value = np.var(np.array(data))
            range_value = max_value - min_value
            new_cloumn.extend([sum_value, max_value, min_value, mean_value, media_value, var_value, range_value])
            i += 1
        all_user_dict[key] = value + new_cloumn

    print(len(all_user_dict['U0000009']))
    return all_user_dict


# write all the features(before polynominal features) into files
def write_all_feature_to_file():
    all_user_dict = load_data()
    all_user_dict = statistic_feature(all_user_dict)

    with open('../task3_feature_final/all_user_dict(395).txt', 'w') as f:
        for key, value in all_user_dict.items():
            f.writelines(key+'\001'+'\001'.join(list(map(str,value))))


# 后面程序会把 load_data 和 statistic_feature的特征写入文件，此函数直接load
def load(filename):
    file_read = open(filename, 'rb')
    all_user_dict = pickle.load(file_read)
    for key, value in all_user_dict.items():
        all_user_dict[key] = list(map(float, value))
    return all_user_dict


# 抽取训练集 和 验证集 的 用户的向量
def return_train_valid_userid(all_user_vector, trainfile, validfile):
    train_file = codecs.open(trainfile, 'r', 'utf-8')
    valid_file = codecs.open(validfile, 'r', 'utf-8')
    x_train = []
    x_valid = []

    y_train = []
    user_train = []
    user_valid = []

    count_no = 0
    for line in train_file.readlines():
        line = line.strip()
        line = line.split(u'\001')
        user_train.append(line[0])
        y_train.append(float(line[1]))
        if line[0] in all_user_vector:
            x_train.append(all_user_vector[line[0]])
        else:
            print('no vector user:', line[0], 'num:', count_no)
            count_no += 1
            x_train.append([0] * 280)

    for line in valid_file.readlines():
        line = line.strip()
        user_valid.append(line)
        if line in all_user_vector:
            x_valid.append(all_user_vector[line])
        else:
            print('valid no vector:', line)
            x_valid.append([0] * 280)

    poly = PolynomialFeatures(interaction_only=True)
    x_train = (poly.fit_transform(np.array(x_train))).tolist()
    x_valid = (poly.fit_transform(np.array(x_valid))).tolist()
    print(len(x_train), len(x_valid))

    return x_train, y_train, x_valid, user_train, user_valid


# 任务三 打分函数
def score(pred, true):
    sum = 0.0
    for i in range(len(pred)):
        if pred[i] == 0 and true[i] == 0:
            sum = sum
        else:
            sum += math.fabs(pred[i] - true[i]) / max([pred[i], true[i]])
    sco = 1 - sum / len(pred)
    return sco


# xgb中用来获取 feature importance
def create_feature_map(features):
    outfile = codecs.open('../task3_feature_first/xgb.fmap', 'w', 'utf-8')
    i = 0
    for feat in features:
        outfile.writelines('{0}\t{1}\tq\n'.format(i, feat))
        i += 1
    outfile.close()


def get_features_name():
    features = [str(i) for i in range(392)]
    return features






def count_score(pred_probs, dtrain):
    true = list(dtrain.get_label())
    pred = list(pred_probs)
    sum = 0.0
    for i in range(len(pred)):
        if pred[i] == 0 and true[i] == 0:
            sum = sum
        else:
            sum += math.fabs(pred[i] - true[i]) / max([pred[i], true[i]])
    sum = sum / len(pred)

    return 'count_score', sum


# for stacking : model 1
def xgbt(data):
    print (time.ctime())
    x_test = xgb.DMatrix(data[4])
    data_train = xgb.DMatrix(data[0], label=data[1])
    data_valid = xgb.DMatrix(data[2], label=data[3])

    watch_list = [(data_valid, 'eval'), (data_train, 'train')]
    params = {}
    params['objective'] = 'reg:logistic'
    params['eta'] = 0.08
    params['max_depth'] = 22
    params['colsample_bytree'] = 1.0
    params['silent'] = 1

    bst = xgb.train(params, data_train, num_boost_round=1500, early_stopping_rounds=50, feval=count_score , verbose_eval=100, evals=watch_list)

    y_predict = bst.predict(data_valid)
    y_test = bst.predict(x_test)

    for i in range(len(y_predict)):
        if y_predict[i] < 0:
            y_predict[i] = math.fabs(y_test[i])

    for i in range(len(y_test)):
        if y_test[i] < 0:
            y_test[i] = math.fabs(y_test[i])

    sco = score(y_predict, data[3])
    return  bst, sco, y_predict.tolist(), y_test.tolist()


# 10 kfold for stacking
def split_data(train_x,y_train):
    kf = KFold(len(train_x), n_folds=10)
    train_x = np.array(train_x)
    y_train = np.array(list(map(float,y_train)))
    i = 1
    for train_index,valid_index in kf:
        globals()['train_x_'+str(i)],globals()['valid_x_'+str(i)] = train_x[train_index],train_x[valid_index]
        globals()['train_y_'+str(i)],globals()['valid_y_'+str(i)] = y_train[train_index],(y_train[valid_index]).tolist()
        i += 1
        
        if i ==11:
            return (train_x_1, train_y_1, valid_x_1, valid_y_1, train_x_2, train_y_2, valid_x_2, valid_y_2, train_x_3,
                    train_y_3, valid_x_3, valid_y_3,
                    train_x_4, train_y_4, valid_x_4, valid_y_4, train_x_5, train_y_5, valid_x_5, valid_y_5,
                    train_x_6, train_y_6, valid_x_6, valid_y_6, train_x_7, train_y_7, valid_x_7, valid_y_7,
                    train_x_8, train_y_8, valid_x_8, valid_y_8, train_x_9, train_y_9, valid_x_9, valid_y_9,
                    train_x_10, train_y_10, valid_x_10, valid_y_10)
       
def stacking(all_user_dict):
    x_train, y_train, x_valid, user_train, user_valid = return_train_valid_userid(all_user_dict,'../SMP2017/任务3训练集/SMPCUP2017_TrainingData_Task3.txt','../valid/SMPCUP2017_TestSet_Task3.txt')

    train_x = []
    test_x = []

    for l in x_train:
        train_x.append(list(map(float, l)))
    for l in x_valid:
        test_x.append(list(map(float, l)))
    print('train x', len(train_x))
    test_x = np.array(test_x)
    train_x_1, train_y_1, valid_x_1, valid_y_1, train_x_2, train_y_2, valid_x_2, valid_y_2, train_x_3, train_y_3, valid_x_3, valid_y_3, train_x_4, train_y_4, valid_x_4, valid_y_4, train_x_5, train_y_5, valid_x_5, valid_y_5, train_x_6, train_y_6, valid_x_6, valid_y_6, train_x_7, train_y_7, valid_x_7, valid_y_7, train_x_8, train_y_8, valid_x_8, valid_y_8, train_x_9, train_y_9, valid_x_9, valid_y_9, train_x_10, train_y_10, valid_x_10, valid_y_10 = split_data(
        train_x, y_train)

    # model 1  xgboost

    all_data = [(train_x_1, train_y_1, valid_x_1, valid_y_1, test_x),
                (train_x_2, train_y_2, valid_x_2, valid_y_2, test_x),
                (train_x_3, train_y_3, valid_x_3, valid_y_3, test_x),
                (train_x_4, train_y_4, valid_x_4, valid_y_4, test_x),
                (train_x_5, train_y_5, valid_x_5, valid_y_5, test_x),
                (train_x_6, train_y_6, valid_x_6, valid_y_6, test_x),
                (train_x_7, train_y_7, valid_x_7, valid_y_7, test_x),
                (train_x_8, train_y_8, valid_x_8, valid_y_8, test_x),
                (train_x_9, train_y_9, valid_x_9, valid_y_9, test_x),
                (train_x_10, train_y_10, valid_x_10, valid_y_10, test_x)]

    result = Parallel(n_jobs=3, backend='threading')(delayed(xgbt)(data) for data in all_data)
    y_predict = []
   
    for i in range(len(result[0][3])):
        data = (result[0][3][i]+result[1][3][i]+result[2][3][i]+result[3][3][i]+result[4][3][i]+result[5][3][i]
                +result[6][3][i]+result[7][3][i]+result[8][3][i]+result[9][3][i])/10
        y_predict.append(data)
  
    # P1

    print('xgb score[max,min,mean]:',max(result[0][1],result[1][1],result[2][1],result[3][1],result[4][1],result[5][1],result[6][1],result[7][1],result[8][1],result[9][1])
           ,min(result[0][1],result[1][1],result[2][1],result[3][1],result[4][1],result[5][1],result[6][1],result[7][1],result[8][1],result[9][1])
          ,np.mean(np.array([result[0][1],result[1][1],result[2][1],result[3][1],result[4][1],result[5][1],result[6][1],result[7][1],result[8][1],result[9][1]]))
           )
   
    base_name = time.ctime()
   
    file_write = codecs.open('../results/' + 'true_pred.txt', 'w', 'utf-8')
    for i in range(len(y_predict)):
        if y_predict[i] < 0:
            y_predict[i] = math.fabs(y_predict[i])
        file_write.writelines(user_valid[i] + ',' + str(y_predict[i]) + '\n')
    file_write.close()



# **********eighth step**************************************
# write_all_feature_to_file()


# *************eleventh step(final step)**********************************

all_user_dict = load('../task3_medium/all_user_dict(395_add_continuous_quarter).txt')
stacking(all_user_dict)










