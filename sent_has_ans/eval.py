#! /usr/bin/env python
import tensorflow as tf
import numpy as np
import os
import time
import datetime
import pickle
from tqdm import tqdm
from sent_has_ans import data_helpers
from sent_has_ans import text_cnn
# from tensorflow.contrib import learn

# 输入数据参数
tf.flags.DEFINE_string("test_data_file", "../BDCI2017-taiyi/features_chandi_1210_new.pkl", "Data source for the training data.")
tf.flags.DEFINE_integer("max_document_length", 40, "Max length of each document")
tf.flags.DEFINE_integer("number_of_class", 7, "Number of class")
tf.flags.DEFINE_string("wordvector_pretrain", "../w2v/fasttext_char_vec/fasttext_cbow_char.model.vec", "wordvector pretrain file")

# 模型的超参数
tf.flags.DEFINE_integer("embedding_dim", 100, "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_string("filter_sizes", "2,3,4", "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_integer("num_filters", 128, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.8, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.1, "L2 regularization lambda (default: 0.0)")
tf.flags.DEFINE_float("learning_rate", 1e-4, "Learning rate (default: 1e-4)")
tf.flags.DEFINE_float("learning_rate_decay", 0.97, "learning rate decay")
tf.flags.DEFINE_string("checkpoint_dir", "./runs/Sun_Dec_10_04_22_53_2017/checkpoints", "dir of checkpoints")

# Misc 参数
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()

for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))

# 导入数据
print("Loading data...")
train2, test2 = data_helpers.load_test_data(FLAGS.test_data_file)

# 构建词向量与词典
vocab,vocab2index,word_embed = data_helpers.get_fasttext(FLAGS.wordvector_pretrain)
print('vocab len: ', len(vocab))
print('word embed: ',word_embed.shape)
max_document_length = FLAGS.max_document_length

# x_text=np.array([data_helpers.sent2index(sent,max_document_length,vocab2index) for sent in sent_data])
print("\nEvaluating...\n")
# print(test_data[2])
# print(type(test_data[2]))

# 开始验证
# ==================================================
checkpoint_file = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
# Sun_Dec__3_17_13_40_2017
graph = tf.Graph()
with graph.as_default():
    session_conf = tf.ConfigProto(
      allow_soft_placement=FLAGS.allow_soft_placement,
      log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        # 导入训练好的模型与变量
        saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
        saver.restore(sess, checkpoint_file)
        # 通过名字导入变量
        input_x = graph.get_operation_by_name("input_x").outputs[0]
        W = graph.get_operation_by_name("embedding/WW").outputs[0]
        h_compress = graph.get_operation_by_name("sentence_dropout/h_compress").outputs[0]
        dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]
        # 验证的Tensor
        predictions = graph.get_operation_by_name("output/predictions").outputs[0]
        print('开始预测！！')
        def pred(x_text):
            ans_num_list = []
            compress_list = []
            for it in x_text:
                sent = ' '.join(''.join(it))
                # print(sent)
                sent2id = data_helpers.sent2index(sent, max_document_length, vocab2index)
                inp = []
                inp.append(sent2id)
                ans_num, compress = sess.run([predictions, h_compress],
                                          {input_x: np.array(inp),
                                           W: word_embed,
                                           dropout_keep_prob: 1.0
                                           })
                ans_num_list.append(ans_num[0])
                compress_list.append(compress[0])
            return ans_num_list,compress_list
        def process_df(data_df):
            test_data = list(data_df['sub_sents_tokenized'])
            print('test: ', data_df.shape)
            ans_num_list = []
            sub_sent_vector = []
            for line in tqdm(test_data):
                t1, t2 = pred(line)
                ans_num_list.append(t1)
                sub_sent_vector.append(t2)
            # print(ans_num_list)
            # print(sub_sent_vector)
            data_df['ans_num'] = ans_num_list
            data_df['sub_sent_vector'] = sub_sent_vector
            print('预测完成！！')
            return data_df
        train2 = process_df(train2)
        test2 = process_df(test2)

        train2 = train2.loc[:, ['sub_sents_tokenized','ans_num','sub_sent_vector']]
        test2 = test2.loc[:,['sub_sents_tokenized','ans_num','sub_sent_vector']]
        with open('train_test_ans.pkl', 'wb') as f:
            pickle.dump(train2, f, 0)
            pickle.dump(test2, f, 0)


