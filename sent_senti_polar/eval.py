#! /usr/bin/env python
import tensorflow as tf
import numpy as np
import os
import time
import datetime
import pickle
from tqdm import tqdm
from sent_senti_polar import data_helpers
import pandas as pd
# from sent_senti_polar import text_cnn
# from tensorflow.contrib import learn

# 输入数据参数
tf.flags.DEFINE_string("test_data_file", "../BDCI2017-taiyi/submission-localValid.csv", "Data source for the training data.")
tf.flags.DEFINE_integer("max_document_length", 40, "Max length of each document")
tf.flags.DEFINE_integer("ts_length", 5, "theme and sentiment length")
tf.flags.DEFINE_string("wordvector_pretrain", "../w2v/fasttext_char_vec/fasttext_cbow_char.model.vec", "wordvector pretrain file")
tf.flags.DEFINE_string("checkpoint_dir", "./runs/Sun_Dec_10_17_48_01_2017/checkpoints", "dir of checkpoints")

# Misc 参数
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()

for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))

# 导入数据
print("Loading data...")
# train2, test2 = data_helpers.load_test_data(FLAGS.test_data_file)
test2 = pd.read_csv(FLAGS.test_data_file, encoding='utf-8')
# print(test2.columns)
# exit(1)
# 构建词向量与词典
vocab, vocab2index, word_embed = data_helpers.get_fasttext(FLAGS.wordvector_pretrain)
print('vocab len: ', len(vocab))
print('word embed: ', word_embed.shape)
max_document_length = FLAGS.max_document_length
ts_length = FLAGS.ts_length

print("\nEvaluating...\n")
# print(test_data[2])
# print(type(test_data[2]))
# print(vocab[3154])
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
        input_theme = graph.get_operation_by_name("input_theme").outputs[0]
        input_position = graph.get_operation_by_name("input_position").outputs[0]
        input_sentiment = graph.get_operation_by_name("input_sentiment").outputs[0]
        sequence_len = graph.get_operation_by_name("sequence_lengths").outputs[0]
        input_zeros = graph.get_operation_by_name("zeros").outputs[0]

        # W = graph.get_operation_by_name("embedding/emb_W").outputs[0]
        # h_compress = graph.get_operation_by_name("sentence_dropout/h_compress").outputs[0]
        predictions_y = graph.get_operation_by_name("output_y/predictions_y").outputs[0]
        predictions_p = graph.get_operation_by_name("output_p/predictions_p").outputs[0]

        dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]
        # 验证的Tensor

        print('开始预测！！')
        def pred(row):
            ans_theme = []
            ans_sentiment = []
            polar = []
            sent_pair = data_helpers.make_test_data(row)
            if len(sent_pair) == 0:
                return ans_theme, ans_sentiment, polar
            for it in sent_pair:
                sent = ' '.join(''.join(it[0]))
                sub_sent = ''.join(it[0])
                theme = it[1]
                sentiment = it[2]
                if theme == 'NULL':
                    theme_s = 0
                    theme_e = 0
                else:
                    theme_s = sub_sent.index(theme)
                    theme_e = theme_s + len(theme) - 1
                theme_pos = []
                for ind in range(min(len(sub_sent), max_document_length)):
                    if ind < theme_s:
                        theme_pos.append(ind - theme_s)
                    if ind >= theme_s and ind <= theme_e:
                        theme_pos.append(0)
                    if ind > theme_e:
                        theme_pos.append(ind - theme_e)

                # sentiment = train_df.loc[i, 'sentiment']
                # sentiment_l = ' '.join(sentiment)
                senti_s = sub_sent.index(sentiment)
                senti_e = senti_s + len(sentiment) - 1
                # print(sub_sent, sentiment, senti_s, senti_e)
                senti_pos = []
                for ind in range(min(len(sub_sent), max_document_length)):
                    if ind < senti_s:
                        senti_pos.append(ind - senti_s)
                    if ind >= senti_s and ind <= senti_e:
                        senti_pos.append(0)
                    if ind > senti_e:
                        senti_pos.append(ind - senti_e)
                pos = [[it[0], it[1]] for it in zip(theme_pos, senti_pos)]

                x_text = np.array([data_helpers.sent2index(sent, max_document_length, vocab2index)])
                x_position = np.array([data_helpers.padding_pos(pos, max_document_length)])
                x_theme = np.array([data_helpers.sent2index(' '.join(it[1]), ts_length, vocab2index)])
                x_sentiment = np.array([data_helpers.sent2index(' '.join(it[2]), ts_length, vocab2index)])
                x_length = np.array([min(len(sent.replace(' ', '')), max_document_length)])
                zeros = np.zeros([1, 3], int)
                # print(sent)
                pred_y, pred_p = sess.run([predictions_y, predictions_p],
                                          {input_x: x_text,
                                           input_position: x_position,
                                           input_theme: x_theme,
                                           input_sentiment: x_sentiment,
                                           sequence_len: x_length,
                                           # W: word_embed,
                                           input_zeros: zeros,
                                           dropout_keep_prob: 1.0,
                                           })
                if pred_y[0]:
                    ans_theme.append(it[1])
                    ans_sentiment.append(it[2])
                    polar.append(pred_p[0]-1)
            # print(row, ans_theme, ans_sentiment, polar)
            return ans_theme, ans_sentiment, polar

        def process_df(data_df):
            test_data = list(data_df['content'])
            print('test: ', data_df.shape)
            theme_list = []
            sentiment_list = []
            polar_list = []
            for line in tqdm(test_data):
                t, s, p = pred(line)
                if len(t) > 0:
                    theme_list.append(';'.join(t)+';')
                    sentiment_list.append(';'.join(s) + ';')
                    p = [str(it) for it in p]
                    polar_list.append(';'.join(p) + ';')
                else:
                    theme_list.append('')
                    sentiment_list.append('')
                    polar_list.append('')

            # print(ans_num_list)
            # print(sub_sent_vector)
            data_df['theme'] = theme_list
            data_df['sentiment_word'] = sentiment_list
            data_df['sentiment_anls'] = polar_list
            print('预测完成！！')
            return data_df
        # train2 = process_df(train2)
        test2 = process_df(test2)
        test2.to_csv('train_localValid.csv', index=None, encoding='utf-8-sig')
        # with open('train_test.pkl', 'wb') as f:
        #     # pickle.dump(train2, f)
        #     pickle.dump(test2, f)


