#! /usr/bin/env python

import tensorflow as tf
import numpy as np
import os
import time
import datetime
from sent_has_ans import data_helpers
from sent_has_ans import text_cnn
from tensorflow.contrib import learn

# 输入数据参数
tf.flags.DEFINE_float("dev_sample_percentage", 0.1, "Percentage of the training data to use for validation")
tf.flags.DEFINE_string("train_data_file", "../BDCI2017-taiyi/data4sent_has_ans_new.txt", "Data source for the training data.")
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

# 训练参数
tf.flags.DEFINE_integer("batch_size", 128, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 10, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 400, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 400, "Save model after this many steps (default: 100)")
tf.flags.DEFINE_integer("num_checkpoints", 4, "Number of checkpoints to store (default: 5)")

# Misc 参数
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()

for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))

# 导入数据
print("Loading data...")
sent_data,label_data= data_helpers.load_train_datalabels(FLAGS.train_data_file)
print('label shape: ',label_data.shape)
# 构建词向量与词典
vocab,vocab2index,word_embed = data_helpers.get_fasttext(FLAGS.wordvector_pretrain)
print('vocab len: ', len(vocab))
print('word embed: ', word_embed.shape)

max_document_length = FLAGS.max_document_length
# vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length=max_document_length)
# x = np.array(list(vocab_processor.fit_transform(sent_data)))
# print(vocab_processor.vocabulary_)
x_text=np.array([data_helpers.sent2index(sent,max_document_length,vocab2index) for sent in sent_data])

# 随机shuffle数据
np.random.seed(10)
shuffle_indices = np.random.permutation(np.arange(len(label_data)))
x_text_shuffled = x_text[shuffle_indices]
y_shuffled = label_data[shuffle_indices]

dev_sample_index = -1 * int(FLAGS.dev_sample_percentage * float(len(y_shuffled)))
text_train = x_text_shuffled[:dev_sample_index]
y_train = y_shuffled[:dev_sample_index]
print('train: ',text_train.shape,y_train.shape)

text_dev = x_text_shuffled[dev_sample_index:]
y_dev = y_shuffled[dev_sample_index:]
print('test: ',text_dev.shape,y_dev.shape)

# 训练
# ==================================================

with tf.Graph().as_default():

    session_conf = tf.ConfigProto(
      allow_soft_placement = FLAGS.allow_soft_placement,
      log_device_placement = FLAGS.log_device_placement)
    sess = tf.Session(config = session_conf)
    with sess.as_default():
        cnn = text_cnn.has_ans_CNN(
            sequence_length = text_train.shape[1],
            num_classes = y_train.shape[1],
            embedding_size = FLAGS.embedding_dim,
            filter_sizes = list(map(int, FLAGS.filter_sizes.split(","))),
            num_filters = FLAGS.num_filters,
            vocab = vocab,
            l2_reg_lambda = FLAGS.l2_reg_lambda)
        # print(list(map(int, FLAGS.filter_sizes.split(","))))
        # 定义训练阶段
        global_step = tf.Variable(0, name="global_step", trainable=False)
        # learning_rate = tf.train.exponential_decay(
        #     FLAGS.learning_rate,    # 基础学习率
        #     global_step,            # 当前迭代的轮数
        #     int(text_train.shape[1]/ FLAGS.batch_size),    # 过完所有的训练数据需要的迭代次数
        #     FLAGS.learning_rate_decay   # 学习率衰减速度
        # )
        learning_rate=FLAGS.learning_rate
        optimizer = tf.train.AdamOptimizer(learning_rate)
        grads_and_vars = optimizer.compute_gradients(cnn.loss)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

        # Output directory for models and summaries
        timestamp = time.asctime().replace(' ', '_').replace(':', '_')
        out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
        print("Writing to {}\n".format(out_dir))

        # 收集loss和acc
        loss_summary = tf.summary.scalar("loss", cnn.loss)
        loss_entropy = tf.summary.scalar("loss_entropy", cnn.losses)
        loss_l2 = tf.summary.scalar("loss_l2", cnn.l2_loss)
        acc_summary = tf.summary.scalar("accuracy", cnn.accuracy)

        # 训练阶段数据收集
        train_summary_op = tf.summary.merge([loss_summary,loss_entropy,loss_l2, acc_summary])
        train_summary_dir = "summary/train"
        train_summary_writer = tf.summary.FileWriter(logdir=train_summary_dir, graph=tf.get_default_graph(), max_queue=1)

        # Dev summaries
        dev_summary_op = tf.summary.merge([loss_summary,loss_entropy,loss_l2, acc_summary])
        dev_summary_dir = "summary/dev"
        dev_summary_writer = tf.summary.FileWriter(logdir=dev_summary_dir,graph=tf.get_default_graph(),max_queue=1)

        # Checkpoint输出路径
        checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)

        # 初始化所有的变量
        sess.run(tf.global_variables_initializer())
        sess.run(cnn.embedding_init,feed_dict={cnn.embedding_placeholder: word_embed})
        print("tensorboard --logdir=PycharmProjects/taiyi_SA/semi_compelition/sent_has_ans/summary")

        def train_step(text_batch, y_batch, epoch):
            feed_dict = {
              cnn.input_x: text_batch,
              cnn.input_y: y_batch,
              cnn.dropout_keep_prob: FLAGS.dropout_keep_prob
            }
            # print(theme_batch)
            _, step, summaries, loss, accuracy = sess.run(
                [train_op, global_step,
                 train_summary_op,
                 cnn.loss, cnn.accuracy],
                feed_dict)
            time_str = datetime.datetime.now().isoformat()
            if step % 20 == 0:
                print("epoch {}:\n{}: train step {}, loss {:g}, acc {:g}".format(epoch, time_str, step, loss, accuracy))
            train_summary_writer.add_summary(summaries, step)
        def get_f1(pred, truth):
            pred_label = pred
            true_label = np.argmax(truth, axis=1)
            FN = 0
            FP = 0
            TN = 0
            TP = 0
            AC = 0
            for it in zip(pred_label, true_label):
                if it[0] and it[1]:
                    TP += 1
                if not it[0] and it[1]:
                    FN += 1
                if it[0] and not it[1]:
                    FP += 1
                if not it[0] and not it[1]:
                    TN += 1
                if it[0] == it[1]:
                    AC += 1
            P = TP / (TP + FP)
            R = TP / (TP + FN)
            F1 = P * R * 2 / (P + R)
            ACC = AC / pred.shape[0]
            return P, R, F1, ACC
        def dev_step(text_batch, y_batch, epoch):
            feed_dict = {
              cnn.input_x: text_batch,
              cnn.input_y: y_batch,
              cnn.dropout_keep_prob: 1.0
            }
            step, summaries, loss, accuracy, prediction = sess.run(
                [global_step, dev_summary_op,
                 cnn.loss, cnn.accuracy, cnn.predictions],
                feed_dict)
            time_str = datetime.datetime.now().isoformat()
            if step % 50 == 0:
                print("epoch {}:\n{}: test step {}, loss {:g}, acc {:g}".format(epoch, time_str, step, loss, accuracy))
                print(get_f1(prediction,y_batch))
            dev_summary_writer.add_summary(summaries, step)

        for epoch in range(FLAGS.num_epochs):
            # 获取批量输入数据
            batches = data_helpers.batch_iter(
                list(zip(text_train, y_train)), FLAGS.batch_size, False)
            # 训练每一批的数据
            # batch_ind=0
            for batch in batches:
                text_batch, y_batch = zip(*batch)
                # batch_ind += 1
                train_step(text_batch, y_batch, epoch)
                current_step = tf.train.global_step(sess, global_step)
                if current_step % FLAGS.evaluate_every == 0:
                    print("\nEvaluation:")
                    dev_step(text_dev, y_dev, epoch)
                if current_step % FLAGS.checkpoint_every == 0:
                    path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                    print("Saved model checkpoint to {}\n".format(path))