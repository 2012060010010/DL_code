#! /usr/bin/env python
import tensorflow as tf
import numpy as np
import os
import time
import datetime
from sent_senti_polar import data_helpers
from sent_senti_polar import text_cnn
import codecs
# from tensorflow.contrib import learn

# 输入数据参数
tf.flags.DEFINE_float("dev_sample_percentage", 0.1, "Percentage of the training data to use for validation")
tf.flags.DEFINE_string("train_data_file", "../BDCI2017-taiyi/data4senti_polar.csv", "Data source for the training data.")
tf.flags.DEFINE_integer("max_document_length", 40, "Max length of each document")
tf.flags.DEFINE_integer("ts_length", 5, "theme and sentiment length")
tf.flags.DEFINE_string("wordvector_pretrain", "../w2v/fasttext_char_vec/fasttext_cbow_char.model.vec", "wordvector pretrain file")

# 模型的超参数
tf.flags.DEFINE_integer("embedding_dim", 100, "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_string("filter_sizes", "2,3,4,5", "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_integer("num_filters", 128, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.8, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.1, "L2 regularization lambda (default: 0.0)")
tf.flags.DEFINE_float("learning_rate", 1e-4, "Learning rate (default: 1e-4)")
tf.flags.DEFINE_float("learning_rate_decay", 0.97, "learning rate decay")

# 训练参数
tf.flags.DEFINE_integer("batch_size", 256, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 15, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 400, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 400, "Save model after this many steps (default: 100)")
tf.flags.DEFINE_integer("num_checkpoints", 4, "Number of checkpoints to store (default: 5)")
tf.flags.DEFINE_integer("lstm_hidden_dim", 100, "dimensions  of LSTM hidden cell ")

# Misc 参数
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()

for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))

max_document_length = FLAGS.max_document_length
ts_length = FLAGS.ts_length

# 导入数据
print("Loading data...")
sent_data, position_data, pair_data, label_data, polar_data = \
    data_helpers.load_train_datalabels(FLAGS.train_data_file, max_document_length)
print('label shape: ', label_data.shape)
# 构建词向量与词典
vocab, vocab2index, word_embed = data_helpers.get_fasttext(FLAGS.wordvector_pretrain)
print('vocab len: ', len(vocab))
print('word embed: ', word_embed.shape)


# vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length=max_document_length)
# x = np.array(list(vocab_processor.fit_transform(sent_data)))
# print(vocab_processor.vocabulary_)
x_text = np.array([data_helpers.sent2index(sent, max_document_length, vocab2index) for sent in sent_data])
x_position = np.array([data_helpers.padding_pos(pos, max_document_length) for pos in position_data])
# for i in range(len(x_position)):
#     if x_position[i].shape != x_position[0].shape:
#         print(i, x_position[i])

x_theme = np.array([data_helpers.sent2index(it[0], ts_length, vocab2index) for it in pair_data])
x_sentiment = np.array([data_helpers.sent2index(it[1], ts_length, vocab2index) for it in pair_data])
x_length = np.array([min(len(sent), max_document_length) for sent in sent_data])

# print(x_text)
# print(x_theme)
# print(x_sentiment)
# print(word_embed[3156])

# 随机shuffle数据
np.random.seed(10)
shuffle_indices = np.random.permutation(np.arange(len(label_data)))
x_text_shuffled = x_text[shuffle_indices]
x_position_shuffled = x_position[shuffle_indices]
x_theme_shuffled = x_theme[shuffle_indices]
x_sentiment_shuffled = x_sentiment[shuffle_indices]
x_length_shuffled = x_length[shuffle_indices]
y_shuffled = label_data[shuffle_indices]
p_shuffled = polar_data[shuffle_indices]

dev_sample_index = -1 * int(FLAGS.dev_sample_percentage * float(len(y_shuffled)))
text_train = x_text_shuffled[:dev_sample_index]
positon_train = x_position_shuffled[:dev_sample_index]
theme_train = x_theme_shuffled[:dev_sample_index]
sentiment_train = x_sentiment_shuffled[:dev_sample_index]
x_length_train = x_length_shuffled[:dev_sample_index]
y_train = y_shuffled[:dev_sample_index]
p_train = p_shuffled[:dev_sample_index]

print('train: ', text_train.shape, y_train.shape, p_train.shape)

text_dev = x_text_shuffled[dev_sample_index:]
position_dev = x_position_shuffled[dev_sample_index:]
theme_dev = x_theme_shuffled[dev_sample_index:]
sentiment_dev = x_sentiment_shuffled[dev_sample_index:]
x_length_dev = x_length_shuffled[dev_sample_index:]
y_dev = y_shuffled[dev_sample_index:]
p_dev = p_shuffled[dev_sample_index:]

print('test: ', text_dev.shape, y_dev.shape, p_dev.shape)

# 训练
# ==================================================

with tf.Graph().as_default():
    session_conf = tf.ConfigProto(
      allow_soft_placement=FLAGS.allow_soft_placement,
      log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        cnn = text_cnn.tsp_multiCNN(
            sequence_length=text_train.shape[1],
            num_classes=y_train.shape[1],
            ts_length=ts_length,
            num_polar=p_train.shape[1],
            embedding_size=FLAGS.embedding_dim,
            filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
            num_filters=FLAGS.num_filters,
            vocab=vocab,
            lstm_hidden_dim=FLAGS.lstm_hidden_dim,
            max_seq_len=max_document_length,
            l2_reg_lambda=FLAGS.l2_reg_lambda)

        # 定义训练阶段
        global_step = tf.Variable(0, name="global_step", trainable=False)
        # learning_rate = tf.train.exponential_decay(
        #     FLAGS.learning_rate,    # 基础学习率
        #     global_step,            # 当前迭代的轮数
        #     int(text_train.shape[1]/ FLAGS.batch_size),    # 过完所有的训练数据需要的迭代次数
        #     FLAGS.learning_rate_decay   # 学习率衰减速度
        # )
        learning_rate = FLAGS.learning_rate
        optimizer = tf.train.AdamOptimizer(learning_rate)
        grads_and_vars = optimizer.compute_gradients(cnn.loss)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

        # Output directory for models and summaries
        timestamp = time.asctime().replace(' ', '_').replace(':', '_')
        out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
        print("Writing to {}\n".format(out_dir))

        # 收集loss和acc
        loss_y_summary = tf.summary.scalar("loss_y", cnn.loss_y)
        loss_p_summary = tf.summary.scalar("loss_p", cnn.loss_p)
        loss_summary = tf.summary.scalar("loss", cnn.loss)
        acc_y_summary = tf.summary.scalar("accuracy_y", cnn.accuracy_y)
        acc_p_summary = tf.summary.scalar("accuracy_p", cnn.accuracy_p)

        # 训练阶段数据收集
        train_summary_op = tf.summary.merge([loss_y_summary, loss_p_summary, loss_summary,
                                             acc_y_summary, acc_p_summary])
        train_summary_dir = "summary/train"
        train_summary_writer = tf.summary.FileWriter(logdir=train_summary_dir, graph=tf.get_default_graph(), max_queue=1)

        # Dev summaries
        dev_summary_op = tf.summary.merge([loss_y_summary, loss_p_summary, loss_summary,
                                           acc_y_summary, acc_p_summary])
        dev_summary_dir = "summary/dev"
        dev_summary_writer = tf.summary.FileWriter(logdir=dev_summary_dir,graph=tf.get_default_graph(),max_queue=1)

        # Checkpoint输出路径
        checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)

        print('开始写入log！！！\n')
        log_file_name = os.path.join(checkpoint_dir, "log.txt")
        log_file = codecs.open(log_file_name, 'w', encoding='utf-8')
        log_file.write('模型参数：\n')
        for attr, value in sorted(FLAGS.__flags.items()):
            log_file.write("{}={}\n".format(attr.upper(), value))
        print('模型参数写入！！\n')
        # 初始化所有的变量
        sess.run(tf.global_variables_initializer())
        sess.run(cnn.embedding_init, feed_dict={cnn.embedding_placeholder: word_embed})
        print("tensorboard --logdir=PycharmProjects/taiyi_SA/semi_compelition/sent_senti_polar/summary")

        def train_step(text_batch, position_batch, theme_batch, sentiment_batch, x_len_batch, y_batch, p_batch, zeros, epoch):
            # print(type(position_batch))
            # print(type(text_batch))
            # print(position_batch.shape)
            # print(position_batch)
            feed_dict = {
                cnn.input_x: text_batch,
                cnn.input_position: position_batch,
                cnn.input_theme: theme_batch,
                cnn.input_sentiment: sentiment_batch,
                cnn.sequence_len: x_len_batch,
                cnn.input_y: y_batch,
                cnn.input_p: p_batch,
                cnn.zeros: zeros,
                cnn.dropout_keep_prob: FLAGS.dropout_keep_prob
            }
            _, step, summaries, loss, accuracy_y, accuracy_p = sess.run(
                [train_op, global_step,
                 train_summary_op,
                 cnn.loss, cnn.accuracy_y,
                 cnn.accuracy_p],
                feed_dict)
            time_str = datetime.datetime.now().isoformat()
            if step % 40 == 0:
                print("epoch {}:\n{}: train step {}, loss {:g}, acc_y {:g}, acc_p {:g}".format(
                    epoch, time_str, step, loss, accuracy_y, accuracy_p))
            train_summary_writer.add_summary(summaries, step)

        def get_f1_y(pred, truth):
            pred_label = pred
            true_label = np.argmax(truth, axis=1)
            fn = 0
            fp = 0
            tn = 0
            tp = 0
            ac = 0
            for it in zip(pred_label, true_label):
                if it[0] and it[1]:
                    tp += 1
                if not it[0] and it[1]:
                    fn += 1
                if it[0] and not it[1]:
                    fp += 1
                if not it[0] and not it[1]:
                    tn += 1
                if it[0] == it[1]:
                    ac += 1
            p = tp / (tp + fp + 0.000001)
            r = tp / (tp + fn + 0.000001)
            f1 = p * r * 2 / (p + r + 0.000001)
            acc = ac / pred.shape[0]
            return p, r, f1, acc

        def get_f1_p(pred, truth):
            pred_label = pred
            true_label = np.argmax(truth, axis=1)
            fn = 0
            fp = 0
            tn = 0
            tp = 0
            ac = 0
            for it in zip(pred_label, true_label):
                if it[0] == it[1] and it[0] != 3:
                    tp += 1
                if it[0] != it[1] and it[1] != 3:
                    fn += 1
                if it[0] != 3 and it[1] == 3:
                    fp += 1
                if it[0] == it[1] and it[1] == 3:
                    tn += 1
                if it[0] == it[1]:
                    ac += 1
            p = tp / (tp + fp + 0.000001)
            r = tp / (tp + fn + 0.000001)
            f1 = p * r * 2 / (p + r + 0.0000001)
            acc = ac / pred.shape[0]
            return p, r, f1, acc

        def get_acc_sp(prediction_y, y_batch, prediction_p, p_batch):
            pred_y = prediction_y
            true_y = np.argmax(y_batch, axis=1)
            pred_p = prediction_p
            true_p = np.argmax(p_batch,axis=1)
            ac = 0
            ac1 = 0
            # print(type(pred_y),type(true_y),type(pred_p),type(true_p))
            # print(p_batch)
            for it in zip(pred_y, true_y, pred_p, true_p):
                if it[0]:
                    if it[1] and it[2] == it[3]:
                        ac += 1
                if it[0] == it[1] and it[0] == 1:
                    ac1 += 1
            acc = ac / sum(true_y)
            acc1 = ac1 / sum(true_y)
            return acc, acc1, sum(pred_y), sum(true_y), true_y.shape[0]
        def dev_step(text_batch, position_batch, theme_batch, sentiment_batch, x_len_batch, y_batch, p_batch, zeros, epoch):
            feed_dict = {
                cnn.input_x: text_batch,
                cnn.input_position: position_batch,
                cnn.input_theme: theme_batch,
                cnn.input_sentiment: sentiment_batch,
                cnn.sequence_len: x_len_batch,
                cnn.input_y: y_batch,
                cnn.input_p: p_batch,
                cnn.zeros: zeros,
                cnn.dropout_keep_prob: 1.0
            }
            step, summaries, loss, accuracy_y, accuracy_p, prediction_y, prediction_p = sess.run(
                [global_step, dev_summary_op,
                 cnn.loss, cnn.accuracy_y, cnn.accuracy_p, cnn.predictions_y, cnn.predictions_p],
                feed_dict)
            time_str = datetime.datetime.now().isoformat()
            if step % 100 == 0:
                print("epoch {}:\n{}: test step {}, loss {:g}, acc_y {:g}, acc_p {:g}".format(
                    epoch, time_str, step, loss, accuracy_y, accuracy_p))
                print(get_f1_y(prediction_y, y_batch))
                print(get_f1_p(prediction_p, p_batch))
                print(get_acc_sp(prediction_y, y_batch, prediction_p, p_batch))
                log_file.write("epoch {}:\n{}: test step {}, loss {:g}, acc_y {:g}, acc_p {:g}\n".format(
                    epoch, time_str, step, loss, accuracy_y, accuracy_p))
                log_file.write(str(get_f1_y(prediction_y, y_batch))+'\n')
                log_file.write(str(get_f1_p(prediction_p, p_batch))+'\n')
                log_file.write(str(get_acc_sp(prediction_y, y_batch, prediction_p, p_batch))+'\n')

            dev_summary_writer.add_summary(summaries, step)

        for epoch in range(FLAGS.num_epochs):
            # 获取批量输入数据
            batches = data_helpers.batch_iter(
                list(zip(text_train, positon_train, theme_train, sentiment_train, x_length_train, y_train, p_train)), FLAGS.batch_size, False)
            # 训练每一批的数据
            # batch_ind=0
            for batch in batches:
                text_batch, position_batch, theme_batch, sentiment_batch, x_len_batch, y_batch, p_batch = zip(*batch)
                # batch_ind += 1
                # print(p_batch.shape[0])
                zeros = np.zeros([len(p_batch), p_train.shape[1]-1], int)
                train_step(text_batch, position_batch, theme_batch, sentiment_batch,
                           x_len_batch, y_batch, p_batch, zeros, epoch)
                current_step = tf.train.global_step(sess, global_step)
                if current_step % FLAGS.evaluate_every == 0:
                    print("\nEvaluation:")
                    zeros_dev = np.zeros([p_dev.shape[0], p_dev.shape[1]-1], int)
                    dev_step(text_dev, position_dev, theme_dev, sentiment_dev, x_length_dev, y_dev, p_dev, zeros_dev, epoch)
                if current_step % FLAGS.checkpoint_every == 0:
                    path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                    print("Saved model checkpoint to {}\n".format(path))
                    log_file.write("Saved model checkpoint to {}\n".format(path))


