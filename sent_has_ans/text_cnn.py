import tensorflow as tf
import numpy as np

class has_ans_CNN(object):
    """
    CNN 文本分类算法.
    使用词向量模型, 接下来是卷积层，池化层，softmanx层.
    """
    def __init__(
      self, sequence_length, num_classes, embedding_size,
            filter_sizes, num_filters, vocab, l2_reg_lambda=0.0):

        # 输入，输出，dropout
        self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        # print(self.input_y.get_shape)
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        # 词向量
        with tf.name_scope("embedding"):
            # 导入初始词向量
            self.W = tf.Variable(tf.constant(0.0, shape=[vocab.__len__(), embedding_size]), trainable=False, name="WW")
            self.embedding_placeholder = tf.placeholder(tf.float32, [vocab.__len__(), embedding_size])
            self.embedding_init = self.W.assign(self.embedding_placeholder)

            # 输入文本中的单词转换为词向量
            self.embedded_chars = tf.nn.embedding_lookup(self.W, self.input_x)
            self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)

        # 为filter_size创建创建一个卷积层和池化层
        pooled_outputs = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope("conv-maxpool-%s"% filter_size):
                # 卷积层
                filter_shape = [filter_size, embedding_size, 1, num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
                conv = tf.nn.conv2d(
                    self.embedded_chars_expanded,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")
                # 使用非线性函数
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                # 池化层Max Pooling
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, sequence_length - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")
                pooled_outputs.append(pooled)
        # 将filter_size中的层合并在一起
        num_filters_total = num_filters * len(filter_sizes)
        self.h_pool = tf.concat(pooled_outputs, 3)
        # print(pooled_outputs[0].get_shape)  # 128*1*1*180
        # print(self.h_pool.get_shape)  # 128*1*1*180
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])
        # print(self.h_pool_flat.get_shape)  # 128*180

        # 加入dropout
        with tf.name_scope("sentence_dropout"):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)
            Wp = tf.get_variable(
                "Wp",  #self.merge_sp.get_shape()[1]
                shape=[self.h_drop.get_shape()[1], 50],  #280 * 2
                initializer = tf.contrib.layers.xavier_initializer())
            bp = tf.Variable(tf.constant(0.1, shape=[50]), name="bp")
            self.h_compress = tf.nn.xw_plus_b(self.h_drop, Wp, bp, name='h_compress')
            # print(self.h_drop.get_shape)  # 128*180

        # 全连接层
        with tf.name_scope("output"):
            Wo = tf.get_variable(
                "Wo",  #self.merge_sp.get_shape()[1]
                shape=[self.h_compress.get_shape()[1], num_classes],  #280 * 2
                initializer = tf.contrib.layers.xavier_initializer())
            bo = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="bo")
            self.l2_loss=tf.nn.l2_loss(Wo)+tf.nn.l2_loss(bo)
            self.scores = tf.nn.tanh(tf.nn.xw_plus_b(self.h_compress, Wo, bo), name="scores") # 128*2
            self.predictions = tf.argmax(self.scores, 1, name="predictions")  #128*1
        # 计算loss函数
        with tf.name_scope("loss"):
            self.losses = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y))
            self.loss = self.losses + l2_reg_lambda * self.l2_loss

        # 计算准确率s
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
