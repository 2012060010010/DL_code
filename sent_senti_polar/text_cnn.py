import tensorflow as tf
# import numpy as np
from tensorflow.contrib.rnn import LSTMCell
# from tensorflow.contrib.rnn import MultiRNNCell

class tsp_multiCNN(object):
    """
    主题、情感词和情感极性的多任务分类算法.
    使用词向量模型, 接下来是卷积层，池化层，softmanx层,
    lstm, attention, position_embeddin,和 多任务联合训练
    """
    def __init__(
      self, sequence_length, num_classes, ts_length, num_polar, embedding_size,
            filter_sizes, num_filters, vocab, lstm_hidden_dim, max_seq_len, l2_reg_lambda=0.0):

        # 输入，输出，dropout
        self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name="input_x")
        self.input_position = tf.placeholder(tf.float32, [None, sequence_length, 2], name="input_position")
        self.input_theme = tf.placeholder(tf.int32, [None, ts_length], name='input_theme')
        self.input_sentiment = tf.placeholder(tf.int32, [None, ts_length], name='input_sentiment')
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        self.input_p = tf.placeholder(tf.float32, [None, num_polar], name="input_p")

        self.zeros = tf.placeholder(tf.int64, [None, num_polar - 1], name="zeros")
        # print(self.input_y.get_shape)
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        self.sequence_len = tf.placeholder(tf.int32, shape=[None], name="sequence_lengths")
        self.hidden_dim = lstm_hidden_dim
        self.max_seq_len = max_seq_len


        # 词向量
        with tf.name_scope("embedding"):
            # 导入初始词向量
            self.W = tf.Variable(tf.constant(0.0, shape=[vocab.__len__(), embedding_size]), trainable=True,
                                 name="emb_W")
            self.embedding_placeholder = tf.placeholder(tf.float32, [vocab.__len__(), embedding_size])
            self.embedding_init = self.W.assign(self.embedding_placeholder)

            # 输入文本中的单词转换为词向量
            self.embedded_chars = tf.nn.embedding_lookup(self.W, self.input_x)
            # print(self.embedded_chars.get_shape())
            self.embedded_chars = tf.concat([self.embedded_chars, self.input_position], axis=2)
            # print(self.embedded_chars.get_shape())
            self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)

            # 候选pair中的单词转换为词向量
            self.embedded_theme = tf.nn.embedding_lookup(self.W, self.input_theme)
            self.embedded_sentiment = tf.nn.embedding_lookup(self.W, self.input_sentiment)
            self.theme = tf.reduce_mean(self.embedded_theme, axis=1)
            self.sentiment = tf.reduce_mean(self.embedded_sentiment, axis=1)
            self.embedded_pair = tf.concat([self.theme, self.sentiment], axis=1)
            # print(self.embedded_pair.get_shape)  # 128*200

        # 为filter_size创建创建一个卷积层和池化层
        pooled_outputs = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope("conv-maxpool-%s"% filter_size):
                # 卷积层
                filter_shape = [filter_size, embedding_size+2, 1, num_filters]
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
        #conv2D 输出结果进行展平
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])
        # print(self.h_pool_flat.get_shape)  # 128*180

        # 加入dropout
        with tf.name_scope("sentence_dropout"):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)
            # print(self.h_drop.get_shape)  # 128*180

        with tf.variable_scope("bi_lstm"):
            cell_fw = LSTMCell(self.hidden_dim)
            cell_bw = LSTMCell(self.hidden_dim)
            (output_fw_seq, output_bw_seq), _ = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=cell_fw,
                cell_bw=cell_bw,
                inputs=self.embedded_chars,
                sequence_length=self.sequence_len,
                dtype=tf.float32)
            out_lstm = tf.concat([output_fw_seq, output_bw_seq], axis=-1)
            self.out_lstm = tf.nn.dropout(out_lstm, self.dropout_keep_prob)
            # print(out_lstm[:,-1,:].get_shape())  # (128, 50, 200)

        # 定义attention layer
        with tf.name_scope('attention'):
            attention_w = tf.Variable(tf.truncated_normal([2 * self.hidden_dim, max_seq_len], stddev=0.1),
                                      name='attention_w')
            attention_b = tf.Variable(tf.constant(0.1, shape=[max_seq_len]), name='attention_b')
            u_list = []
            for t in range(max_seq_len):
                u_t = tf.tanh(tf.matmul(self.out_lstm[:, t, :], attention_w) + attention_b)
                # print(self.out_lstm[:,t,:].get_shape())   #128*200
                u_list.append(u_t)
            u_w = tf.Variable(tf.truncated_normal([max_seq_len, 1], stddev=0.1), name='attention_uw')
            attn_z = []
            for t in range(max_seq_len):
                z_t = tf.matmul(u_list[t], u_w)
                attn_z.append(z_t)
            # transform to batch_size * sequence_length
            attn_zconcat = tf.concat(attn_z, axis=1)
            self.alpha = tf.nn.softmax(attn_zconcat)
            # print(self.alpha.get_shape())   #128*40
            # transform to sequence_length * batch_size * 1 , same rank as outputs
            self.alpha_trans = tf.reshape(self.alpha, [-1, max_seq_len, 1])
            # print(self.alpha_trans.get_shape())  #128*40*1
            self.final_output = tf.reduce_sum(self.out_lstm * self.alpha_trans, 1)
            # print(self.final_output.get_shape())  #128*200

        with tf.name_scope('pair_fc'):
            # self.pair_flat = tf.reshape(self.embedded_pair, [-1, embedding_size*2])
            # print(self.pair_flat.get_shape)
            w_pair = tf.get_variable(
                shape=[self.embedded_pair.get_shape()[1], embedding_size],
                initializer=tf.contrib.layers.xavier_initializer(),
                name='w_pair'
            )
            b_pair = tf.Variable(tf.constant(0.1, shape=[embedding_size], name='b_pair'))
            self.pair_emb = tf.nn.xw_plus_b(self.embedded_pair, w_pair, b_pair, name='pair_emb')
            # print(self.pair_emb.get_shape)  # 128*100

        with tf.name_scope('merge_sent_pair'):
            # self.h_drop 128*180
            # self.pair_emb 128*100
            dim_compress = 200
            self.merge_sp = tf.concat([self.h_drop, self.pair_emb, self.final_output], axis=1, name='merge_sp')
            w_comp = tf.get_variable(
                "W_comp",  # self.merge_sp.get_shape()[1]
                shape=[self.merge_sp.get_shape()[1], dim_compress],  # 280 * 2
                initializer=tf.contrib.layers.xavier_initializer())
            b_comp = tf.Variable(tf.constant(0.1, shape=[dim_compress]), name="b_pair")
            self.h_compress = tf.nn.xw_plus_b(self.merge_sp, w_comp, b_comp, name='h_compress')
            # print(self.merge_sp.get_shape)  # 128*280

        # 全连接层 for ans
        with tf.name_scope("output_y"):
            wy = tf.get_variable(
                # self.merge_sp.get_shape()[1]
                "Wy",
                shape=[self.h_compress.get_shape()[1], num_classes],  #280 * 2
                initializer=tf.contrib.layers.xavier_initializer())
            by = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="by")
            self.l2_loss_y = tf.nn.l2_loss(wy)+tf.nn.l2_loss(by)
            self.scores_y = tf.nn.tanh(tf.nn.xw_plus_b(self.h_compress, wy, by), name="scores_y") # 128*2
            # self.clip_scores = tf.clip_by_value(self.scores, 0.1, 1)
            self.predictions_y = tf.argmax(self.scores_y, 1, name="predictions_y")  #128*1

        # 计算loss函数
        with tf.name_scope("loss_y"):
            self.losses_y = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.scores_y,
                                                                                 labels=self.input_y))
            self.loss_y = self.losses_y + l2_reg_lambda * self.l2_loss_y
        # 计算准确率
        with tf.name_scope("accuracy_y"):
            correct_predictions_y = tf.equal(self.predictions_y, tf.argmax(self.input_y, 1))
            self.accuracy_y = tf.reduce_mean(tf.cast(correct_predictions_y, "float"), name="accuracy_y")

        # 全连接层 for polar
        with tf.name_scope("output_p"):
            wp = tf.get_variable(
                # self.merge_sp.get_shape()[1]
                "Wp",
                shape=[self.h_compress.get_shape()[1], num_polar],  #280 * 2
                initializer=tf.contrib.layers.xavier_initializer())
            bp = tf.Variable(tf.constant(0.1, shape=[num_polar]), name="bp")
            self.l2_loss_p = tf.nn.l2_loss(wp)+tf.nn.l2_loss(bp)
            self.scores_ = tf.nn.tanh(tf.nn.xw_plus_b(self.h_compress, wp, bp), name="scores_p")  # 128*4
            #实现如下功能： 如果答案预测为0，那么强行把p_prediction 第四维加1，相当于预测为3
            pred_y = tf.reshape(self.predictions_y, [-1, 1])
            self.scores_p = tf.subtract(self.scores_, tf.to_float(tf.concat([self.zeros, tf.negative(pred_y)], axis=1)))
            # 如果答案为1,前3列加1，只可能预测0,1,2
            self.scores_p = tf.add(self.scores_p, tf.to_float(
                tf.concat([pred_y, pred_y, pred_y, tf.subtract(pred_y, pred_y)], axis=1)))
            self.predictions_p = tf.argmax(self.scores_p, 1, name="predictions_p")  # 128*1

        # 计算loss函数
        with tf.name_scope("loss_p"):
            self.losses_p = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.scores_p,
                                                                                 labels=self.input_p))
            self.loss_p = self.losses_p + l2_reg_lambda * self.l2_loss_p
        # 计算准确率
        with tf.name_scope("accuracy_p"):
            correct_predictions_p = tf.equal(self.predictions_p, tf.argmax(self.input_p, 1))
            self.accuracy_p = tf.reduce_mean(tf.cast(correct_predictions_p, "float"), name="accuracy_p")

        with tf.name_scope('loss'):
            self.loss = self.loss_y + self.loss_p


