import pdb
import random
import copy
from collections import defaultdict

import numpy as np
import tensorflow as tf
from tensorflow.contrib import nn, rnn, legacy_seq2seq

import data_utils


class S2SModel(object):
    def __init__(self,
                 source_vocab_size,  # source源词典词汇数目大小
                 target_vocab_size,  # target目标词典词汇数目大小
                 buckets,  # 桶的大小
                 size,  # LSTM每层神经元数量, 也就是LSTM输出的维度大小
                 dropout,  # dropout保留率
                 num_layers,  # 网络层数
                 max_gradient_norm,  # 梯度最大阈值
                 batch_size,  # 批次大小
                 learning_rate,  # 学习率
                 num_samples,  # 负采样的样本数目
                 forward_only=False,  # 是否只有前向，也就是是否进行训练
                 dtype=tf.float32):
        # init member variales
        self.source_vocab_size = source_vocab_size
        self.target_vocab_size = target_vocab_size
        self.buckets = buckets
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        # LSTM cells(Multi RNN Cell, num_layers)
        # 定义多层 lstm cell细胞
        cell = rnn.BasicLSTMCell(size)
        cell = rnn.DropoutWrapper(cell, output_keep_prob=dropout)
        cell = rnn.MultiRNNCell([cell] * num_layers)
        # 定义一个字典，默认value为list的字典
        self.bucket_to_summary_list = defaultdict(list)

        # 设定 输出映射
        output_projection = None
        # 设定 交叉熵损失函数，采用负采样损失函数
        softmax_loss_function = None

        # 如果vocabulary太大，我们还是按照vocabulary来sample的话，内存会爆
        if num_samples > 0 and num_samples < self.target_vocab_size:
            print('开启投影：{}'.format(num_samples))
            # 投影，字符数，负采样的数
            w_t = tf.get_variable("proj_w", dtype=dtype,
                                  shape=[self.target_vocab_size, size])
            # 进行转制操作
            w = tf.transpose(w_t)
            # 设置 偏置项 b
            b = tf.get_variable(
                "proj_b",
                [self.target_vocab_size],
                dtype=dtype
            )

            # 预测过程中，设置投影，由小变大
            output_projection = (w, b)  # 仅在预测过程中使用，训练过程中不使用

            # 损失函数
            def sampled_loss(labels, logits):
                labels = tf.reshape(labels, [-1, 1])
                # 因为选项有选fp16的训练，这里全部转换为fp32
                local_w_t = tf.cast(w_t, tf.float32)
                local_b = tf.cast(b, tf.float32)
                local_inputs = tf.cast(logits, tf.float32)
                return tf.cast(
                    tf.nn.sampled_softmax_loss(
                        weights=local_w_t,
                        biases=local_b,
                        labels=labels,
                        inputs=local_inputs,  # logits
                        num_sampled=num_samples,
                        num_classes=self.target_vocab_size
                    ),
                    dtype
                )

            softmax_loss_function = sampled_loss

        # seq2seq_f seq前向操作
        def seq2seq_f(encoder_inputs, decoder_inputs, do_decode):
            print("当前桶的Seq2Seq模型构建.....")
            # encoder 先将cell及逆行deepcopy 因为seq2seq模型是两个相同的模型（encoder和decoder），但是模型参数不共享，所以encoder和decoder要使用两个不同的RNNcell
            tmp_cell = copy.deepcopy(cell)

            # cell：RNNCell常见的一些RNNCell定义都可以用
            # num_encoder_symbols：source的vocab_size大小，用于embedding矩阵定义
            # num_decoder_symbols：source的vocab_size大小，用于embedding矩阵定义
            # embedding_size：embedding向量的维度
            # num_heads：Attention头的个数，就是使用多少中attention的加权方式，用更多的参数来求出集中attention向量
            # output_projection：输出的映射层，因为decoder输出的维度是output_size，所以想要得到num_decoder_symbols对应的词还需要增加一个映射层， 仅用于预测过程
            # feed_previous：是否将上一时刻输出作为下一时刻输入，一般测试的时候设置为True，此时decoder_inputs除了第一个元素之外其他元素都不会使用， 仅用于预测过程

            return legacy_seq2seq.embedding_attention_seq2seq(
                encoder_inputs,
                decoder_inputs,
                tmp_cell,  # 自定义的cell，可以是GRU/LSTM，设置multiayer等
                num_encoder_symbols=source_vocab_size,  # 词典大小
                num_decoder_symbols=target_vocab_size,  # 目标词典大小
                embedding_size=size,  # embedding维度
                output_projection=output_projection,  # 不设定的化输出的维度可能很大（取决于此表大小），设定的话投射到一个低维向量
                feed_previous=do_decode,
                dtype=dtype
            )

        print("开始构建模型输入占位符.....")
        # inputs
        self.encoder_inputs = []  # 编码器输入
        self.decoder_inputs = []  # 解码器输入
        self.decoder_weights = []  # Loss损失函数计算的权重系数
        # encoder_inputs 这个列表对象中的每一个元素表示一个占位符，起名字分别为enconder0,encoder1....encoder{i}的几何意义是编码器再时刻i的输入
        # buckets中的最后一个是最大的（即第“-1”个）
        for i in range(buckets[-1][0]):
            self.encoder_inputs.append(tf.placeholder(
                tf.int32,
                shape=[None],
                name='encoder_input_{}'.format(i)
            ))
        # 输出比输入大 1，这是为了保证下面的targets可以向左shift 1位<空出一个结束符号>
        for i in range(buckets[-1][1] + 1):
            self.decoder_inputs.append(tf.placeholder(
                tf.int32,
                shape=[None],
                name='decoder_input_{}'.format(i)
            ))
            self.decoder_weights.append(tf.placeholder(
                dtype,
                shape=[None],
                name='decoder_weight_{}'.format(i)
            ))
        targets = [
            self.decoder_inputs[i + 1] for i in range(buckets[-1][1])
        ]

        print("开始构建模型....")
        # 跟language model类似，targets变量是decoder inputs 平移一个单位的结果，
        # encoder Inputs ：encoder的输入，一个tensor的列表，列表中每一项都是encoder时的一个词（batch)
        # decoder_inpits :decoder的输入，同上
        # targets ：目标值，与decoder_inputs只相差一个<eos>符号，int32型
        # buckets ：就是定义的bucket的值（编码器数据长度，解码器数据长度），是一个列表
        # seq2seq:定义好的seq2seq模型，可以使用后面介绍的embedding_attention_seq2seq,embedding_rnn_seq2seq,basic_rnn_seq2等
        # softmax_loss_fuction:计算误差的函数（labels,logits)默认为sqarse_softmax_cross_entroy_with_logits
        # per_example_loss:如果为真，则调用sequence_loss_by_example,返回一个列表，其每个元素就是一个样本的loss值，
        # 如果为假，则调用sequence_loss函数，对一个
        if forward_only:  # 测试阶段
            self.outputs, self.losses = legacy_seq2seq.model_with_buckets(
                self.encoder_inputs,  # 编码器输入
                self.decoder_inputs,  # 解码器输入
                targets,  # 实际值, 仅在loss损失函数构建的时候使用
                self.decoder_weights,  # 解码器权重
                buckets,  # 盒子
                lambda x, y: seq2seq_f(x, y, True),  # seq操作
                softmax_loss_function=softmax_loss_function  # 损失函数
            )
            if output_projection is not None:
                for b in range(len(buckets)):
                    self.outputs[b] = [
                        tf.matmul(
                            output,
                            output_projection[0]
                        ) + output_projection[1]
                        for output in self.outputs[b]
                    ]
        else:
            # 训练阶段
            # 将输入长度分成不同的间隔，这样数据在填充时只需要填充到相应的bucket长度即可，不需要都填充到最大长度
            # 比如buckets取[(5,10),(10,20),(20,30)...](每个bucket的第一个数字表示source填充的长度)
            # 第二个数字表示target填充的长度，eg:'我爱你'->'I love you'。应该会被分配到第一个bucket中
            # 然后'我爱你'会被pad成长度为5的序列，'I love you'会被pad成长度为10的序列，其实就是每个bucket表示一个模型的参数配置
            # 这样对每个bucket都构造一个模型，然后训练时取相应长度的序列进行，而这样模型将会共享参数
            # 其实这一部分可以参考现在的dynamic_rnn来及逆行理解，dynamic_rnn是对每个batch的数据讲起pad至本batch中长度最大的样本
            # 而bucket则时在数据预处理环节先对数据长度进行聚类操作，明白其原理之后我们来看一下这个函数的参数和内容实现

            # 跟language model类似，targets变量是decoder inputs 平移一个单位的结果，
            # encoder Inputs ：encoder的输入，一个tensor的列表，列表中每一项都是encoder时的一个词（batch)
            # decoder_inpits :decoder的输入，同上
            # targets ：目标值，与decoder_inputs只相差一个<eos>符号，int32型
            # buckets ：就是定义的bucket的值（编码器数据长度，解码器数据长度），是一个列表
            # seq2seq:定义好的seq2seq模型，可以使用后面介绍的embedding_attention_seq2seq,embedding_rnn_seq2seq,basic_rnn_seq2等
            # softmax_loss_fuction:计算误差的函数（labels,logits)默认为sqarse_softmax_cross_entroy_with_logits
            # per_example_loss:如果为真，则调用sequence_loss_by_example,返回一个列表，其每个元素就是一个样本的loss值，
            # 如果为假，则调用sequence_loss函数，对一个元素计算loss
            self.outputs, self.losses = legacy_seq2seq.model_with_buckets(
                self.encoder_inputs,  # 编码器输入
                self.decoder_inputs,  # 解码器输入
                targets,  # 实际值, 仅在loss损失函数构建的时候使用
                self.decoder_weights,  # 解码器权重
                buckets,  # 盒子
                lambda x, y: seq2seq_f(x, y, False),
                softmax_loss_function=softmax_loss_function # 损失函数
            )
            # 每个桶分别设置loss的可视化
            for b_idx in range(len(buckets)):
                bucket_loss_scalar = tf.summary.scalar('loss_{}'.format(b_idx), self.losses[b_idx])
                self.bucket_to_summary_list[b_idx].append(bucket_loss_scalar)

        if not forward_only:  # 只有训练阶段才需要计算梯度和参数更新
            print("开始构建优化器对象....")
            # 获取所有训练参数
            params = tf.trainable_variables()

            # 定义优化器
            opt = tf.train.AdamOptimizer(
                learning_rate=learning_rate
            )
            self.gradient_norms = []
            self.updates = []
            for output, loss in zip(self.outputs, self.losses):  # 获取得到每个桶的输出和损失函数的值
                # 基于给定的损失函数以及参数列表，计算参数列表对应的梯度值
                gradients = tf.gradients(loss, params)
                # 基于给定的最大梯度值(max_gradient_norm, 求参数的梯度值进行一个截断操作)
                clipped_gradients, norm = tf.clip_by_global_norm(
                    gradients,
                    max_gradient_norm
                )

                # 添加结果数据(全局norm以及参数更新操作)
                self.gradient_norms.append(norm)
                self.updates.append(opt.apply_gradients(
                    zip(clipped_gradients, params)
                ))

        # 定义模型持久化的对象
        self.saver = tf.train.Saver(
            tf.global_variables(),
            write_version=tf.train.SaverDef.V2
        )

    # 训练
    def step(
            self,
            session,
            encoder_inputs,
            decoder_inputs,
            decoder_weights,
            bucket_id,
            forward_only,
            merge=None
    ):
        encoder_size, decoder_size = self.buckets[bucket_id]
        if len(encoder_inputs) != encoder_size:
            raise ValueError(
                "Encoder length must be equal to the one in bucket,"
                " %d != %d." % (len(encoder_inputs), encoder_size)
            )
        if len(decoder_inputs) != decoder_size:
            raise ValueError(
                "Decoder length must be equal to the one in bucket,"
                " %d != %d." % (len(decoder_inputs), decoder_size)
            )
        if len(decoder_weights) != decoder_size:
            raise ValueError(
                "Weights length must be equal to the one in bucket,"
                " %d != %d." % (len(decoder_weights), decoder_size)
            )

        input_feed = {}
        for i in range(encoder_size):
            input_feed[self.encoder_inputs[i].name] = encoder_inputs[i]
        for i in range(decoder_size):
            input_feed[self.decoder_inputs[i].name] = decoder_inputs[i]
            input_feed[self.decoder_weights[i].name] = decoder_weights[i]

        # 理论上decoder inputs和decoder target都是n位
        # 但是实际上decoder inputs分配了n+1位空间
        # 不过inputs是第[0, n)，而target是[1, n+1)，刚好错开一位
        # 最后这一位是没东西的，所以要补齐最后一位，填充0
        last_target = self.decoder_inputs[decoder_size].name
        input_feed[last_target] = np.zeros([self.batch_size], dtype=np.int32)

        output_feed = []
        if not forward_only:  # 训练阶段
            # 根据桶id获取当前桶对应的训练对象
            output_feed.append(self.updates[bucket_id])
            # 根据桶id获取当前桶对应的梯度norm值
            output_feed.append(self.gradient_norms[bucket_id])
            # 根据桶id获取当前桶对应的loss值
            output_feed.append(self.losses[bucket_id])
            if merge is not None:
                output_feed.append(merge)
            # 根据桶idbucket_id获取对应的输出值，然后获取最后一个时刻的decoder输出
            output_feed.append(self.outputs[bucket_id][i])
        else:
            output_feed.append(self.losses[bucket_id])
            # 添加所有时刻对应的输出
            for i in range(decoder_size):
                output_feed.append(self.outputs[bucket_id][i])

        outputs = session.run(output_feed, input_feed)
        if not forward_only:
            if merge is not None:
                return outputs[1], outputs[2], outputs[3], outputs[4:]
            else:
                return outputs[1], outputs[2], None, outputs[3:]
        else:
            return None, outputs[0], outputs[1:]

    def get_batch_data(self, bucket_dbs, bucket_id):
        data = []
        data_in = []
        bucket_db = bucket_dbs[bucket_id]
        for _ in range(self.batch_size):
            ask, answer = bucket_db.random()
            data.append((ask, answer))
            data_in.append((answer, ask))
        return data, data_in

    def get_batch(self, bucket_id, data):
        # 获取bucket_id这个桶对应的ask和answer的字符长度大小限制值
        encoder_size, decoder_size = self.buckets[bucket_id]
        encoder_inputs, decoder_inputs = [], []
        for encoder_input, decoder_input in data:
            # ids化
            encoder_input = data_utils.sentence_indice(encoder_input)
            decoder_input = data_utils.sentence_indice(decoder_input)
            # Encoder Padding
            encoder_pad = [data_utils.PAD_ID] * (
                    encoder_size - len(encoder_input)
            )
            encoder_inputs.append(list(reversed(encoder_input + encoder_pad)))
            # Decoder Padding
            decoder_pad_size = decoder_size - len(decoder_input) - 2
            decoder_inputs.append(
                [data_utils.GO_ID] +
                decoder_input +
                [data_utils.EOS_ID] +
                [data_utils.PAD_ID] * decoder_pad_size
            )
        batch_encoder_inputs, batch_decoder_inputs, batch_weights = [], [], []
        # batch encoder
        for i in range(encoder_size):
            batch_encoder_inputs.append(np.array(
                [encoder_inputs[j][i] for j in range(self.batch_size)],
                dtype=np.int32
            ))
        # batch decoder
        for i in range(decoder_size):
            batch_decoder_inputs.append(np.array(
                [decoder_inputs[j][i] for j in range(self.batch_size)],
                dtype=np.int32
            ))
            batch_weight = np.ones(self.batch_size, dtype=np.float32)
            for j in range(self.batch_size):
                if i < decoder_size - 1:
                    target = decoder_inputs[j][i + 1]
                if i == decoder_size - 1 or target == data_utils.PAD_ID:
                    batch_weight[j] = 0.0
            batch_weights.append(batch_weight)
        return batch_encoder_inputs, batch_decoder_inputs, batch_weights

    def predict(self, buckets, sentence, bucket, sess):
        # 判断参数是位于哪个bucket中的
        bucket_id = min([
            b for b in range(len(buckets))
            if buckets[b][0] > len(sentence)
        ])
        # 输入句子处理
        data, _ = self.get_batch_data(
            {bucket_id: bucket},
            bucket_id
        )
        # 编码解码输入
        encoder_inputs, decoder_inputs, decoder_weights = self.get_batch(
            bucket_id,
            data
        )
        # 模型执行
        _, _, output_logits = self.step(
            sess,
            encoder_inputs,
            decoder_inputs,
            decoder_weights,
            bucket_id,
            True
        )
        outputs = [int(np.argmax(logit, axis=1)) for logit in output_logits]
        ret = data_utils.indice_sentence(outputs)
        return ret
