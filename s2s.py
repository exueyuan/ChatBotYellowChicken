#!/usr/bin/env python3

import os
import sys
import math
import time

import numpy as np
import tensorflow as tf

import data_utils
import s2s_model

tf.app.flags.DEFINE_float(
    'learning_rate',
    0.0003,
    '学习率'
)
tf.app.flags.DEFINE_float(
    'max_gradient_norm',
    5.0,
    '梯度最大阈值'
)
tf.app.flags.DEFINE_float(
    'dropout',
    0.7,
    '每层输出DROPOUT的大小'
)
tf.app.flags.DEFINE_integer(
    'batch_size',
    16,
    '批量梯度下降的批量大小'
)
tf.app.flags.DEFINE_integer(
    'size',
    128,
    'LSTM每层神经元数量'
)
tf.app.flags.DEFINE_integer(
    'num_layers',
    2,
    'LSTM的层数'
)
tf.app.flags.DEFINE_integer(
    'num_epoch',
    50000,
    '训练几轮'
)
tf.app.flags.DEFINE_integer(
    'num_samples',
    100,
    '计算损失函数的时候，负采样的类别数目'
)
tf.app.flags.DEFINE_integer(
    'num_per_epoch',
    50,
    '每轮训练多少随机样本'
)
tf.app.flags.DEFINE_string(
    'buckets_dir',
    './bucket_dbs',
    'sqlite3数据库所在文件夹'
)
tf.app.flags.DEFINE_string(
    'model_dir',
    './model',
    '模型保存的目录'
)
tf.app.flags.DEFINE_string(
    'model_name',
    'model',
    '模型保存的名称'
)
tf.app.flags.DEFINE_boolean(
    'use_fp16',
    False,
    '是否使用16位浮点数（默认32位）'
)
tf.app.flags.DEFINE_integer(
    'bleu',
    -1,
    '是否测试bleu'
)
tf.app.flags.DEFINE_boolean(
    'train',
    False,
    '是否在测试'
)
tf.app.flags.DEFINE_string(
    'gpu_fraction',
    '3/3',
    'idx / # of gpu fraction e.g. 1/3, 2/3, 3/3'
)

FLAGS = tf.app.flags.FLAGS
buckets = data_utils.buckets


def calc_gpu_fraction(fraction_string):
    """
    基于参数计算分配GPU的话，分配的百分比是多少
    :param fraction_string:
    :return:
    """
    idx, num = fraction_string.split('/')
    idx, num = float(idx), float(num)

    fraction = 1 / (num - idx + 1)
    print(" [*] GPU : %.4f" % fraction)
    return fraction


# forward_only 向前
def create_model(session, forward_only):
    """建立模型"""
    dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
    model = s2s_model.S2SModel(
        data_utils.dim,
        data_utils.dim,
        buckets,
        FLAGS.size,
        FLAGS.dropout,
        FLAGS.num_layers,
        FLAGS.max_gradient_norm,
        FLAGS.batch_size,
        FLAGS.learning_rate,
        FLAGS.num_samples,
        forward_only,
        dtype
    )
    return model


def train():
    # 流程
    # 1.数据预处理
    # 2.seq2seq

    # ========================================================
    # 准备数据
    print("train mode.......")
    print('准备数据')
    if not os.path.exists(FLAGS.model_dir):
        os.makedirs(FLAGS.model_dir)

    # 数据预处理
    # buckets_dir 训练数据目录
    bucket_dbs = data_utils.read_bucket_dbs(FLAGS.buckets_dir)
    bucket_sizes = []
    for i in range(len(buckets)):
        # 语句的尺寸
        bucket_size = bucket_dbs[i].size  # 不同的桶的数据量
        bucket_sizes.append(bucket_size)
        print('bucket {} 中有数据 {} 条'.format(i, bucket_size))
    total_size = sum(bucket_sizes)  # 获取所有的样本数目
    print('共有数据 {} 条'.format(total_size))

    # 开始建模与训练
    gpu_options = tf.GPUOptions(
        allow_growth=True,  # 允许GPU分配是一种增量分配的方式
        per_process_gpu_memory_fraction=calc_gpu_fraction(FLAGS.gpu_fraction)
    )

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                          gpu_options=gpu_options)) as sess:
        # 　构建模型（每个桶对应一个训练对象\损失函数\summary相关信息；但是这四套代码是参数共享的）
        model = create_model(sess, False)

        # 初始化变量&模型恢复
        print("开始进行模型初始化以及模型恢复操作.....")
        sess.run(tf.global_variables_initializer())
        ckpt = tf.train.get_checkpoint_state(FLAGS.model_dir)
        if ckpt and ckpt.model_checkpoint_path:
            print("Load old model from : ", ckpt.model_checkpoint_path)
            model.saver.restore(sess, ckpt.model_checkpoint_path)
            model.saver.recover_last_checkpoints(ckpt.all_model_checkpoint_paths)
        else:
            print("Not exist old model")

        # 计算每个桶的样本的累计占比(1号桶的占比， 1+2号桶的占比， 1+2+3号桶的占比， 1+2+3+4号桶的占比)
        buckets_scale = [
            sum(bucket_sizes[:i + 1]) / total_size
            for i in range(len(bucket_sizes))
        ]

        # 开始训练
        metrics = '  '.join([
            '\r[{}]',
            '{:.1f}%',
            '{}/{}',
            'loss={:.3f}',
            '{}/{}'
        ])

        bars_max = 20
        writer = tf.summary.FileWriter('log', graph=sess.graph)
        merges = []
        # 针对每个桶(每个训练对象)获取对应的summary的的可视化输出对象
        for b_idx in model.bucket_to_summary_list:
            merges.append(tf.summary.merge(model.bucket_to_summary_list[b_idx]))
        print("开始模型训练.....")
        with tf.device('/gpu:0'):
            for epoch_index in range(1, FLAGS.num_epoch + 1):
                print('Epoch {}:'.format(epoch_index))
                time_start = time.time()
                epoch_trained = 0
                batch_loss = []
                while True:
                    # 随机选择一个要训练的bucket
                    random_number = np.random.random_sample()
                    bucket_id = min([
                        i for i in range(len(buckets_scale))
                        if buckets_scale[i] > random_number
                    ])
                    # 获取数据（从随机的桶中获取数据，获取batch_size条数据）
                    data, _ = model.get_batch_data(
                        bucket_dbs,
                        bucket_id
                    )
                    encoder_inputs, decoder_inputs, decoder_weights = model.get_batch(
                        bucket_id,
                        data
                    )

                    # run 迭代训练
                    _, step_loss, summary_merge, output = model.step(
                        sess,
                        encoder_inputs,
                        decoder_inputs,
                        decoder_weights,
                        bucket_id,
                        False,
                        merges[bucket_id]
                    )

                    epoch_trained += FLAGS.batch_size
                    batch_loss.append(step_loss)
                    time_now = time.time()
                    time_spend = time_now - time_start
                    time_estimate = time_spend / (epoch_trained / FLAGS.num_per_epoch)
                    percent = min(100, epoch_trained / FLAGS.num_per_epoch) * 100
                    bars = math.floor(percent / 100 * bars_max)
                    sys.stdout.write(metrics.format(
                        '=' * int(bars) + '-' * int(bars_max - bars),
                        percent,
                        epoch_trained, FLAGS.num_per_epoch,
                        np.mean(batch_loss),
                        data_utils.time(time_spend), data_utils.time(time_estimate)
                    ))
                    sys.stdout.flush()
                    if summary_merge is not None:
                        writer.add_summary(summary_merge, global_step=epoch_index)
                    if epoch_trained >= FLAGS.num_per_epoch:
                        model.saver.save(sess, os.path.join(FLAGS.model_dir, FLAGS.model_name), global_step=epoch_index)
                        break
                print('\n')

        # 最终再来一次模型持久化输出
        model.saver.save(sess, os.path.join(FLAGS.model_dir, FLAGS.model_name))


def test_bleu(count):
    """测试bleu, 这个方法我们不看"""
    print("bleu test mode")
    from nltk.translate.bleu_score import sentence_bleu
    from tqdm import tqdm
    # 准备数据
    print('准备数据')
    bucket_dbs = data_utils.read_bucket_dbs(FLAGS.buckets_dir)
    bucket_sizes = []
    for i in range(len(buckets)):
        bucket_size = bucket_dbs[i].size
        bucket_sizes.append(bucket_size)
        print('bucket {} 中有数据 {} 条'.format(i, bucket_size))
    total_size = sum(bucket_sizes)
    print('共有数据 {} 条'.format(total_size))
    # bleu设置0的话，默认对所有样本采样
    if count <= 0:
        count = total_size
    buckets_scale = [
        sum(bucket_sizes[:i + 1]) / total_size
        for i in range(len(bucket_sizes))
    ]
    with tf.Session() as sess:
        # 　构建模型
        model = create_model(sess, True)
        model.batch_size = 1
        # 初始化变量
        sess.run(tf.initialize_all_variables())
        model.saver.restore(sess, os.path.join(FLAGS.model_dir, FLAGS.model_name))

        total_score = 0.0
        for i in tqdm(range(count)):
            # 选择一个要训练的bucket
            random_number = np.random.random_sample()
            bucket_id = min([
                i for i in range(len(buckets_scale))
                if buckets_scale[i] > random_number
            ])
            data, _ = model.get_batch_data(
                bucket_dbs,
                bucket_id
            )
            encoder_inputs, decoder_inputs, decoder_weights = model.get_batch(
                bucket_id,
                data
            )
            _, _, output_logits = model.step(
                sess,
                encoder_inputs,
                decoder_inputs,
                decoder_weights,
                bucket_id,
                True
            )
            outputs = [int(np.argmax(logit, axis=1)) for logit in output_logits]
            ask, _ = data[0]
            all_answers = bucket_dbs[bucket_id].all_answers(ask)
            ret = data_utils.indice_sentence(outputs)
            if not ret:
                continue
            references = [list(x) for x in all_answers]
            score = sentence_bleu(
                references,
                list(ret),
                weights=(1.0,)
            )
            total_score += score
        print('BLUE: {:.2f} in {} samples'.format(total_score / count * 10, count))


def test():
    print("test mode")

    class TestBucket(object):
        def __init__(self, sentence):
            self.sentence = sentence

        def random(self):
            return sentence, ''

    with tf.Session() as sess:
        # 　构建模型
        model = create_model(sess, True)
        model.batch_size = 1
        # 初始化变量
        sess.run(tf.initialize_all_variables())
        ckpt = tf.train.get_checkpoint_state(FLAGS.model_dir)
        if ckpt and ckpt.model_checkpoint_path:
            model.saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            print("Have no model")
            return

        print("Input 'exit()' to exit test mode!")
        sys.stdout.write("me > ")
        sys.stdout.flush()
        sentence = sys.stdin.readline().strip()
        if "exit()" in sentence:
            sentence = False
        while sentence:
            # 获取最小的分桶id
            # 根据句子的长度，进行匹配最小的
            bucket_id = min([
                b for b in range(len(buckets))
                if buckets[b][0] > len(sentence)
            ])
            # 输入句子处理
            data, _ = model.get_batch_data(
                {bucket_id: TestBucket(sentence)},
                bucket_id
            )
            encoder_inputs, decoder_inputs, decoder_weights = model.get_batch(
                bucket_id,
                data
            )
            _, _, output_logits = model.step(
                sess,
                encoder_inputs,
                decoder_inputs,
                decoder_weights,
                bucket_id,
                True
            )
            outputs = [int(np.argmax(logit, axis=1)) for logit in output_logits]
            ret = data_utils.indice_sentence(outputs)
            print("AI >", ret)
            print("me > ", end="")
            sys.stdout.flush()
            sentence = sys.stdin.readline()
            if "exit()" in sentence:
                break


def main(_):
    if FLAGS.bleu > -1:
        test_bleu(FLAGS.bleu)
    elif FLAGS.train:
        train()
    else:
        test()


if __name__ == '__main__':
    # 设置随机数种子
    np.random.seed(0)
    tf.set_random_seed(0)
    # 运行，默认触发当前py文件中的main函数
    tf.app.run()
