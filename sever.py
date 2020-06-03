from gevent import monkey

monkey.patch_all()
from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import numpy as np
from s2s import create_model, FLAGS
import data_utils

buckets = data_utils.buckets
app = Flask(__name__)


class TestBucket(object):
    def __init__(self, sentence):
        self.sentence = sentence

    def random(self):
        return self.sentence, ''


gpu_memory_fraction = 0.5
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction, allow_growth=True)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
# 　构建模型
model = create_model(sess, True)
model.batch_size = 1
# 初始化变量
sess.run(tf.global_variables_initializer())
with tf.Graph().as_default():
    ckpt = tf.train.get_checkpoint_state(FLAGS.model_dir)
    if ckpt and ckpt.model_checkpoint_path:
        model.saver.restore(sess, ckpt.model_checkpoint_path)
    else:
        raise FileNotFoundError('Having no model')  # Get input and output tensors


    # 显示首页
    @app.route('/')
    def index():
        return render_template('index.html')


    # 调用回复
    @app.route('/chart')
    def chart():
        # 獲取前端传递过来的文本
        sentence = request.args.get('sentence')

        # # 1. 意图识别
        # flag = model1.predict(sentence)
        # if flag == '意图1':
        #     # 调用
        #     ret = chat_model1.predict(buckets, sentence, TestBucket(sentence), sess)
        # elif flag == '意图2':
        #     # 调用
        #     ret = chat_model2.predict(buckets, sentence, TestBucket(sentence), sess)
        # elif flag == '意图3':
        #     # 调用
        #     ret = chat_model3.predict(buckets, sentence, TestBucket(sentence), sess)
        # else:
        #     # 闲聊模式
        #     ret = model.predict(buckets, sentence, TestBucket(sentence), sess)

        # 调用
        ret = model.predict(buckets, sentence, TestBucket(sentence), sess)
        return jsonify({'state': 0, 'result': ret})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8088, debug=True)
