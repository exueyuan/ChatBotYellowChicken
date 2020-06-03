# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'D:\code\python\AI_Project\tf_聊天机器人\chart.ui'
#
# Created by: PyQt5 UI code generator 5.11.3
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import *
from PyQt5.QtGui import *
import tensorflow as tf
import numpy as np
from s2s import create_model, FLAGS
import data_utils
import sys
import time


class TestBucket(object):
    def __init__(self, sentence):
        self.sentence = sentence

    def random(self):
        return self.sentence, ''


class Ui_Form(object):
    def setupUi(self, Form):
        self.sess = Form.sess
        self.model = Form.model
        self.buckets = Form.buckets
        self.Form = Form
        Form.setObjectName("Form")
        Form.resize(640, 635)
        self.label = QtWidgets.QLabel(Form)
        self.label.setGeometry(QtCore.QRect(10, 550, 75, 15))
        self.label.setObjectName("QLabel")
        self.label.setText('请输入...')
        self.textEdit = QtWidgets.QLineEdit(Form)
        self.textEdit.setGeometry(QtCore.QRect(10, 570, 611, 25))
        self.textEdit.setObjectName("lineEdit")
        self.textEdit.returnPressed.connect(self.buttonClicked)
        self.pushButton = QtWidgets.QPushButton(Form)
        self.pushButton.setGeometry(QtCore.QRect(450, 600, 75, 25))
        self.pushButton.setObjectName("pushButton")
        self.pushButton_2 = QtWidgets.QPushButton(Form)
        self.pushButton_2.setGeometry(QtCore.QRect(540, 600, 75, 25))
        self.pushButton_2.setObjectName("pushButton_2")
        self.textbrowser = QtWidgets.QTextBrowser(Form)
        self.textbrowser.setGeometry(QtCore.QRect(10, 10, 610, 530))
        font = QtGui.QFont()
        font.setPointSize(15)
        self.textbrowser.setFont(font)
        self.textbrowser.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.textbrowser.setFrameShadow(QtWidgets.QFrame.Raised)
        self.textbrowser.setObjectName("textbrowser")
        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "Form"))
        self.pushButton.setText(_translate("Form", "关闭"))
        self.pushButton_2.setText(_translate("Form", "发送"))
        self.pushButton_2.clicked.connect(self.buttonClicked)
        self.pushButton.clicked.connect(self.closeFrom)

    def keyPressEvent(self, event):
        if event.key() == QtCore.Qt.Key_Enter:
            self.buttonClicked()

    def closeFrom(self):
        reply = QtWidgets.QMessageBox.question(self.Form, '智能客服', '是否确认退出',
                                               QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
                                               QtWidgets.QMessageBox.No)
        if reply == QtWidgets.QMessageBox.Yes:
            self.Form.close()

    def buttonClicked(self):
        text = self.textEdit.text()
        self.textEdit.clear()
        if self.textbrowser.toPlainText().strip():
            last_text = self.textbrowser.toHtml()
        else:
            last_text = ''
        self.textbrowser.setText(last_text + '<p>Me>>{}</p>'.format(str(text)))
        cursor = self.textbrowser.textCursor()
        self.textbrowser.moveCursor(cursor.End)
        # 調用模型輸出
        self.OptimThread = ModelThread(text, self.sess, self.model, self.buckets)

        self.OptimThread.InfoSignal.connect(self.OptimEnd)

        self.OptimThread.start()

    def OptimEnd(self, ret):
        last_text = self.textbrowser.toHtml()
        self.textbrowser.setText(last_text + '<p style="text-align:right">{}&lt;&lt;AI</p>'.format(ret))
        cursor = self.textbrowser.textCursor()
        self.textbrowser.moveCursor(cursor.End)


class ModelThread(QThread):
    # 声明一个信号,接受返回值 generator_image,bs,loss
    InfoSignal = QtCore.pyqtSignal(str)

    def __init__(self, sentence, sess, model, buckets):
        super(ModelThread, self).__init__(None)
        self.sentence = sentence
        self.sess_gen = sess
        self.model = model
        self.buckets = buckets

    def run(self):
        bucket_id = min([
            b for b in range(len(self.buckets))
            if self.buckets[b][0] > len(self.sentence)
        ])
        # 输入句子处理
        data, _ = self.model.get_batch_data(
            {bucket_id: TestBucket(self.sentence)},
            bucket_id
        )
        # 编码解码输入
        encoder_inputs, decoder_inputs, decoder_weights = self.model.get_batch(
            bucket_id,
            data
        )
        # 模型执行
        _, _, output_logits = self.model.step(
            self.sess_gen,
            encoder_inputs,
            decoder_inputs,
            decoder_weights,
            bucket_id,
            True
        )
        outputs = [int(np.argmax(logit, axis=1)) for logit in output_logits]
        ret = data_utils.indice_sentence(outputs)
        self.InfoSignal.emit(ret)


class LoginDlg(QtWidgets.QWidget, Ui_Form):
    def __init__(self, parent=None):
        super(LoginDlg, self).__init__(parent)
        self.loadModel()
        self.setupUi(self)
        self.setWindowTitle("智能客服")

    def loadModel(self):
        self.buckets = data_utils.buckets
        gpu_memory_fraction = 0.9
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction, allow_growth=True)
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        # 　构建模型
        self.model = create_model(self.sess, True)
        self.model.batch_size = 1
        # 初始化变量
        self.sess.run(tf.global_variables_initializer())
        ckpt = tf.train.get_checkpoint_state(FLAGS.model_dir)
        if ckpt and ckpt.model_checkpoint_path:
            self.model.saver.restore(self.sess, ckpt.model_checkpoint_path)
        else:
            raise FileNotFoundError('Having no model')  # Get input and output tensors


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    dlg = LoginDlg()
    dlg.show()
    sys.exit(app.exec_())
