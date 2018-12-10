#!/usr/bin/python
# encoding: utf-8

"""
@author: Ian
@contact:yongguiluo@hotmail.com
@file: views.py
@time: 2018/12/10 15:19
"""
import pandas as pd
import numpy as np
import tushare as ts
from datetime import datetime, timedelta
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense
from keras.layers.recurrent import LSTM
from keras.optimizers import Adam
from keras.models import load_model
import time


class tt:
    def __init__(self):
        ts.set_token('5fd1639100f8a22b7f86e882e03192009faa72bae1ae93803e1172d5')
        self._pro = ts.pro_api()

    def index_daily(self, ts_code, start_date, end_date):
        '''
        获取行情数据
        由于ts的接口一次只能获取1800个交易日（一年大概有250个交易日。约7年）的数据
        '''
        startdate = datetime.strptime(start_date, '%Y%m%d')
        enddate = datetime.strptime(end_date, '%Y%m%d')
        df = pd.DataFrame()
        while enddate.year - startdate.year > 6:
            print(startdate.strftime('%Y%m%d'),
                  (startdate.replace(year=(startdate.year + 6)) - timedelta(days=1)).strftime('%Y%m%d'))
            t = self._pro.index_daily(ts_code=ts_code, start_date=startdate.strftime('%Y%m%d'), end_date=(
                        startdate.replace(year=(startdate.year + 6)) - timedelta(days=1)).strftime('%Y%m%d'))
            if not df.empty:
                df = pd.concat([df, t], axis=0)
            else:
                df = t
            startdate = startdate.replace(year=(startdate.year + 6))
        else:
            print(startdate.strftime('%Y%m%d'), end_date)
            t = self._pro.index_daily(ts_code=ts_code, start_date=startdate.strftime('%Y%m%d'), end_date=end_date)
            if not df.empty:
                df = pd.concat([df, t], axis=0)
            else:
                df = t
        return df.sort_values('trade_date')

    def normlize_field(self, df, fieldnamelist):
        '''
        #设df中某一字段 第一日净值为1
        '''
        df1 = pd.DataFrame()
        df1['ts_code'] = df['ts_code']
        df1['trade_date'] = df['trade_date']
        for f in fieldnamelist:
            df1[f + '_1'] = df.apply(lambda x: x[f] / df.iloc[0][f], axis=1)
        return df1

    def build_flag(self, df, series_len, pro_len, fieldnamelist):
        '''
        构建训练集

        series_len: 参考的之前的序列范围。如以之前的series_len个序列预测下一个序列，则series_len=series_len
        pro_len: 预测日以后的天数（含预测日）
        '''
        r = list()
        l1 = list(df['close_1'])
        for i in range(series_len, (df.shape[0] - pro_len)):
            final_list = list()
            laa = l1[i:i + pro_len]
            # 最低点和买点的关系：最低点一定是买点，买点不一定是最低点
            # 买点特征
            # 买点日之后14日最高收盘价涨幅超过0.05，最低价不得低于买点日收盘价
            # 最低点的特征：在买点特征的基础上
            # 最低日收盘价低于前一日收盘价

            f1 = 0  # 买点标志 1表示买点
            f2 = 0  # 最低点标志 1表示最低点
            if (max(laa) - l1[i]) / l1[i] > 0.05 and min(laa[1:]) > l1[i]:  # 未来pro_len日最高收盘价涨幅超过0.05
                f1 = 1
                if l1[i] < l1[i - 1]:
                    f2 = 1
            final_list.append(df[fieldnamelist].values[(i - series_len): i])
            final_list.append(f1)
            final_list.append(f2)
            r.append(final_list)
        return r

    def getNum(self, ll):
        '''
        获取买点、最低点的个数
        '''
        y = [(i[-2], i[-1]) for i in ll]
        return Counter(y)

    def preprocess(self, ll, balance=True):
        '''
        数据预处理，获得可用于训练的set
        '''
        ll1 = [i for i in ll if i[1] == 1]  # 买点数据
        # 均衡数据
        ll2 = ll
        if balance:
            ll2 += ll1 * (round(len(ll) / len(ll1)) - 1)
        ll3 = [(i[1], i[2]) for i in ll2]
        x = np.array([i[0] for i in ll2])
        y1 = np.array([[i[1]] for i in ll2])  # 买点
        y2 = np.array([[i[2]] for i in ll2])  # 最低点
        print(Counter(ll3))
        return x, np_utils.to_categorical(y1, num_classes=2), np_utils.to_categorical(y2, num_classes=2)

    def splitData(self, x, y):
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
        return x_train, x_test, y_train, y_test

    def buildModel(self, n):
        '''
        # 创建模型
        n 考虑的特征的个数
        '''
        model = Sequential()

        # 循环神经网络
        model.add(LSTM(
            units=256,  # 输出
            input_shape=(180, n),  # 输入
        ))
        model.add(Dense(200, activation='tanh'))
        # 输出层
        model.add(Dense(2, activation='softmax'))

        # 定义优化器
        adam = Adam(lr=1e-4)

        # 定义优化器，loss function，训练过程中计算准确率
        model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
        print(model.summary())
        self._model = model

    def train(self, x_train, y_train, batch_size=128, epochs=300):
        # 训练模型
        start = time.time()
        self._model.fit(x_train, y_train, batch_size, epochs)
        print('@ Total Time Spent: %.2f seconds' % (time.time() - start))

    def evaluate(self, x_test, y_test):
        # 评估模型
        loss, accuracy = t._model.evaluate(x_test, y_test)
        print('test loss', loss)
        print('test accuracy', accuracy)
        return loss, accuracy

    def build_x(self, df, series_len, start):
        '''
        构建预测序列

        series_len: 参考的之前的序列范围。如以之前的series_len个序列预测下一个序列，则series_len=series_len
        '''
        ll = list()
        l1 = list(df['close_1'])
        l4 = list(df['vol_1'])
        l5 = list(df['amount_1'])
        for i in range(df.shape[0] - start, df.shape[0]):
            final_list = list()
            l2 = l1[i - series_len: i]
            ll.append(list(zip(l1[i - series_len: i], l4[i - series_len: i], l5[i - series_len: i])))
        return ll

    def predict(self, x_list):
        '''
        x_list: 预测序列的list
        '''
        r = self._model.predict(x_list)
        return r

    def loadModel(self, filepath):
        self._model = load_model(filepath)

    def evaluateScore(self, y_test, y_predict):
        score = accuracy_score(np.argmax(y_test, axis=1), np.argmax(y_predict, axis=1))
        c = classification_report(np.argmax(y_test, axis=1), np.argmax(y_predict, axis=1))
        print(score)
        print(c)