#!/usr/bin/python
# encoding: utf-8

"""
@author: Ian
@contact:yongguiluo@hotmail.com
@file: main.py
@time: 2018/12/10 15:20
"""
from stock.views import tt
import logging
import logging.config
import sys
import yaml
import os
from datetime import datetime
import pandas as pd
import numpy as np
from pyecharts import Line


def setup_logging(default_path='config.yaml', default_level=logging.INFO):
    path = default_path
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            config = yaml.load(f)
            logging.config.dictConfig(config)
    else:
        logging.basicConfig(level=default_level)


if __name__ == '__main__':
    yaml_path = '../logging_config.yaml'
    setup_logging(yaml_path)
    logger = logging.getLogger('main.core')
    t = tt()
    logger.info('获取20010101至今的000001.SH的数据')
    endDate = datetime.now().strftime('%Y%m%d')
    # df = t.index_daily('000001.SH', start_date='20010101', end_date=endDate)
    # logging.info('计算成交额涨幅')
    # a = df['amount']
    # l1 = list()
    # l1.append(1)
    # for i in range(1, len(df['amount'])):
    #     l1.append((a.iloc[i] - a.iloc[i - 1]) / a.iloc[i - 1])
    # df['amount_ratio'] = l1
    # print(df.iloc[1:, :].head())
    # df.iloc[1:, :].to_csv('000001.SH.csv')
    df = pd.read_csv('000001.SH.csv', index_col=0)
    fieldnamelist = ['close','pct_chg', 'amount', 'amount_ratio']
    df1 = t.normlize_field(df, fieldnamelist)
    ll = t.build_flag(df1, 180, 15, ['{}_1'.format(i) for i in fieldnamelist])
    y = [(i[-2], i[-1]) for i in ll]
    df2 = df1.iloc[180:(df1.shape[0] - 15)]
    df2['y1'] = [i[0] for i in y]
    df2['y2'] = [i[1] for i in y]
    logging.info('数据预处理')
    x, y1, y2 = t.preprocess(ll[:-250])
    x_train, x_test, y_train, y_test = t.splitData(x, y1)
    logging.info('构建模型')
    t.buildModel(len(fieldnamelist))
    count = 0
    # gg = [{'coord': [str(line[2]), line[3]]} for line in df2[df2['y1'] == 1].itertuples()]
    # line = Line("000001.SH可视化观察(周期)", title_pos='center', subtitle="预测值")
    # line.add(
    #     '上涨指数收盘价',
    #     [str(i) for i in list(df2.iloc[:, 1])],
    #     list(df2.iloc[:, 2]),
    #     mark_point=gg
    # )
    # line.render("预测值.html")
    def train_test_predict(t, epochs):
        """
        :param count 已经训练的周期
        :param epochs: 再训练的周期
        :return:
        """
        global count, df2
        logging.info('开始训练模型(周期: {}-{})'.format(count, count+epochs))
        count += epochs

        t.train(x_train, y_train, epochs=epochs)
        modelname = 'model_{}_{}.h5'.format(endDate, count)
        logging.info('保存模型至{}'.format(modelname))
        t._model.save(modelname)
        logging.info('评估模型(周期:{})'.format(count))
        loss, accuracy = t._model.evaluate(x_test, y_test)

        logging.info('test loss：{}'.format(loss))
        logging.info('test accuracy：{}'.format(accuracy))

        logging.info('预测(周期:{})'.format(count))
        x_p, y1_p, y2_p = t.preprocess(ll[-250:], balance=False)
        y_predict = t.predict(x_p)
        logging.info(t.evaluateScore(y1_p, y_predict))

        dft = df2.iloc[-250:]
        dft['y1_predict'] = np.argmax(y_predict, axis=1)
        gg = [{'coord': [str(line[2]), line[3]]} for line in dft[dft['y1'] == 1].itertuples()]
        line = Line("000001.SH可视化观察(周期:{})".format(count), title_pos='center', subtitle="真实值")
        line.add(
            '上涨指数收盘价',
            [str(i) for i in df2.iloc[-250:, 1]], df2.iloc[-250:, 2],
            legend_top='bottom',
            is_more_utils=True,
            mark_point=gg,
            mark_point_symbolsize=40
        )
        line.render("真实值(周期_{}).html".format(count))
        gg1 = [{'coord': [str(line[2]), line[3]]} for line in dft[dft['y1_predict'] == 1].itertuples()]
        line1 = Line("000001.SH可视化观察(周期_{})".format(count), title_pos='center', subtitle="预测值")
        line1.add(
            '上涨指数收盘价',
            [str(i) for i in df2.iloc[-250:, 1]], df2.iloc[-250:, 2],
            legend_top='bottom',
            is_more_utils=True,
            mark_point=gg1,
            mark_point_symbolsize=80
        )
        line1.render("预测值(周期_{}).html".format(count))

    # train_test_predict(t, 200)
    # train_test_predict(t, 50)
    # train_test_predict(t, 50)
    # train_test_predict(t, 50)
    # train_test_predict(t, 50)
    train_test_predict(t, 1)
    # train_test_predict(t, 10)

    logging.info('----------------------finish----------------------')