#!/usr/bin/python
# encoding: utf-8

"""
@author: Ian
@contact:yongguiluo@hotmail.com
@file: pred.py
@time: 2018/12/12 10:11
"""
from stock.views import tt
import pandas as pd
import numpy as np
from datetime import datetime

if __name__ == '__main__':
    t = tt()
    t.loadModel('model_20181211_350.h5')
    tsCode = '000001.SH'
    endDate = datetime.now().strftime('%Y%m%d')
    df = t.index_daily(tsCode,'20010101', endDate)
    a = df['amount']
    l1 = list()
    l1.append(1)
    for i in range(1, len(df['amount'])):
        l1.append((a.iloc[i] - a.iloc[i - 1]) / a.iloc[i - 1])
    df['amount_ratio'] = l1
    print(df.head())
    print(df.tail())
    fieldnamelist = ['close', 'pct_chg', 'amount', 'amount_ratio']
    df1 = t.normlize_field(df.iloc[1:], fieldnamelist)

    x = t.build_x(df1, 180, 45, ['{}_1'.format(i) for i in fieldnamelist])

    y_p = t.predict(x)
    print(np.argmax(y_p, axis=1))
    dff = df1.iloc[-45:,:2]
    dff['y_p'] = np.argmax(y_p, axis=1)[:-1]
    print(dff)