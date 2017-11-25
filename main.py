#!/usr/bin/env python
# encoding: utf-8

"""
@version: python3
@author: ‘aprilkuo‘
@license: Apache Licence 
@contact: aprilvkuo@gmail.com
@site: 
@software: PyCharm Community Edition
@file: main.py
@time: 2017/11/25 上午12:51
"""
from sklearn.linear_model import LogisticRegression
from sklearn import svm,ensemble
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.utils import shuffle
import PreProcess
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn import preprocessing
import numpy as np


def train_func(x_data, y_data, class_name=LogisticRegression):
    train_model = class_name(class_weight='balanced')
    train_model.fit(x_data, y_data)
    return train_model


def result_vectoring(v):
    v = v.reshape(-1, 1)
    return np.concatenate((v*(-1)+1, v), axis=1)


if __name__ == "__main__":
    ham = './DataSet/ham_new.txt'
    spam = './DataSet/spam_new.txt'
    models = [LogisticRegression, ensemble.RandomForestClassifier, svm.LinearSVC]

    pre_process = 1
    is_test = 1
    if pre_process:
        PreProcess.process_data([ham, spam])  # 处理原始数据，tf-idf训练后并保存
    if is_test:
        x, y, model = PreProcess.load_data()  # 加载tf-idf数据
        print len(model.get_feature_names())
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.9)
        lr = train_func(x_train, y_train)
        #print len(lr.coef_[0])
        feature_map = dict(zip(model.get_feature_names(),map(abs,lr.coef_[0])))
        f = sorted(feature_map.items(), key=lambda x:x[1])[-1:-100:-1]
        for key,value in f:
           print key,value
        #print sorted(lr.get_params().items(), key=lambda x:x[1])[-100:]
        y_pred = lr.predict(x_test)
        print 'classification_report\n', metrics.classification_report(y_test, y_pred)
        print lr.score(x_test,y_test)

    else:
        x, y, _ = PreProcess.load_data()  # 加载tf-idf数据
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
        print np.bincount(y)
        for model in models:
            print 'Using %s :'%(model.__name__)
            model = train_func(x_train, y_train)
            y_score = model.predict_proba(x_test)
            y_pred = model.predict(x_test)
            y_vec = result_vectoring(y_test)
            print 'roc_auc_score:', metrics.roc_auc_score(y_vec, y_score)
            print 'accuracy_score:', metrics.accuracy_score(y_test, y_pred)
            print 'classification_report:\n', metrics.classification_report(y_test, y_pred)
            print 5*'\n'
