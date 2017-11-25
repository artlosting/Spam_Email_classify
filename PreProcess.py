#!/usr/bin/env python
# encoding: utf-8


"""
@author: aprilvkuo
@license: Apache Licence 
@contact: aprilvkuo@live.com
@site: 
@software: PyCharm Community Edition
@file: PreProcess.py
@time: 2017/11/24 22:02
"""
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer,HashingVectorizer,CountVectorizer
import numpy as np
import pickle


def segment(line):
    '''
    分词
    :param line: 
    :return: 
    '''
    return list(jieba.cut(line))


def process_data(file_list, save_name='tf-idf.model'):
    '''
    加载数据，tf-idf处理后， 然后保存数据
    :param file_list: 
    :param save_name: 
    :return: 
    '''
    vectorizer = CountVectorizer(input='content', tokenizer=segment, analyzer='word')
    content = []
    file_cnt = []
    for file_name in file_list:
        before_size = len(content)
        content.extend(open(file_name,'r').readlines())
        file_cnt.append(len(content)-before_size)
    x = vectorizer.fit_transform(content)

    y = np.concatenate((np.repeat([[0,1]], file_cnt[0],axis=0),
                        np.repeat([[1, 0]], file_cnt[1], axis=0)), axis=0)
    data = pickle.dumps((x, y))
    with open(save_name, 'w') as f:
        f.write(data)
    return


def load_data(model_name='tf-idf.model'):
    '''
    加载tf-idf数据
    :param model_name: 
    :return: 
    '''
    x, y = pickle.loads(open(model_name,'r').read())
    return x, y



