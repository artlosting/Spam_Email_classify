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


def segment(line):
    return list(jieba.cut(line))

vectorizer = CountVectorizer(input='content',tokenizer=segment,analyzer='word')





def read_data(filelist):
    global vectorizer
    for file in filelist:
    X = vectorizer.fit_transform()



    # print(filename)
    # for line in open(filename, 'r'):
    #     line = line.decode('utf-8', errors='ignore')
    #     line = line.encode('gbk', errors='ignore')
    #     print list(jieba.cut(line))



if __name__ == "__main__":
    ham = './DataSet/ham_5000.utf8'
    spam = './DataSet/spam_5000.utf8'
    read_data(ham)