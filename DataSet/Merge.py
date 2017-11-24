#!/usr/bin/env python
# encoding: utf-8

"""
@version: python3
@author: ‘aprilkuo‘
@license: Apache Licence 
@contact: aprilvkuo@gmail.com
@site: 
@software: PyCharm Community Edition
@file: Merge.py
@time: 2017/11/25 上午1:26
"""


import os


for dir_name in os.listdir('./'):
    if os.path.isdir(dir_name):
        f_out = open(dir_name+'.txt','w')
        for file_name in os.listdir(dir_name):
            f_out.write(open(os.path.join(dir_name,file_name),'r').read()+'\n')
        f_out.close()


