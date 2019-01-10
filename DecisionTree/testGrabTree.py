#!/usr/bin/env python
# encoding: utf-8
'''
@author: hehaiyang
@contact: 1272114024@qq.com
@file: testGrabTree.py
@time: 2019/1/6 16:25
'''

'''
读取决策树
'''
import pickle

def grabTree(filename):
    fr = open(filename,'rb')
    return pickle.load(fr)

if __name__=='__main__':
    myTree = grabTree('storeTree.txt')
    print(myTree)