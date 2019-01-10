#!/usr/bin/env python
# encoding: utf-8
'''
@author: hehaiyang
@contact: 1272114024@qq.com
@file: testStoreTree.py
@time: 2019/1/6 16:20
'''

import pickle
'''
使用python模块pickle序列化对象，序列化对象可以在磁盘上保存对象，并且在需要的时候读取出来

'''
def storeTree(inputTree,filename):
    with open(filename, 'wb') as fw:
        pickle.dump(inputTree, fw)


if __name__=="__main__":
    myTree = {'有自己的房子': {0: {'有工作': {0: 'no', 1: 'yes'}}, 1: 'yes'}}
    storeTree(myTree,'storeTree.txt')
    print('success')

