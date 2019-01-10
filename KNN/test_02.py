#!/usr/bin/env python
# encoding: utf-8
'''
@author: hehaiyang
@contact: 1272114024@qq.com
@file: test_02.py
@time: 2019/1/4 14:39
'''
import datashape
import numpy as np
import operator
import time
'''
创建数据集和标签


'''
def createDataSet():
	#四组二维特征
	group = np.array([[1,101],[5,89],[108,5],[115,8]])
	#四组特征的标签
	labels = ['爱情片','爱情片','动作片','动作片']
	return group, labels


def classify0(inX,dataSet,labels,k):
    '''
    使用欧式距离公式，计算两个向量点之间的距离
    '''
    #通过numpy下的函数，shape[0]计算样本的总数,标签向量元素的数目，与矩阵dataSet的行数相同
    dataSetSize = dataSet.shape[0]
    #在列方向上重复inX共1次横向，行向量方向
    #二维特征向量相减之后，平方
    diffMat = np.tile(inX, (dataSetSize, 1)) - dataSet
    sqDiffMat = diffMat ** 2
    #sum()所有元素相加，sum（1）行相加
    sqDistances = sqDiffMat.sum(axis=1)
    #开方，计算出欧式距离
    distances = sqDistances ** 0.5
    '''
    选择距离最小的k个点
    '''
    #返回distance中元素从小到大排序之后的索引值
    sortedDistIndices = distances.argsort()
    #类别计数词典
    classCount = {}
    for i in range(k):
        #取出前k个类别标签
        voteIlabel = labels[sortedDistIndices[i]]
        #get方法返回指定的键值，如果键值不在的话，返回默认值0
        #计算类别的次数
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1

    '''
    从大到小的次序进行排序，最后返回发生频率最高的元素标签
    '''
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


if __name__ =='__main__':
    #创建数据集
    group,labels = createDataSet()
    #测试集
    test = [101,20]
    #利用KNN进行分类
    test_class = classify0(test,group,labels,3)
    #输出分类的结果
    print("预测类别:"+test_class)

