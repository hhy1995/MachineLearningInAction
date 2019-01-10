#!/usr/bin/env python
# encoding: utf-8
'''
@author: hehaiyang
@contact: 1272114024@qq.com
@file: KNN_02.py
@time: 2019/1/4 19:39
'''

import  numpy as np
import operator
from os import listdir

'''
k-近邻算法
函数说明
    @:parameter
        inX 用于分类的输入向量
        dataSet 输入的样本训练集
        labels 标签向量
        k 用于选择最近邻居的数目
    @:return sortedClassCount[0][0]
        所属类别信息
'''

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
'''
   将二进制图像矩阵转化成一维向量
   @:parameter  filename 文件名
    
   @:return returnVect 返回转化之后的矩阵

'''
def imgToVector(filename):
    #定义向量，并且初始化
    returnVect = np.zeros((1,1024))
    #打开文件
    fr = open(filename)
    #进行转化
    for i in range(32):
        #每次只读取一行，然后写回到数组returnVect当中
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0,32*i+j] = int(lineStr[j])
    return returnVect
'''
    
'''
def handwritingClassTest():
    #手写数字的标签
    hwLabels = []
    #获取目录的内容
    trainingFileList = listdir('trainingDigits')
    #计算训练文件的数量
    m = len(trainingFileList)
    #m行n列的训练矩阵，每行存储一个图像
    trainingMat = np.zeros((m,1024))
    for i in range(m):
        #从文本名当中解析出分类的数字
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)
        trainingMat[i,:] = imgToVector('trainingDigits/%s' % fileNameStr)
    testFileList = listdir('testDigits')
    #记录分类错误的数量
    errorCount = 0.0
    mTest = len(testFileList)
    for j in range(mTest):
        fileNameStr = testFileList[j]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = imgToVector('testDigits/%s' % fileNameStr)
        classifierResult = classify0(vectorUnderTest,trainingMat,hwLabels,5)
        print("分类返回结果：%d\t真实值：%d" % (classifierResult,classNumStr))
        if(classifierResult != classNumStr):
            errorCount += 1.0
    print("分类失败总数：%d" % errorCount)
    print("错误率：%f%%" % (errorCount/float(mTest)*100))

if __name__=="__main__":
    handwritingClassTest()

'''
if __name__=="__main__":
    filename = 'testDigits/0_13.txt'
    testVect = imgToVector(filename)
    print(testVect[0,0:31])
    print(testVect[0,32:63])
    
'''




