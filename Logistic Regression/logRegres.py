#!/usr/bin/env python
# encoding: utf-8
'''
@author: hehaiyang
@contact: 1272114024@qq.com
@file: logRegres.py
@time: 2019/1/11 14:46
'''
import numpy as np
import matplotlib.pyplot as plt

"""
函数说明：读取文本文件，加载数据集，划分出数据部分和标签部分

"""

def loadDataSet():
    dataMat = []
    labelMat = []
    fr = open('testSet.txt')
    for line in fr.readlines():
        lineArr = line.strip().split()
        dataMat.append([1.0,float(lineArr[0]),float(lineArr[1])])           #为了方便计算，将X0设置为1.0
        labelMat.append(int(lineArr[2]))
    fr.close()
    return dataMat,labelMat

"""
函数说明：sigmoid函数
    将任意的函数值映射到（0，1）区间上，转化成概率值

"""
def sigmoid(inX):
    return 1.0/(1 + np.exp(-inX))

"""
函数说明：求解梯度上升法的最优的回归系数

"""
def gradAscent(dataMat,classLabels):
    dataMatrix = np.mat(dataMat)                                    #将数据转成矩阵形式便于计算
    labelMat = np.mat(classLabels).transpose()                      #类别标签，进行转置将行向量转化成列向量
    m,n = np.shape(dataMatrix)                                      #得到矩阵的行和列的值
    alpha = 0.001                                                   #学习率
    maxCycles = 500                                                 #最大迭代次数
    weights = np.ones((n,1))
    for k in range(maxCycles):
        #矩阵相乘
        h = sigmoid(dataMatrix * weights)                           #梯度上升矢量化公式
        error = (labelMat - h)                                      #计算真实类别与预测类别的差值
        weights = weights + alpha * dataMatrix.transpose() * error  #按照上述计算出来的差值的方向调整回归系数
    return weights
"""
函数说明：画出决策边界
    @:parameter     weights 权重参数数组

"""
def plotBestFit(weights):
    dataMat , labelMat = loadDataSet()
    dataArr = np.array(dataMat)                                     #转化为ndarray数组
    n = np.shape(dataArr)[0]
    xcord1 = []; ycord1 = []
    xcord2 = []; ycord2 = []
    for i in range(n):
        if int(labelMat[i]) == 1:
            xcord1.append(dataArr[i,1]);ycord1.append(dataArr[i,2])
        else:
            xcord2.append(dataArr[i,1]);ycord2.append(dataArr[i,2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1,ycord1,s=30,c='red',marker='s',alpha=0.5)
    ax.scatter(xcord2,ycord2,s=30,c='green',alpha=0.5)
    x = np.arange(-3.0,3.0,0.5)
    y = (-weights[0]-weights[1]*x)/weights[2]
    ax.plot(x,y)
    plt.title('BestFit')
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()

if __name__=='__main__':
    dataMat , labelMat = loadDataSet()
    weights = gradAscent(dataMat,labelMat)
    #print(weights)
    plotBestFit(weights.getA())                 #getA()函数作用与mat()函数作用相反