#!/usr/bin/env python
# encoding: utf-8
'''
@author: hehaiyang
@contact: 1272114024@qq.com
@file: logRegres.py
@time: 2019/1/11 14:46
'''
import matplotlib.font_manager as mfm
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
    weights_array = np.array([])
    for k in range(maxCycles):
        #矩阵相乘
        h = sigmoid(dataMatrix * weights)                           #梯度上升矢量化公式
        error = (labelMat - h)                                      #计算真实类别与预测类别的差值
        weights = weights + alpha * dataMatrix.transpose() * error  #按照上述计算出来的差值的方向调整回归系数
        weights_array = np.append(weights_array,weights)
    weights_array = weights_array.reshape(maxCycles, n)
    # 将矩阵转换为数组，返回权重数组
    # mat.getA()将自身矩阵变量转化为ndarray类型变量
    return weights.getA(), weights_array
"""
函数说明：随机梯度上升算法：

"""
def stocGradAscent0(dataMatrix,classLabels):
    dataArr = np.array(dataMatrix)
    m , n = np.shape(dataMatrix)
    alpha = 0.01
    weights = np.ones(n)                                            #权重数组初始化为1
    for i in range(m):
        h = sigmoid(sum(dataMatrix[i]*weights))
        error = classLabels[i] - h
        weights = weights + alpha * error * dataArr[i]
    return weights

"""
函数说明：改进的随机梯度下降算法
    上述的stocGradAscent0算法存在的问题：收敛慢，需要至少200次迭代过程才能收敛；大的波动停止之后还存在一些小的周期性的波动，
    原因是样本中存在一些不是先行可分的样本点，每次迭代的过程中存在着回归系数的剧烈改变。
"""
def stocGradAscent1(dataMatrix,classLabels,numIter = 150):              #默认迭代150次
    dataArr = np.array(dataMatrix)
    m,n = np.shape(dataMatrix)
    weights = np.ones(n)
    weights_array = np.array([])
    for j in range(numIter):
        dataIndex = range(m)
        for i in range(m):
            alpha = 4/(1.0 + i + j) + 0.01                              #alpha在每次迭代的过程中都会调整，避免了数据波动或者高频波动
            randIndex = int(np.random.uniform(0,len(dataIndex)))        #随机选取样本来更新回归系数
            h = sigmoid(sum(dataArr[randIndex] * weights))
            error = classLabels[randIndex] - h
            weights = weights + alpha * error * dataArr[randIndex]
            weights_array = np.append(weights_array,weights)
            del(list(dataIndex)[randIndex])                             #从列表当中选择已经被选中过的数值
    # 改变维度
    weights_array = weights_array.reshape(numIter * m, n)
    # 返回
    return weights, weights_array
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
    x = np.arange(-3.0,3.0,0.1)
    y = (-weights[0]-weights[1]*x)/weights[2]
    ax.plot(x,y)
    plt.title('BestFit')
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()


"""
函数说明：绘制回归系数与迭代次数的关系
Parameters:
    weights_array1 - 回归系数数组1
    weights_array2 - 回归系数数组2

Returns:
    None
Modify:
    2018-07-22
"""


def plotWeights(weights_array1, weights_array2):
    # 设置汉字格式为14号简体字
    font = mfm.FontProperties(fname=r"C:\Windows\Fonts\simsun.ttc", size=14)
    # 将fig画布分隔成1行1列，不共享x轴和y轴，fig画布的大小为（20, 10）
    # 当nrows=3，ncols=2时，代表fig画布被分为6个区域，axs[0][0]代表第一行第一个区域
    fig, axs = plt.subplots(nrows=3, ncols=2, sharex=False, sharey=False, figsize=(20, 10))
    # x1坐标轴的范围
    x1 = np.arange(0, len(weights_array1), 1)
    # 绘制w0与迭代次数的关系
    axs[0][0].plot(x1, weights_array1[:, 0])
    axs0_title_text = axs[0][0].set_title(u'改进的梯度上升算法，回归系数与迭代次数关系', FontProperties=font)
    axs0_ylabel_text = axs[0][0].set_ylabel(u'w0', FontProperties=font)
    plt.setp(axs0_title_text, size=20, weight='bold', color='black')
    plt.setp(axs0_ylabel_text, size=20, weight='bold', color='black')
    # 绘制w1与迭代次数的关系
    axs[1][0].plot(x1, weights_array1[:, 1])
    axs1_ylabel_text = axs[1][0].set_ylabel(u'w1', FontProperties=font)
    plt.setp(axs1_ylabel_text, size=20, weight='bold', color='black')
    # 绘制w2与迭代次数的关系
    axs[2][0].plot(x1, weights_array1[:, 2])
    axs2_title_text = axs[2][0].set_title(u'迭代次数', FontProperties=font)
    axs2_ylabel_text = axs[2][0].set_ylabel(u'w2', FontProperties=font)
    plt.setp(axs2_title_text, size=20, weight='bold', color='black')
    plt.setp(axs2_ylabel_text, size=20, weight='bold', color='black')

    # x2坐标轴的范围
    x2 = np.arange(0, len(weights_array2), 1)
    # 绘制w0与迭代次数的关系
    axs[0][1].plot(x2, weights_array2[:, 0])
    axs0_title_text = axs[0][1].set_title(u'梯度上升算法，回归系数与迭代次数关系', FontProperties=font)
    axs0_ylabel_text = axs[0][1].set_ylabel(u'w0', FontProperties=font)
    plt.setp(axs0_title_text, size=20, weight='bold', color='black')
    plt.setp(axs0_ylabel_text, size=20, weight='bold', color='black')
    # 绘制w1与迭代次数的关系
    axs[1][1].plot(x2, weights_array2[:, 1])
    axs1_ylabel_text = axs[1][1].set_ylabel(u'w1', FontProperties=font)
    plt.setp(axs1_ylabel_text, size=20, weight='bold', color='black')
    # 绘制w2与迭代次数的关系
    axs[2][1].plot(x2, weights_array2[:, 2])
    axs2_title_text = axs[2][1].set_title(u'迭代次数', FontProperties=font)
    axs2_ylabel_text = axs[2][1].set_ylabel(u'w2', FontProperties=font)
    plt.setp(axs2_title_text, size=20, weight='bold', color='black')
    plt.setp(axs2_ylabel_text, size=20, weight='bold', color='black')

    plt.show()

if __name__=='__main__':
    # 加载数据集
    dataMat, labelMat = loadDataSet()
    # 训练权重
    weights2, weights_array2 = gradAscent(dataMat, labelMat)
    # 新方法训练权重
    weights1, weights_array1 = stocGradAscent1(np.array(dataMat), labelMat)
    # 绘制数据集中的y和x的散点图
    plotBestFit(weights1)
    print(gradAscent(dataMat, labelMat))
    #plotWeights(weights_array1, weights_array2)