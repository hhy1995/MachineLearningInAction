#!/usr/bin/env python
# encoding: utf-8
'''
从疝气病症预测病马的死亡率
@author: hehaiyang
@contact: 1272114024@qq.com
@file: logRgeTest.py
@time: 2019/1/11 17:55
'''
import numpy as np

"""
函数说明：sigmoid函数
    将任意的函数值映射到（0，1）区间上，转化成概率值

"""
def sigmoid(inX):
    return 1.0/(1 + np.exp(-inX))

"""
函数说明：改进的随机梯度下降算法
    上述的stocGradAscent0算法存在的问题：收敛慢，需要至少200次迭代过程才能收敛；大的波动停止之后还存在一些小的周期性的波动，
    原因是样本中存在一些不是先行可分的样本点，每次迭代的过程中存在着回归系数的剧烈改变。
"""
def stocGradAscent1(dataMatrix,classLabels,numIter = 150):              #默认迭代150次
    dataArr = np.array(dataMatrix)
    m,n = np.shape(dataMatrix)
    weights = np.ones(n)
    for j in range(numIter):
        dataIndex = range(m)
        for i in range(m):
            alpha = 4/(1.0 + i + j) + 0.01                              #alpha在每次迭代的过程中都会调整，避免了数据波动或者高频波动
            randIndex = int(np.random.uniform(0,len(dataIndex)))        #随机选取样本来更新回归系数
            h = sigmoid(sum(dataArr[randIndex] * weights))
            error = classLabels[randIndex] - h
            weights = weights + alpha * error * dataArr[randIndex]
            del(list(dataIndex)[randIndex])                             #从列表当中选择已经被选中过的数值
    # 返回权重参数向量
    return weights

"""
下面这部分是：从疝气病症预测病马死亡率的代码

"""
"""
回归分类函数
@:parameter
    inX    回归系数
    weights     梯度上升法求得的特征向量
@:return
     1  活着
     0  已经死亡
     
     这边对确实值的处理是，用0来代替，因为sigmoid(0) = 0.5 ，它对结果的预测不具有任何倾向性
"""
def classifyVector(inX,weights):
    prob = sigmoid(sum(inX*weights))
    if prob > 0.5 :
        return 1
    else:
        return 0

"""
函数说明：用python写的Logistic分类器做预测
    接收数据，并且根据数据的格式对数据进行处理，根据测试集计算预测的准确率
@:parameter
    None
@:return
    errorRate   预测的准确率

"""
def colicTest():
    #分别打开训练文本和测试文本
    frTrain = open('horseColicTraining.txt')
    frTest = open('horseColicTest.txt')
    #训练集集合和训练集标签
    trainingSet = []
    trainingLabels = []
    for line in frTrain.readlines():                                #分别读取训练集的每一行数据
        currentLine = line.strip().split('\t')                      #将每一行数据去除行首行尾空格，并且用\t进行切分
        lineArr = []
        for i in range(21):
            lineArr.append(float(currentLine[i]))
        trainingSet.append(lineArr)
        trainingLabels.append(float(currentLine[21]))               #最后一列为样本类别标签
    trainWeights = stocGradAscent1(np.array(trainingSet),trainingLabels,1000)        #使用改进的随机梯度上升训练算法计算回归系数向量
    errorCount = 0                                                  #记录分类错误的个数
    numTestVector = 0.0                                             #记录测试集和总数

    for line in frTest.readlines():
        numTestVector += 1.0
        currentLine = line.strip().split('\t')
        lineArr = []
        for i in range(21):
            lineArr.append(float(currentLine[i]))
        if int(classifyVector(np.array(lineArr),trainWeights)) != int(currentLine[21]):
            errorCount += 1
    frTrain.close()
    frTest.close()
    errorRate = float(errorCount)/numTestVector                    #计算错误率
    print('Error Rate:%f' % errorRate)                             #输出错误率
    return errorRate                                               #返回错误率，供multiTest使用

def multiTest():
    numTests = 50
    errorSum = 0.0
    for k in range(numTests):
        errorSum += colicTest()
    print("After %d iterations the average error rate is: %f" % (numTests,errorSum/float(numTests)))

if __name__=='__main__':
    multiTest()

