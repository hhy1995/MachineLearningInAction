#!/usr/bin/env python
# encoding: utf-8
'''
@author: hehaiyang
@contact: 1272114024@qq.com
@file: trees.py
@time: 2019/1/5 14:48
'''

import math
import operator


def calcShannonEnt(dataSet):
    numEntries = len(dataSet)                           #得到数据集的行数
    labelCounts = {}                                    #类标签的字典，其键值是最后一列的数值
    #为所有的可能分类创建字典
    for featVec in dataSet:                             #当前标签
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys():      #如果当前的标签不在标签字典当中，加入标签字典当中，并且标签数量加1
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    shannonEnt = 0.0                                    #香农信息熵
    for key in labelCounts:
        prob = float(labelCounts[key])/numEntries;      #选择该标签的概率
        shannonEnt -= prob * math.log(prob,2)           #计算香农信息熵
    return shannonEnt

def createData():
    dataSet = [
        [1, 1, 'yes'],
        [1, 1, 'yes'],
        [1, 0, 'no'],
        [0, 1, 'no'],
        [0, 1, 'no'],
    ]

    labels = ['no surfacing','flippers']

    return dataSet,labels
'''
函数说明：按照给定的特征划分数据集
    @:parameter
        dataSet  待划分的数据集
        axis   划分数据集的特征
        value   需要返回的特征的值
'''
def splitDataSet(dataSet,axis,value):
    #创建新的list对象
    returnDataSet = []
    for featVec in dataSet:
        if featVec[axis] ==value:
            #将符合特征的数据抽取出来
            reducedFeatVec = featVec[:axis]
            reducedFeatVec.extend(featVec[axis+1:])
            returnDataSet.append(reducedFeatVec)
    return returnDataSet


'''
函数说明：选择最好的数据集划分方式
    @:parameter
        dataSet  输入的数据集
    @:return
        最好的划分数据集的特征
'''
def chooseBestFeatureToSplit(dataSet):
    numFeatures = len(dataSet[0]) - 1                       #计算数据集的特征属性的数量
    baseEntropy = calcShannonEnt(dataSet)                   #计算整个数据集的熵值
    bestInfoGain = 0.0
    bestFeature = -1
    for i in range(numFeatures):
        featList = [example[i] for example in dataSet]      #创建一个唯一的分类标签列表
        uniqueVals = set(featList)                          #从列表中创建集合，使得列表中的元素唯一
        newEntropy = 0.0
        for value in uniqueVals:                            #计算每种划分方式的信息熵
            subDataSet = splitDataSet(dataSet,i,value)      #根据属性值value划分数据集
            prob = len(subDataSet)/float(len(dataSet))
            newEntropy += prob*calcShannonEnt(subDataSet)   #对所有唯一特征值得到的熵求和
        infoGain = baseEntropy - newEntropy                 #信息增益
        #print("第%d个特征的信息增益为%.3f" % (i,infoGain))
        if(infoGain > bestInfoGain):                        #计算最好的信息增益
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature                                      #返回最好特征划分的索引值

'''
上面代码完成了经验熵的计算和最优特征值的选择：
    得到原始数据集然后基于最好的属性值划分数据集，由于特蒸煮可能多于两个，因此存在大于两个分支的数据集划分。
第一次划分之后，数据集被向下传递到树的分支的下一个节点。在这个节点上，再次划分数据。采用递归的原则处理数据集。

'''


'''
函数说明：统计在classList当中出现最多元素（类标签）
    @:parameter
        classList 类标签列表
    @:return
        sortedClassCount[0][0]
    次数最多的分类名称
'''
def majorityCnt(classList):
    classCount = {}
    for vote in classList:                                  #统计classList中每个元素出现的次数
        if vote not in classCount.keys():classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.items(),key = operator.itemgetter(1),reverse=True)             #根据字典序降序排序

    return sortedClassCount[0][0]                           #返回出现次数最多的分类名称


def createTree(dataSet,labels,featLables):
    classList = [example[-1] for example in dataSet]
    if classList.count(classList[0]) == len(classList):     #如果所有类标签完全相同，则直接返回该类标签
        return classList[0]
    if len(dataSet[0]) == 1 or len(labels) == 0:
        return majorityCnt(classList)
    bestFeat = chooseBestFeatureToSplit(dataSet)
    bestFeatLabel = labels[bestFeat]
    featLables.append(bestFeatLabel)
    myTree = {bestFeatLabel:{}}                             #使用字典类型存储树的信息
    #得到列表包含的所有的属性值
    del(labels[bestFeat])
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)
    for value in uniqueVals:                                #递归构造树结构
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet,bestFeat,value),labels,featLables)

    return myTree
'''
接下来可以进行决策树可视化显示

'''

def classify(inputTree,featLables,testVec):
    firstStr = next(iter(inputTree))
    secondDict = inputTree[firstStr]
    featIndex = featLables.index(firstStr)
    for key in secondDict.keys():
        if testVec[featIndex] == key:
            if type(secondDict[key]).__name__=='dict':
                classLabel = classify(secondDict[key],featLables,testVec)
            else: classLabel = secondDict[key]
    return classLabel

if __name__=="__main__":
    dataSet,labels = createData()
    print(labels)
    featLables = []
    myTree = createTree(dataSet,labels,featLables)
    testVec = [1,1]                                     #测试数据
    result = classify(myTree,featLables,testVec)
    if result=='yes':
        print('鱼')
    if result=='no':
        print('非鱼')




'''
测试最好的数据集的划分方式
'''
'''

if __name__=="__main__":
    dataSet,labels = createData();
    print(str(chooseBestFeatureToSplit(dataSet)));


'''

'''
if __name__=="__main__":
    myData,labels = createData();

    print(splitDataSet(myData,0,1))
    print(splitDataSet(myData,0,0))
'''

'''
if __name__=="__main__":
    myData,labels = createData();

    print(myData)
    print(calcShannonEnt(myData))

    myData [0][-1] = 'maybe'
    print(myData)
    print(calcShannonEnt(myData))
'''