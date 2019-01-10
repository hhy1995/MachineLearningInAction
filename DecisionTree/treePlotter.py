#!/usr/bin/env python
# encoding: utf-8
'''
@author: hehaiyang
@contact: 1272114024@qq.com
@file: treePlotter.py
@time: 2019/1/5 21:07
'''
from matplotlib.font_manager import FontProperties
import _elementtree
import matplotlib.pyplot as plt
import math
import operator
import treePlotter


def calcShannonEnt(dataSet):
    numEntries = len(dataSet)  # 得到数据集的行数
    labelCounts = {}  # 类标签的字典，其键值是最后一列的数值
    # 为所有的可能分类创建字典
    for featVex in dataSet:  # 当前标签
        currentLabel = featVex[-1]
        if currentLabel not in labelCounts.keys():  # 如果当前的标签不在标签字典当中，加入标签字典当中，并且标签数量加1
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    shannonEnt = 0.0  # 香农信息熵
    for key in labelCounts:
        prob = float(labelCounts[key]) / numEntries;  # 选择该标签的概率
        shannonEnt -= prob * math.log(prob, 2)  # 计算香农信息熵
    return shannonEnt


def createData():
    dataSet = [
        [1, 1, 'yes'],
        [1, 1, 'yes'],
        [1, 0, 'no'],
        [0, 1, 'no'],
        [0, 1, 'no'],
    ]

    labels = ['no surfacing', 'flippers']

    return dataSet, labels


'''
函数说明：按照给定的特征划分数据集
    @:parameter
        dataSet  待划分的数据集
        axis   划分数据集的特征
        value   需要返回的特征的值
'''


def splitDataSet(dataSet, axis, value):
    # 创建新的list对象
    returnDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            # 将符合特征的数据抽取出来
            reducedFeatVec = featVec[:axis]
            reducedFeatVec.extend(featVec[axis + 1:])
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
    numFeatures = len(dataSet[0]) - 1  # 计算数据集的特征属性的数量
    baseEntropy = calcShannonEnt(dataSet)  # 计算整个数据集的熵值
    bestInfoGain = 0.0
    bestFeature = -1
    for i in range(numFeatures):
        featList = [example[i] for example in dataSet]  # 创建一个唯一的分类标签列表
        uniqueVals = set(featList)  # 从列表中创建集合，使得列表中的元素唯一
        newEntropy = 0.0
        for value in uniqueVals:  # 计算每种划分方式的信息熵
            subDataSet = splitDataSet(dataSet, i, value)  # 根据属性值value划分数据集
            prob = len(subDataSet) / float(len(dataSet))
            newEntropy += prob * calcShannonEnt(subDataSet)  # 对所有唯一特征值得到的熵求和
        infoGain = baseEntropy - newEntropy  # 信息增益
        # print("第%d个特征的信息增益为%.3f" % (i,infoGain))
        if (infoGain > bestInfoGain):  # 计算最好的信息增益
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature  # 返回最好特征划分的索引值


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
    for vote in classList:  # 统计classList中每个元素出现的次数
        if vote not in classCount.keys(): classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)  # 根据字典序降序排序

    return sortedClassCount[0][0]  # 返回出现次数最多的分类名称


'''
函数说明:创建决策树（递归函数）
Parameters:
    dataSet - 训练数据集
    labels - 分类属性标签
    featLabels - 存储选择的最优特征标签
Returns:
    myTree - 决策树
'''


def createTree(dataSet, labels):
    classList = [example[-1] for example in dataSet]  # 取数据集的类别标签
    if classList.count(classList[0]) == len(classList):
        return classList[0]  # 递归停止条件一：如果类别完全相同则停止继续划分
    if len(dataSet[0]) == 1:  # 递归停止条件二：遍历完所有特征时返回出现次数最多的类标签
        return majorityCnt(classList)
    bestFeat = chooseBestFeatureToSplit(dataSet)  # 选择最优特征
    bestFeatLabel = labels[bestFeat]  # #最优特征的标签
    myTree = {bestFeatLabel: {}}  # 根据最优特征的标签生成树
    del (labels[bestFeat])
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        subLabels = labels[:]  # 复制标签，递归创建决策树
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value), subLabels)
    return myTree



#定义文本框和箭头形式
decisionNode = dict(boxstyle="sawtooth",fc="0.8")
leafNode = dict(boxstyle="round4", fc = "0.8")

def createDataSet():
    dataSet = [[0, 0, 0, 0, 'no'],                        #数据集
            [0, 0, 0, 1, 'no'],
            [0, 1, 0, 1, 'yes'],
            [0, 1, 1, 0, 'yes'],
            [0, 0, 0, 0, 'no'],
            [1, 0, 0, 0, 'no'],
            [1, 0, 0, 1, 'no'],
            [1, 1, 1, 1, 'yes'],
            [1, 0, 1, 2, 'yes'],
            [1, 0, 1, 2, 'yes'],
            [2, 0, 1, 2, 'yes'],
            [2, 0, 1, 1, 'yes'],
            [2, 1, 0, 1, 'yes'],
            [2, 1, 0, 2, 'yes'],
            [2, 0, 0, 0, 'no']]
    labels = ['年龄', '有工作', '有自己的房子', '信贷情况']        #特征标签
    return dataSet, labels


def createPlot():
    fig = plt.figure(1,facecolor='white')
    fig.clf()
    createPlot().ax1 = plt.subplots(111,frameon=False)
    plotNode('决策节点',(0.5,0.1),(0.1,0.5),decisionNode)
    plotNode('叶节点',(0.8,0.1),(0.3,0.8),leafNode)
    plt.show()


'''
函数说明：选择最好的数据集划分方式
    @:parameter
        dataSet  输入的数据集
    @:return
        最好的划分数据集的特征
'''

'''
函数说明：获取叶节点的数目
    @:parameter myTree
        输入参数是决策树
    @:return    numLeafs
        输出决策树叶子节点的数目
'''
def getNumLeafs(myTree):
    numLeafs = 0                                        #初始化叶子节点
    firstStr = next(iter(myTree))
    secondDict = myTree[firstStr]                       #获取下一组字典
    for key in secondDict.keys():
        if type(secondDict[key]).__name__=='dict':      #判断该节点是否是字典
            numLeafs += getNumLeafs(secondDict[key])
        else: numLeafs += 1
    return numLeafs                                     #返回叶子节点的数目

'''
函数说明：求决策树最大的深度
    @:parameter     myTree
        输入参数是决策树
    @:return    maxDepth
        输出结果是决策树最大的深度
'''
def getTreeDepth(myTree):
    maxDepth = 0                                        #初始化决策树的深度为0
    firstStr = next(iter(myTree))
    secondDict = myTree[firstStr]                       #获取下一个字典
    for key in secondDict.keys():
        if type(secondDict[key]).__name__=='dict':      #type()函数可以判断该节点是不是字典
            thisDepth = 1 + getTreeDepth(secondDict[key])
        else: thisDepth = 1
        if thisDepth > maxDepth:
            maxDepth = thisDepth
    return maxDepth                                     #返回树的最大的深度

'''
函数说明：预先存储树的信息，避免了每次测试代码时都要从数据中创建树的麻烦

'''
def retrieveTree(i):
    listOfTrees =[{'no surfacing': {0: 'no', 1: {'flippers': {0: 'no', 1: 'yes'}}}},
                  {'no surfacing': {0: 'no', 1: {'flippers': {0: {'head': {0: 'no', 1: 'yes'}}, 1: 'no'}}}}
                  ]
    return listOfTrees[i]


'''
函数说明：绘制带箭头的注解
        使用文本注解绘制节点
'''

def plotNode(nodeTxt,centerPt,parentPt,nodeType):
    arrow_args = dict(arrowstyle="<-")
    font = FontProperties(fname=r"c:\windows\fonts\simsun.ttc",size=15)
    createPlot.ax1.annotate(nodeTxt,xy=parentPt,xycoords='axes fraction',xytext=centerPt,textcoords='axes fraction',
                            va="center",ha="center",bbox=nodeType,arrowprops=arrow_args,FontProperties=font)
'''
函数说明：在父子节点间填充文本信息
    @:parameter
        cntrPt 、parentPt 用于计算标注的位置
        txtString 需要填充的文本信息
    无返回值
'''

def plotMidText(cntrPt,parentPt,txtString):
    xMid = (parentPt[0] - cntrPt[0])/2.0 + cntrPt[0]       #计算父节点和子节点的中间位置（xMid,yMid）
    yMid = (parentPt[1] - cntrPt[1])/2.0 + cntrPt[1]
    createPlot().ax1.text(xMid,yMid,txtString,va="center", ha="center", rotation=30)

def plotTree(myTree, parentPt, nodeTxt):
    decisionNode = dict(boxstyle="sawtooth", fc="0.8")                                        #设置结点格式
    leafNode = dict(boxstyle="round4", fc="0.8")                                            #设置叶结点格式
    numLeafs = getNumLeafs(myTree)                                                          #获取决策树叶结点数目，决定了树的宽度
    depth = getTreeDepth(myTree)                                                            #获取决策树层数
    firstStr = next(iter(myTree))                                                            #下个字典
    cntrPt = (plotTree.xOff + (1.0 + float(numLeafs))/2.0/plotTree.totalW, plotTree.yOff)    #中心位置
    plotMidText(cntrPt, parentPt, nodeTxt)                                                    #标注有向边属性值,标记子节点的属性值
    plotNode(firstStr, cntrPt, parentPt, decisionNode)                                        #绘制结点
    secondDict = myTree[firstStr]                                                            #下一个字典，也就是继续绘制子结点
    plotTree.yOff = plotTree.yOff - 1.0/plotTree.totalD                                        #y偏移
    for key in secondDict.keys():
        if type(secondDict[key]).__name__=='dict':                                            #测试该结点是否为字典，如果不是字典，代表此结点为叶子结点
            plotTree(secondDict[key],cntrPt,str(key))                                        #不是叶结点，递归调用继续绘制
        else:                                                                                #如果是叶结点，绘制叶结点，并标注有向边属性值
            plotTree.xOff = plotTree.xOff + 1.0/plotTree.totalW
            plotNode(secondDict[key], (plotTree.xOff, plotTree.yOff), cntrPt, leafNode)
            plotMidText((plotTree.xOff, plotTree.yOff), cntrPt, str(key))
    plotTree.yOff = plotTree.yOff + 1.0/plotTree.totalD
'''
函数说明:创建绘制面板
Parameters:
    inTree - 决策树(字典)
Returns:
    无
'''


def createPlot(inTree):
    fig = plt.figure(1, facecolor='white')
    fig.clf()  # 清空fig
    axprops = dict(xticks=[], yticks=[])
    createPlot.ax1 = plt.subplot(111, frameon=False, **axprops)  # 去掉x、y轴
    plotTree.totalW = float(getNumLeafs(inTree))
    plotTree.totalD = float(getTreeDepth(inTree))
    plotTree.xOff = -0.5 / plotTree.totalW;
    plotTree.yOff = 1.0;
    plotTree(inTree, (0.5, 1.0), '')
    plt.show()


if __name__ == '__main__':
    dataSet, labels = createData()
    myTree = createTree(dataSet, labels)
    treePlotter.createPlot(myTree)
    print(myTree)



