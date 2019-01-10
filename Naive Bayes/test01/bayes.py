#!/usr/bin/env python
# encoding: utf-8
'''
@author: hehaiyang
@contact: 1272114024@qq.com
@file: bayes.py
@time: 2019/1/7 19:41
'''

import numpy as np


"""
函数说明：创建实验样本
Parameters:
    None

Returns:
    postingList - 进行词条切分后的文档的集合
    classVec - 类别标签向量（主要包括侮辱性和非侮辱性）
"""


def loadDataSet():
    # 切分的词条
    postingList = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                   ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                   ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                   ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                   ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                   ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0, 1, 0, 1, 0, 1]                                        # 类别标签向量，1代表侮辱性词汇，0代表正常言论
    return postingList, classVec                                         # 返回实验样本切分的词条、类别标签向量
'''
函数说明：创建一个包含在所有文档中出现的不重复词的列表
    @:parameter
        dataSet :输入的实验样本
    @:return
        不重复词的列表

'''
def createVocabList(dataSet):
    vocabSet = set([])                                                 #创建一个空的集合
    for document in dataSet:
        vocabSet = vocabSet | set(document)                            #创建两个集合的并集
    return list(vocabSet)

'''
函数说明：
    @:parameter
        vocabList 词汇表
        inputSet 某个文档
    @:return
        returnVec 文档向量
'''

def setOfWords2Vec(vocabList,inputSet):
    returnVec = [0]*len(vocabList)                                    #创建一个其中所含元素都为0的向量,与词汇表等长的向量
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)]=1                        #如果出现了词汇表中的词，将文档向量对应元素位置置0
        else:
             print("The word %s is not in my VocabList!" % word)
    return returnVec
'''
函数说明：朴素贝叶斯分类器训练函数
    @:parameter
        trainMatrix
        trainCategory
    @:return
        p0Vect  属于侮辱类的条件概率数组
        p1Vect  属于非侮辱类的条件概率数组
        pAbusive;   文档属于侮辱类的概率
'''
def trainNB0(trainMatrix,trainCategory):
    numTrainDocs = len(trainMatrix)                                    #计算每个类别的文档数
    numWords = len(trainMatrix[0])                                    #计算每篇文档的词条数
    pAbusive = sum(trainCategory)/float(numTrainDocs)                    #文档属于侮辱类的概率
    #初始化概率
    p0Num = np.ones(numWords)                                        #初始化概率,初始化分子分母变量
    p1Num = np.ones(numWords)                                        #为了避免某个值概率为0的情况，进行拉普拉斯平滑，将所有词条出现次数置1，
    p0Denom = 2.0                                                    #认为每个词至少出现一次，并且将分母初始化为2
    p1Denom = 2.0
    for i in range(numTrainDocs):
        #统计属于侮辱类的条件概率所需要的数据，
        if trainCategory[i] == 1:
            p1Num += trainMatrix[i]                         #计算侮辱性词语总数
            p1Denom += sum(trainMatrix[i])
        #统计属于非侮辱类的条件概率所需要的数据
        else:
            p0Num += trainMatrix[i]                         #计算正常词语总数
            p0Denom += sum(trainMatrix[i])
    #避免下溢出问题，对乘积取自然对数，  ln(a*b) = ln(a) + ln(b)
    p1Vect = np.log(p1Num/p1Denom);                         #每个侮辱类单词出现的概率
    p0Vect = np.log(p0Num/p0Denom);                         #每个非侮辱类单词出现的概率
    return p0Vect,p1Vect,pAbusive;
'''
函数说明：朴素贝叶斯分类器分类函数
@:parameter
    vec2Classify  待分类的词条向量
    p0Vec   侮辱性词条概率数组
    p1Vec   正常词条概论数组
    pClass1     文档属于侮辱类的概率
@:return
    1   代表侮辱类
    0   代表正常词条
'''
def classifyNB(vec2Classify,p0Vec,p1Vec,pClass1):
    #对应元素相乘
    p1 = sum(vec2Classify * p1Vec) + np.log(pClass1)
    p0 = sum(vec2Classify * p0Vec) + np.log(1.0 - pClass1)
    print("p0",p0)
    print("p1",p1)
    if p1 > p0:                                             #如果p1>p0 认为其是侮辱性词
        return 1
    else:                                                   #否则认为其实正常词
        return 0

'''
函数说明：朴素贝叶斯测试函数
    @:parameter
        None
    $:return
        预测结果

'''
def testNB():
    postingList,classVec = loadDataSet()
    myVocabList = createVocabList(postingList)
    trainMat = []
    for postinDoc in postingList:
        trainMat.append(setOfWords2Vec(myVocabList,postinDoc))
    p0V,p1V,pAb = trainNB0(trainMat,classVec)
    testEntry01 = ['love','my','dalmation']
    #对词条进行向量化，构建词向量
    testEntry01Doc = setOfWords2Vec(myVocabList,testEntry01)
    #分类的结果
    result01 = classifyNB(testEntry01Doc,p0V,p1V,pAb)
    if result01 == 1:
        print("侮辱性言论\n")
    else:
        print("正常言论\n")

    testEntry02 = ['stupid','garbage']
    #对词条进行向量化，构建词向量
    testEntry02Doc = setOfWords2Vec(myVocabList,testEntry02)
    #分类的结果
    result02 = classifyNB(testEntry02Doc,p0V,p1V,pAb)
    if result02 == 1:
        print("侮辱性言论\n")
    else:
        print("正常言论\n")

if __name__=='__main__':
    #postingList,classVec = loadDataSet()
    #print("postingList:\n", postingList)
    #myVocabList = createVocabList(postingList)
    #print("myVocabList:\n",myVocabList)
    #trainMat = []
    #for postinDoc in postingList:
    #    trainMat.append(setOfWords2Vec(myVocabList,postinDoc))
    #print("trainMat:\n",trainMat)
    #p0V,p1V,pAb = trainNB0(trainMat,classVec)
    #print("pAb:",pAb)
    #print("p0V:\n",p0V)
    #print("p1V:\n",p1V)
    testNB()



