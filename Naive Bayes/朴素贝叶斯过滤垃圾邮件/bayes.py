#!/usr/bin/env python
# encoding: utf-8
'''
@author: hehaiyang
@contact: 1272114024@qq.com
@file: bayes.py
@time: 2019/1/8 16:29
'''

import numpy as np
import re
import random

'''
函数说明：创建一个包含在所有文档中出现的不重复词的列表
    @:parameter
        dataSet :输入的实验样本
    @:return
        不重复的词列表

'''
def createVocabList(dataSet):
    vocabSet = set([])                                                 #创建一个空的集合
    for document in dataSet:
        vocabSet = vocabSet | set(document)                            #创建两个集合的并集
    return list(vocabSet)

'''
文档词集模型
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
文档词袋模型：每个单词可以出现多次

函数说明：朴素贝叶斯词袋模型
    @:parameter
        vocabList  词向量列表
        inputSet    某个文档
    @:return
        returnVec   文档的词袋模型
'''
def bagOfWords2VecMN(vocabList,inputSet):
    returnVec = [0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1                   #与词集模型的区别是这边是增加词向量的对应值
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
    #print("p0",p0)
    #print("p1",p1)
    if p1 > p0:                                             #如果p1>p0 认为其是侮辱性词
        return 1
    else:                                                   #否则认为其实正常词
        return 0
'''
函数说明：将一个字符串解析成字符串列表
    @:parameter
        bigString 输入的文本
    @:return
        解析得到的字符串列表
'''
def textParse(bigString):
    listOfTokens = re.split(r'\\W*',bigString)
    #去掉解析过程中的空字符串，并且将大写字母转化成小写字母
    return [tok.lower() for tok in listOfTokens if len(tok)>2]

def spamTest():
    docList = []
    classList = []
    fullText = []
    #导入并且解析文本文件
    for i in range(1,26):
        wordList = textParse(open('email/spam/%d.txt' % i,'r').read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)                                         #标记1为垃圾邮件
        wordList = textParse(open('email/ham/%d.txt' % i,'r').read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)                                         #标记0为正常邮件
    #随机选择10个文件构建训练集
    vocabList = createVocabList(docList)
    trainingSet = list(range(50))
    testSet = []
    for i in range(10):
        randIndex = int(random.uniform(0,len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex])
    trainMat = []
    trainClasses = []
    for docIndex in trainingSet:
        trainMat.append(setOfWords2Vec(vocabList,docList[docIndex]))
        trainClasses.append(classList[docIndex])

    #训练朴素贝叶斯模型，计算得到垃圾邮件的条件概率，正常邮件的概率，
    p0V,p1V,pSpam = trainNB0(trainMat,trainClasses)

    #分类错误的次数
    errorCount = 0.0

    #对测试集进行分类
    for docIndex in testSet:
        #构造测试集词向量
        wordVector = setOfWords2Vec(vocabList,docList[docIndex])
        if classifyNB(wordVector,p0V,p1V,pSpam) != classList[docIndex]:             #如果分类结果和实际不相符合，则邮件分类错误，错误数加1
            errorCount += 1
    #输出错误率
    print("错误率:%.2f%%" % (float(errorCount) / len(testSet) * 100))

if __name__=="__main__":
    spamTest()



