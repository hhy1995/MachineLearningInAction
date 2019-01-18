#!/usr/bin/env python
# encoding: utf-8
'''
@author: hehaiyang
@contact: 1272114024@qq.com
@file: logRgeTestSklearn.py
@time: 2019/1/11 22:44
'''
from sklearn.linear_model import LogisticRegression

def colicSklearn(iter_times):
    frTrain = open('horseColicTraining.txt')
    frTest = open('horseColicTest.txt')
    trainingSet = []
    trainingLabels = []
    testSet = []
    testLabels = []
    for line in frTrain.readlines():
        currentLine = line.strip().split('\t')
        lineArr = []
        for i in range(len(currentLine) - 1):
            lineArr.append(float(currentLine[i]))
        trainingSet.append(lineArr)
        trainingLabels.append(float(currentLine[-1]))
    for line in frTest.readlines():
        currentLine = line.strip().split('\t')
        lineArr = []
        for i in range(len(currentLine) - 1):
            lineArr.append(float(currentLine[i]))
        testSet.append(lineArr)
        testLabels.append(float(currentLine[-1]))
    classifier = LogisticRegression(solver='liblinear', max_iter=iter_times).fit(trainingSet, trainingLabels)
    test_accuracy = classifier.score(testSet, testLabels) * 100
    print('Iter_times %d ,The accuracy:%f%%' % (iter_times,test_accuracy))
    return test_accuracy

def mutilTest():
    numTests = 10
    sum_accuracy = 0.0
    iter_times = 2
    for i in range(numTests):
        sum_accuracy += colicSklearn(iter_times)
        iter_times += 2
    ave_accuracy = sum_accuracy / float(numTests)
    #准确率基本稳定在73%多一些
    print("After %d iterations the average accuracy rate is: %f" % (numTests, ave_accuracy))

if __name__ == '__main__':
    mutilTest()



