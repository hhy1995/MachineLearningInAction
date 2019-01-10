#!/usr/bin/env python
# encoding: utf-8
'''
@author: hehaiyang
@contact: 1272114024@qq.com
@file: KNN.py
@time: 2019/1/4 13:56
'''

from matplotlib.font_manager import FontProperties
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import operator

'''
k-近邻算法
函数说明
    @:parameter
        inX 用于分类的输入向量
        dataSet 输入的样本训练集
        labels 标签向量
        k 用于选择最近邻居的数目
    @:return
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
将文本记录转化成Numpy的解析程序
    对数据进行分类：1代表不喜欢的人 2代表魅力一般的人  3代表极具魅力的人
    @:parameter
        filename  文件名
    

'''
def fileToMatrix(filename):
    #打开文件，得到文件行数
    fr = open(filename)
    #读取文件的所有内容
    arrayOLines = fr.readlines()
    #得到文件的行数
    numberOfLines = len(arrayOLines)

    #创建返回的NumPy矩阵
    #返回NumPy矩阵（实际上是一个二维数组）：前一个参数是行，后一个参数是列
    returnMat = np.zeros((numberOfLines,3))
    #创建类别标签向量
    classLabelVector = []
    #行的索引值
    index = 0
    #解析文件数据到列表
    for line in arrayOLines:
        line = line.strip()
        #line.split()截取掉所有的回车符，用'\t'将整行数据分割成一个元素列表
        listFromLine = line.split('\t')
        #选区前3个元素，存到特征矩阵中
        returnMat[index,:] = listFromLine[0:3]
        # -1表示列表中的最后一个元素
        #根据文本中标记的喜欢程度进行分类
        if listFromLine[-1] == 'didntLike':
            classLabelVector.append(1)
        elif listFromLine[-1] == 'smallDoses':
            classLabelVector.append(2)
        elif listFromLine[-1] == 'largeDoses':
            classLabelVector.append(3)
        index += 1
    return returnMat,classLabelVector
'''
    将数据可视化显示
    @:parameter 
        datingDataMat  经过转换的训练样本矩阵
        datingLabel  样本标签向量
    @:return
        可视化散点图
'''

def showDatas(datingDataMat,datingLabels):
    #设置汉字格式
    font = FontProperties(fname=r"c:\windows\fonts\simsun.ttc",size=14)
    #将fig画布分割成一行一列，不共享x轴和y轴，画布大小为（14，8）
    #nrows=2, ncols=2, 画布被分成四个区域   axs[0][0]代表第一个区域
    fig,axs = plt.subplots(nrows=2, ncols=2, sharex=False, sharey=False, figsize=(13, 8))

    #获得标签的长度
    numberOfLabels = len(datingLabels)
    LabelsColors = []
    for i in datingLabels:
        if i==1:
            LabelsColors.append('yellow')
        if i==2:
            LabelsColors.append('red')
        if i==3:
            LabelsColors.append('blue')

    # 画出散点图,以datingDataMat矩阵的第一(飞行常客例程)、第二列(玩游戏)数据画散点数据,散点大小为15,透明度为0.5
    axs[0][0].scatter(x=datingDataMat[:,0], y=datingDataMat[:,1], color=LabelsColors, s=15, alpha=.5)
    # 设置标题,x轴label,y轴label
    axs0_title_text = axs[0][0].set_title(u'每年获得的飞行常客里程数与玩视频游戏所消耗时间占比', FontProperties=font)
    axs0_xlabel_text = axs[0][0].set_xlabel(u'每年获得的飞行常客里程数', FontProperties=font)
    axs0_ylabel_text = axs[0][0].set_ylabel(u'玩视频游戏所消耗时间占', FontProperties=font)
    plt.setp(axs0_title_text, size=9, weight='bold', color='red')
    plt.setp(axs0_xlabel_text, size=7, weight='bold', color='black')
    plt.setp(axs0_ylabel_text, size=7, weight='bold', color='black')

    # 画出散点图,以datingDataMat矩阵的第一(飞行常客例程)、第三列(冰激凌)数据画散点数据,散点大小为15,透明度为0.5
    axs[0][1].scatter(x=datingDataMat[:,0], y=datingDataMat[:,2], color=LabelsColors, s=15, alpha=.5)
    # 设置标题,x轴label,y轴label
    axs1_title_text = axs[0][1].set_title(u'每年获得的飞行常客里程数与每周消费的冰激淋公升数', FontProperties=font)
    axs1_xlabel_text = axs[0][1].set_xlabel(u'每年获得的飞行常客里程数', FontProperties=font)
    axs1_ylabel_text = axs[0][1].set_ylabel(u'每周消费的冰激淋公升数', FontProperties=font)
    plt.setp(axs1_title_text, size=9, weight='bold', color='red')
    plt.setp(axs1_xlabel_text, size=7, weight='bold', color='black')
    plt.setp(axs1_ylabel_text, size=7, weight='bold', color='black')

    # 画出散点图,以datingDataMat矩阵的第二(玩游戏)、第三列(冰激凌)数据画散点数据,散点大小为15,透明度为0.5
    axs[1][0].scatter(x=datingDataMat[:,1], y=datingDataMat[:,2], color=LabelsColors, s=15, alpha=.5)
    # 设置标题,x轴label,y轴label
    axs2_title_text = axs[1][0].set_title(u'玩视频游戏所消耗时间占比与每周消费的冰激淋公升数', FontProperties=font)
    axs2_xlabel_text = axs[1][0].set_xlabel(u'玩视频游戏所消耗时间占比', FontProperties=font)
    axs2_ylabel_text = axs[1][0].set_ylabel(u'每周消费的冰激淋公升数', FontProperties=font)
    plt.setp(axs2_title_text, size=9, weight='bold', color='red')
    plt.setp(axs2_xlabel_text, size=7, weight='bold', color='black')
    plt.setp(axs2_ylabel_text, size=7, weight='bold', color='black')
    # 设置图例
    didntLike = mlines.Line2D([], [], color='black', marker='.',
                              markersize=6, label='didntLike')
    smallDoses = mlines.Line2D([], [], color='orange', marker='.',
                               markersize=6, label='smallDoses')
    largeDoses = mlines.Line2D([], [], color='red', marker='.',
                               markersize=6, label='largeDoses')
    # 添加图例
    axs[0][0].legend(handles=[didntLike, smallDoses, largeDoses])
    axs[0][1].legend(handles=[didntLike, smallDoses, largeDoses])
    axs[1][0].legend(handles=[didntLike, smallDoses, largeDoses])
    # 显示图片
    plt.show()

'''
    归一化特征值函数
    @:parameter
        dataSet  数据集

    归一化公式：newValue = (oldValue - min)/(max - min)
    @:return
        normDataSet 数据归一化后得到的举证
        ranges 数据取值范围
        minVals 数据最小值
'''
def autoNorm(dataSet):
    #获取数据集每列的最大和最小的值   共有三列minVals maxVals 和 ranges的值都是1×3的
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    #得到数据的可能的取值范围
    ranges = maxVals - minVals
    #创建新的返回矩阵
    normDataSet = np.zeros(np.shape(dataSet))
    m = dataSet.shape[0]
    #tile 重复数组minVals，（m，1）次来构建新的数组
    normDataSet = dataSet - np.tile(minVals,(m,1))
    #特征值相除
    normDataSet = normDataSet/np.tile(ranges,(m,1))
    return normDataSet,ranges,minVals
'''
    完整程序验证分类器
    随机选择10%的数据去测试分类器，
'''
def datingClassTest():
    #划分出10%的数据去测试分类器
    hoRatio = 0.10
    #从文件中读取数据，并且转化成归一化特征值
    datingDataMat,datingLabels = fileToMatrix('datingTestSet.txt')
    normMat,ranges,minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]
    #计算测试向量的数量
    numTestVecs = int(m*hoRatio)
    #分类错误计数
    errorCount = 0.0
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i,:],normMat[numTestVecs:m,:],datingLabels[numTestVecs:m],3)
        print("分类结果：%d,\t真实结果：%d" % (classifierResult,datingLabels[i]))
        if(classifierResult != datingLabels[i]):
            errorCount += 1.0

    #print(errorCount)
    #print(numTestVecs)
    print("错误率：%f%%" % (errorCount/float(numTestVecs)*100))

def classifyPerson():
    #结果集列表，三种可能的预测结果
    resultList = ['不喜欢','有一点兴趣','非常喜欢']
    #读入测试数据
    percentPlayVideoGame = float(input("玩视频游戏所耗费时间的百分比："))
    ffMiles = float(input("每年获得的飞行常客里程数："))
    iceCream = float(input("每周消费的冰淇淋公斤数："))
    #读入数据，计算样本特征举证和标签向量
    filename = "datingTestSet.txt"
    datingDataMat,datingLabels = fileToMatrix(filename)
    #训练集归一化
    normMat,ranges,minVals = autoNorm(datingDataMat)
    #生成NumPy数据，测试集
    inArr = np.array([percentPlayVideoGame,ffMiles,iceCream])
    # 测试集归一化
    norminArr = (inArr - minVals)/ranges
    #返回分类结果
    classifierResult = classify0(norminArr,normMat,datingLabels,3)
    #输出结果
    print("海伦可能",resultList[classifierResult],"这个人")

if __name__ =="__main__":
    classifyPerson()

'''
if __name__=='__main__':
    filename = "datingTestSet.txt"
    datingDataSet,datingLabels = fileToMatrix(filename)
    normDataSet,ranges,minVals = autoNorm(datingDataSet)
    print(normDataSet)
    print(ranges)
    print(minVals)
'''

'''
if __name__=='__main__':
    filename = "datingTestSet.txt"
    datingDataSet,datingLabels = fileToMatrix(filename)
    showDatas(datingDataSet,datingLabels)

'''

'''
if __name__=='__main__':
    #打开的文件名
    filename = "datingTestSet.txt"
    #计算得到训练样本矩阵和类标签向量
    datingDataMat,datingLabel = fileToMatrix(filename)
    print(datingDataMat)
    print(datingLabel)
    
'''