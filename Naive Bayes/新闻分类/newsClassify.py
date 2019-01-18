#!/usr/bin/env python
# encoding: utf-8
'''
@author: hehaiyang
@contact: 1272114024@qq.com
@file: newsClassify.py
@time: 2019/1/9 10:57
'''

import os
import jieba
import random
import matplotlib.pyplot as plt
from sklearn.naive_bayes import MultinomialNB
''''
函数说明：
    @:parameter 
        folder_path 文本的路径
        test_size 测试集的大小
    @:return
        all_words_list,     切分出来所有词汇的列表
        train_data_list,    训练数据集列表
        test_data_list,     测试数据集列表
        train_class_list,       训练类别列表
        test_class_list         测试类别列表

'''
def TextProcessing(folder_path,test_size = 0.2):
    folder_list = os.listdir(folder_path)           #找到文件夹下面的列表
    data_list = []
    class_list = []

    for folder in folder_list:
        new_folder_path = os.path.join(folder_path,folder)      #生成新的文件夹，os.path.join两个路径名进行拼接，形成新的路径名
        files = os.listdir(new_folder_path)                     #从新文件夹下得到所有的文件名

        j = 1
        for file in files:
            if j>100:
                break
            with open(os.path.join(new_folder_path,file),'r',encoding = 'utf-8') as f :    #打开txt文件
                raw = f.read()

            word_cut = jieba.cut(raw,cut_all=False)         #调用jieba.cut分词模块，raw是需要进行分词的字符串，True为全模式  False为精简模式
            word_list = list(word_cut)                      #列表化

            data_list.append(word_list)                     #将分词结果加入到列表当中
            class_list.append(folder)                       #存储上一级文件夹名称
            j += 1

    data_class_list = list(zip(data_list,class_list))       #zip压缩合并，将数据与类标签压缩合并，将对象当中的元素打包成一个个元组，返回这些元组组成的列表
    random.shuffle(data_class_list)                         #将序列的所有元素随机排序，保证结果的客观性
    index = int(len(data_class_list) * test_size) + 1       #划分训练集和测试集
    train_list = data_class_list[index:]
    test_list = data_class_list[:index]
    train_data_list,train_class_list = zip(*train_list)      #训练集解压缩        *将元组解压为列表
    test_data_list,test_class_list = zip(*test_list)

    all_words_dict = {}                                      #统计训练集的词频
    for word_list in train_data_list:
        for word in word_list:
            if word in all_words_dict.keys():
                all_words_dict[word] += 1
            else:
                all_words_dict[word] = 1                     #拉普拉斯平滑

    all_words_tuple_list = sorted(all_words_dict.items(),key = lambda f:f[1],reverse=True)      #按照词频对单词进行降序排序
    all_words_list,all_words_nums = zip(*all_words_tuple_list)            #字典解压缩为列表
    all_words_list = list(all_words_list)
    return all_words_list,train_data_list,test_data_list,train_class_list,test_class_list
'''
函数说明：读取文件里的内容，并去重
@:parameter    words_file
    文件名
@:return
    返回不含重复元素的集合 words_set
'''
def MakeWordsSet(words_file):
    words_set = set()                                           #创建set集合
    with open(words_file,'r',encoding='utf-8') as f:            #打开文件，指定打开文件的编码方式
        for line in f.readlines():
            word = line.strip()                                 #去除每行两边的空字符
            if len(word)>0:                                     #如果单词的长度大于0，则加入到set集合当中
                #集合的add方法，将传入元素作为一个整体添加到集合当中
                #集合的update方法，把要传入的元素进行拆分之后，作为每个个体传入到集合当中
                words_set.add(word)
    return words_set
"""
函数说明：文本特征提取
@:parameter
    all_words_list,     训练集的所有文本列表
    deleteN,    删除词频最高的若干个词
    stopwprds_set = set()   指定的结束语
@:return
    features_word   特征集
"""

def words_dict(all_words_list,deleteN,stopwprds_set=set()):
    features_word = []                                          #创建一个空的特征集合
    n = 1
    for t in range(deleteN,len(all_words_list),1):
        if n > 1000:
            break
            #自定义的划分词的规则，这个词不是数字，并且不是指定的结束语，单词长度大于1小于5，那么这个词就可以作为特征词
        if not all_words_list[t].isdigit() and all_words_list[t] not in stopwprds_set and 1 < len(all_words_list[t]) < 5:
            features_word.append(all_words_list[t])
        n += 1
    return features_word

"""
函数说明:根据feature_words将文本向量化,得到训练集和测试集的特征列表
Parameters:
	train_data_list - 训练集
	test_data_list - 测试集
	feature_words - 特征集
Returns:
	train_feature_list - 训练集向量化列表
	test_feature_list - 测试集向量化列表
"""
def TextFeatures(train_data_list, test_data_list, feature_words):
    # 出现在特征集中，则置1
    def text_features(text, feature_words):
        # set是一个无序且不重复的元素集合
        text_words = set(text)
        features = [1 if word in text_words else 0 for word in feature_words]
        return features
    train_feature_list = [text_features(text, feature_words) for text in train_data_list]
    test_feature_list = [text_features(text, feature_words) for text in test_data_list]
    # 返回结果
    return train_feature_list, test_feature_list

"""
函数说明:新闻分类器
Parameters:
	train_feature_list - 训练集向量化的特征文本
	test_feature_list - 测试集向量化的特征文本
	train_class_list - 训练集分类标签
	test_class_list - 测试集分类标签
Returns:
	test_accuracy - 分类器精度
"""
def TextClassifier(train_feature_list, test_feature_list, train_class_list, test_class_list):
    #MutinomialNB默认有三个参数，详细查看文档https://scikit-learn.org/dev/modules/generated/sklearn.naive_bayes.MultinomialNB.html
	classifier = MultinomialNB().fit(train_feature_list, train_class_list)
    #返回平均准确率
	test_accuracy = classifier.score(test_feature_list, test_class_list)
	return test_accuracy

if __name__ == '__main__':
	#文本预处理
    folder_path = './SogouC/Sample'				#训练集存放地址
    all_words_list, train_data_list, test_data_list, train_class_list, test_class_list = TextProcessing(folder_path, test_size=0.2)
    stopwords_file = './stopwords_cn.txt'                   # 生成stopwords_set
    stopwords_set = MakeWordsSet(stopwords_file)            #生成stopwords_wet集合
    test_accuracy_list = []
    #通过下面的deleteNs进行绘图观察，当删除词在前450个的时候，准确率保持在50%以上，这边就将deleteN固定，测试分类的准确率
    feature_words = words_dict(all_words_list,450,stopwords_set)
    train_feature_list,test_feature_list_ = TextFeatures(train_data_list,test_data_list,feature_words)
    test_accuracy = TextClassifier(train_feature_list,test_feature_list_,train_class_list,test_class_list)
    test_accuracy_list.append(test_accuracy)
    ave_accuracy = sum(test_accuracy_list)/len(test_accuracy_list)
    print('删去前450个高频词分类的精确率为：%.5f' % ave_accuracy)


	#deleteNs = range(0, 1000, 20)				#0 20 40 60 ... 980
"""
	for deleteN in deleteNs:
		feature_words = words_dict(all_words_list, deleteN, stopwords_set)
		train_feature_list, test_feature_list = TextFeatures(train_data_list, test_data_list, feature_words)
		test_accuracy = TextClassifier(train_feature_list, test_feature_list, train_class_list, test_class_list)
		test_accuracy_list.append(test_accuracy)
	# ave = lambda c: sum(c) / len(c)
	# print(ave(test_accuracy_list))

	plt.figure()
	plt.plot(deleteNs, test_accuracy_list)
	plt.title('Relationship of deleteNs and test_accuracy')
	plt.xlabel('deleteNs')
	plt.ylabel('test_accuracy')
	plt.show()
"""