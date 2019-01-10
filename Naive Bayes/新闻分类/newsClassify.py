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
import matplotlib as plt
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
        new_folder_path = os.path.join(folder_path,folder)      #生成新的文件夹
        files = os.listdir(new_folder_path)                     #从新文件夹下得到所有的文件名

        j = 1
        for file in files:
            if j>100:
                break
            with open(os.path.join(new_folder_path,file),'r',encoding = 'utf-8') as f :    #打开txt文件
                raw = f.read()

            word_cut = jieba.cut(raw,cut_all=False)         #调用jieba分词模块，True为全模式   False为精简模式
            word_list = list(word_cut)                      #列表化

            data_list.append(word_list)                     #将分词结果加入到列表当中
            class_list.append(folder)
            j += 1

    data_class_list = list(zip(data_list,class_list))       #zip压缩合并，将数据与类标签压缩合并
    random.shuffle(data_class_list)
    index = int(len(data_class_list) * test_size) + 1
    train_list = data_class_list[index:]
    test_list = data_class_list[:index]
    train_data_list,train_class_list = zip(*train_list)      #训练集解压缩
    test_data_list,test_class_list = zip(*test_list)

    all_words_dict = {}                                      #统计训练集的词频
    for word_list in train_data_list:
        for word in word_list:
            if word in all_words_dict.keys():
                all_words_dict[word] += 1
            else:
                all_words_dict[word] = 1

    all_words_tuple_list = sorted(all_words_dict.items(),key = lambda f:f[1],reverse=True)      #按照词频对单词进行降序排序
    all_words_list,all_words_nums = zip(*all_words_tuple_list)            #解压缩
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
    with open(words_file,'r',encoding='utf-8') as f:            #打开文件
        for line in f.readlines():
            word = line.strip()                                 #去除文档中的回车
            if len(word)>0:                                     #
                words_set.add(word)
    return words_set


def words_dict(all_words_list,deleteN,stopwprds_set = set()):
    features_word = []
    n = 1
    for t in range(deleteN,len(all_words_list),1):
        if n>1000:
            break
        if not all_words_list[t].isdigit() and all_words_list[t] not in stopwprds_set:
            features_word.append(all_words_list)
            n += 1
    return features_word

"""
函数说明:根据feature_words将文本向量化
Parameters:
	train_data_list - 训练集
	test_data_list - 测试集
	feature_words - 特征集
Returns:
	train_feature_list - 训练集向量化列表
	test_feature_list - 测试集向量化列表
"""
def TextFeatures(train_data_list, test_data_list, feature_words):
	def text_features(text, feature_words):						#出现在特征集中，则置1
		text_words = set(text)
		features = [1 if word in text_words else 0 for word in feature_words]
		return features
	train_feature_list = [text_features(text, feature_words) for text in train_data_list]
	test_feature_list = [text_features(text, feature_words) for text in test_data_list]
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
	classifier = MultinomialNB().fit(train_feature_list, train_class_list)
	test_accuracy = classifier.score(test_feature_list, test_class_list)
	return test_accuracy

if __name__ == '__main__':
	#文本预处理
	folder_path = './SogouC/Sample'				#训练集存放地址
	all_words_list, train_data_list, test_data_list, train_class_list, test_class_list = TextProcessing(folder_path, test_size=0.2)

	# 生成stopwords_set
	stopwords_file = './stopwords_cn.txt'
	stopwords_set = MakeWordsSet(stopwords_file)


	test_accuracy_list = []
	deleteNs = range(0, 1000, 20)				#0 20 40 60 ... 980
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

'''

if __name__=='__main__':
    folder_path = './SogouC/Sample'             #数据存放地址
    all_words_list, train_data_list, test_data_list, train_class_list, test_class_list = TextProcessing(folder_path,test_size = 0.2)

    #生成stopwords_set
    stopwords_file = './stopwords_cn.txt'
    stopwords_set = MakeWordsSet(stopwords_file)

    features_word = words_dict(all_words_list,100,stopwords_set)
    print(features_word)
'''