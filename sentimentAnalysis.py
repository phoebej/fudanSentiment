#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__title__ = ''
__author__ = 'phoebe'
__mtime__ = '2019/7/19'
# code is far away from bugs with the god animal protecting
    I love animals. They taste delicious.
"""
import collections
import re
import string
import time

import matplotlib as plt
import numpy as np
import pandas as pd

from scipy import sparse
from scipy.sparse import coo_matrix
from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from sklearn.metrics import accuracy_score

url = "train.csv"
urlt = 'test.csv'
# tmp=pd.read_csv(url,sep='\t')


def readData(url):
    """

    :param url: string
    :return: list[string],list[int]
    """
    file = open(url, 'r')
    # tmp = np.loadtxt(file,dtype=str, delimiter='\t',skiprows=42892)
    # print(tmp)
    data = []
    for l in file.readlines():
        c = l.strip().split('\t')
        data.append(c)
    tmp = np.array(data)
    n, m = tmp.shape
    column = tmp[0, :]
    X = tmp[1:, :m - 1]
    Y = tmp[1:, -1].astype(int)
    file.close()
    return X, Y


def readData1(url):
    """

    :param url: string
    :return: list[string],list[int]
    """
    file = open(url, 'r')
    tmp = np.loadtxt(file, dtype=str, delimiter='\t')
    X = tmp[1:, :]
    file.close()
    return X
# s.translate(table, string.punctuation)
# map(int,X[:,:2])
# def buildDict(X):


def createCount1(X):
    word = []
    # url1 = 'stopwords.txt'
    # file = open(url1, 'r')
    # stopwords = []
    # for l in file.readlines():
    #     stopwords.append(l.strip())
    # file.close()
    punctuation = list(string.punctuation)
    for line in X[:, -1]:
        s = ""
        l = line.strip().split()
        for w in l:
            w = re.sub('[^a-zA-Z]', '', w)
            # print(w)
            for p in punctuation:
                w = w.replace(p, '')
            # if w not in stopwords:
            # w = w.lower()
            s = s + ' ' + w
            # print(s)
        word.append(s)
    print("corpus:{}".format(word))
    print("length of word:{}".format(len(word)))
    vec = CountVectorizer(
        stop_words='english',
        max_features=1000,
        ngram_range=(
            1,
            2))
    w = vec.fit_transform(word)
    print("vector:{}".format(w.toarray()))
    print("vector's shape:{}".format(w.toarray().shape))
    return w

def createCount(X):
    url1 = 'stopwords.txt'
    file = open(url1, 'r')
    stopwords = []
    for l in file.readlines():
        stopwords.append(l.strip())
    file.close()
    punctuation = [",", ":", ";", ".", "'", '"', "@"
                   "’", "?", "/", "-", "+", "&", "(", ")"]
    word = []
    print('start removing stopwords,lowering and removing punctuation')
    # 变为小写，去掉非英文字母字符，去掉停用词，去掉标点符号
    start1 = time.time()
    for w in X[:, -1]:
        wordDict = []
        line = w.strip().split()
        for l in line:
            l = re.sub('[^a-zA-Z]', '', l)
            for punc in punctuation:
                l = l.replace(punc, "")
            if l not in stopwords:
                l = l.lower()
                wordDict.append(l)
        word.append(wordDict)
    end1 = time.time()
    print('Finish removing stopwords, uses {}s'.format(end1 - start1))

    print('building vocabulary dictionary!')
    wordDict = []
    for w in word:
        for line in w:
            wordDict.append(line.strip())
    # wordDict=list(set(wordDict))
    w = dict(collections.Counter(wordDict).most_common(len(wordDict) - 1))
    print(len(w))
    unique_tokens = []
    single_tokens = []
    for words, values in w.items():
        if values == 1:
            single_tokens.append(words)
        else:
            unique_tokens.append(words)
    print(len(single_tokens))
    # counts = pd.DataFrame(0, index=np.arange(len(word)), columns=unique_tokens)
    # counts = np.zeros((len(word), len(unique_tokens)),dtype='float16')
    counts = coo_matrix((len(word), len(unique_tokens)),
                        dtype=np.int8).toarray()
    print("the shape of count:{}".format(counts.shape))

    print("start vectorization")
    s2 = time.time()
    for i, item in enumerate(word):
        for t in item:
            if t in unique_tokens:
                counts[i][unique_tokens.index(t)] += 1
    e2 = time.time()
    # word_counts=counts.sum(axis=0)
    # counts=counts[:,(word_counts>=5)&(word_counts<=100)]
    print("finished vectorization, using {}s".format(e2 - s2))
    c = sparse.csr_matrix(counts)
    print('type of counts:{}'.format(type(counts)))
    return c

def main():
    #===================读入训练集和测试集，并用bow模型结合n-gram向量化===============
    X_tmp, Y = readData(url)
    # c = createCount1(X)
    n, m = X_tmp.shape
    x_test = readData1(urlt)
    data = np.concatenate((X_tmp, x_test), axis=0)
    # # print(data.shape)
    vectorData = createCount1(data)
    train = vectorData[:n, :]
    test = vectorData[n:, :]
    # #
    X_train, X_cv, y_train, y_cv = train_test_split(train, Y, test_size=0.2, random_state=1)
    #---------------------寻找最优参数------------------------------------------
    params = {'C':[0.0001,1,100,1000],
              'max_iter':[200,800,1000],
              'multi_class':['multinomial'],
              'class_weight':['balanced',None],
              'solver':['sag','lbfgs','newton-cg']}
    lr = LogisticRegression()
    clf = GridSearchCV(lr,param_grid=params,cv=10)
    clf.fit(X_train,y_train)
    print("best params:{}".format(clf.best_params_))
    #===============构建最优lr模型，这里注意方法========================
    classifier = LogisticRegression(**clf.best_params_)
    classifier.fit(X_train,y_train)
    #===============用CV集测试=======================================
    cv_pred = classifier.predict(X_cv)
    cv_score = classifier.predict_proba(X_cv)[:,1]
    acc = accuracy_score(y_cv,cv_pred)
    print("accuracy of cv:{}".format(acc))

    predictions = classifier.predict(test)
    predictions = predictions.reshape(len(predictions), 1)
    x_test = x_test[:, 0].reshape(len(x_test[:, 0]), 1).astype(int)
    # print(x_test.shape)
    # print(predictions.shape)
    result = np.concatenate((x_test, predictions), axis=1)
    df = pd.DataFrame(result, columns=['PhraseId', 'Sentiment'])
    df.to_csv('result2.csv')

    # fpr, tpr, threshold = metrics.roc_curve(y_cv, cv_score)
    # roc_auc = metrics.auc(fpr, tpr)
    # f, ax = plt.subplots(figsize=(8, 6))
    # plt.stackplot(fpr, tpr, color='#338DFF', alpha=0.5, edgecolor='black')
    # plt.plot(fpr, tpr, color='black', lw=1.5)
    # plt.plot([0, 1], [0, 1], color='red', linestyle='--', lw=2)
    # plt.text(0.5, 0.3, 'ROC curve (area = %0.2f)' % roc_auc)
    # plt.xlabel('1-Specificity')
    # plt.ylabel('Sensitivity')
    # plt.show()

main()




    # clf1 = LogisticRegression(
    #     solver='sag',
    #     multi_class='multinomial',
    #     max_iter=400, n_jobs=-1)
    #
    # print("shape of X_train:{}".format(X_train.shape))
    # print(len(Y))
    #
    # clf1.fit(X_train, Y)
    # predictions = clf1.predict(X_test)
    # predictions = predictions.reshape(len(predictions), 1)
    # x_test = x_test[:, 0].reshape(len(x_test[:, 0]), 1).astype(int)
    # print(x_test.shape)
    # print(predictions.shape)
    # result = np.concatenate((x_test, predictions), axis=1)
    # df = pd.DataFrame(result, columns=['PhraseId', 'Sentiment'])
    # df.to_csv('result1.csv')


    # mse = sum((y_test - predictions) ** 2) / len(predictions)
    # print(mse)
    # scores1=cross_val_score(clf1,counts,Y,cv=10, scoring="accuracy",n_jobs=-1,pre_dispatch='1*n_jobs')
    # print(scores1)
    #
    # vectorizer = CountVectorizer()
    # vectorizer.fit(word)
    # print(vectorizer.vocabulary_)
    # print(countdict)

    # string = re.sub("[\s+\.\!\/_,$%^*(+\"\')]+|[+——()?【】“”！，。？、~@#￥%……&*（）]+', "",line)
    # print(wordDict)
    # count = dict(collections.Counter(word))
    # print(count)
