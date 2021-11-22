'''
Author: XiangxinZhong
Date: 2021-11-21 20:12:57
LastEditors: XiangxinZhong
LastEditTime: 2021-11-22 23:02:22
Description: ML Lab3 决策树分类
'''

import numpy as np
import pandas as pd
from pandas.core.indexing import convert_from_missing_indexer_tuple
from pandas.plotting import andrews_curves
from sklearn.model_selection import train_test_split
import sklearn.metrics as mtr
from sklearn.datasets import load_iris, load_wine
from pydotplus.graphviz import graph_from_dot_data
from sklearn.tree import export_graphviz
from draw_tree import createPlot
import seaborn as sns
import matplotlib.pyplot as plt


class Node:
    '''
    description: 节点
    param {*} self 
    param {*} val 内部节点：特征 叶节点：类别
    param {*} tag 特征划分点
    return {*}
    '''

    def __init__(self, val=0.0, tag=None):
        self.val = val
        self.tag = tag
        # 左右子树
        self.lc = None
        self.rc = None
        
    def __str__(self):
        return f'val: {self.val}, tag: {self.tag}'

class Clf:
    '''
    description: 初始化
    param {*} self
    param {*} features 特征名列表
    return {*}
    '''

    def __init__(self, features=None):
        self.tree = None
        self.features = features
        self.dictTree = {} # 保存成字典

    '''
    description:  计算基尼系数
    param {*} self
    param {*} label 类别
    return {*}
    '''

    def Gini(self, label):
        gini = 0
        for (ck, cnt) in zip(*np.unique(label, return_counts=True)):
            prob_ck = cnt / len(label)
            gini += prob_ck * (1 - prob_ck)
        return gini

    '''
    description: 挑选最优划分
    param {*} self
    param {*} col 一个特征的所有值
    param {*} label 类别
    return {*}
    '''

    def get_best_split(self, col, label):
        # 连续值需要排序
        sort_col = np.unique(np.sort(col, axis=0))
        # 计算划分点
        pos = (sort_col[1:] + sort_col[:-1])/2

        tmp_best_gini = float('inf')  # 临时最优基尼系数
        tmp_best_split = 0  # 临时最优划分点
        for spot in pos:
            smaller_col = col < spot
            bigger_col = col > spot
            # 计算此划分的基尼系数
            gini = (sum(smaller_col)*self.Gini(label[smaller_col])+sum(
                bigger_col)*self.Gini(label[bigger_col]))/len(label)
            if gini < tmp_best_gini:
                tmp_best_gini = gini
                tmp_best_split = spot
        return tmp_best_gini, tmp_best_split

        '''
    description: 建树
    param {*} self
    param {*} features 特征矩阵
    param {*} labels 标签
    return {*}
    '''

    def buildTree(self, data, labels):
        kinds, cnts = np.unique(labels, return_counts=True)  # 类和每类的数量
        # 若样本全部属于同一类别，节点标记为该类
        if len(kinds) == 1:
            return Node(kinds[0])
        if data.shape[0] == 0:
            return None
        self.features = list(data.columns)

        best_gini = float('inf')  # 最优基尼系数
        best_split = None  # 最优划分
        best_val = 0  # 最优划分点
        best_feature = None

        # 遍历每个特征的每个值
        for i in range(data.shape[1]):
            gini, split = self.get_best_split(data.iloc[:, i], labels)
            if gini < best_gini:
                best_gini = gini
                best_split = split
                best_val = i
                best_feature = data.columns[i]

        if best_gini < 1e-3:
            return Node(kinds[cnts.argmax(0)])  # 返回最多的那个类

        # 初始化根节点
        tree = Node(self.features[best_val], best_split)
        ss = [str(best_feature),str(best_val)]
        # ss = "-".join(ss)
        # print("ss: {}".format(ss))
        # dictTree = {ss:{}}
        # print("dictTree: {}".format(dictTree))
        # 连续值二分左右子树 递归建树
        left = data.iloc[:, best_val] < best_split
        right = data.iloc[:, best_val] > best_split
        tree.lc= self.buildTree(data[left], labels[left])
        tree.rc= self.buildTree(data[right], labels[right])
        return tree

    '''
    description: 目的是为了存储tree
    param {*} self
    param {*} x
    param {*} y
    return {*}
    '''    
    def fit(self, x, y):
        print("Start to train...")
        self.tree= self.buildTree(x,y)
        print("End of training...")
        

    '''
    description: 前序遍历决策树
    param {*} self
    return {*}
    '''    
    def preOrder(self, tree):
        if tree == None:
            return
        print(tree)
        self.preOrder(tree.lc)
        self.preOrder(tree.rc)

    '''
    description: 判断某个样本的类别
    param {*} self
    param {*} x
    return {*}
    '''    
    def get_label(self, x):
        root = self.tree
        tag = root.tag
        while tag is not None:
            idx = self.features.index(root.val)
            if x[idx] < root.tag:
                root = root.lc
            else:
                root = root.rc
            tag = root.tag
        return root.val


    '''
    description: 分类
    param {*} self
    param {*} x
    param {*} y
    return {*}
    '''    
    def pred(self, x, y):
        if self.tree == None:
            return
        y_pred = []
        for ss in x:
            y_pred.append(self.get_label(ss))
        y_pred = np.array(y_pred)
        if y is not None:
            scores = np.count_nonzero(y == y_pred) / len(y)
        return y_pred, scores

    '''
    description: 树结构->字典，方便可视化
    '''
    def get_dict_tree(self, tree):
        tree = tree
        # if tree != None:
        ss = "-".join([str(tree.val),str(tree.tag)])
        dictTree = {ss:{}}
        # print(ss)
        if tree.lc == None and tree.rc == None:
            return tree.val
        dictTree[ss]['小于'] = self.get_dict_tree(tree.lc)
        dictTree[ss]['不小于'] = self.get_dict_tree(tree.rc)
        return dictTree

def confusion(y_true, y_pred):
    sns.set()
    f, ax = plt.subplots()
    # y1 = np.argmax(y_true, axis=0).flatten()  # y_true
    # y2 = np.argmax(y_pred, axis=0).flatten()  # y_pred
    C2 = mtr.confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    sns.heatmap(C2, annot=True, ax=ax)  # 画热力图
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("predict")
    ax.set_ylabel("true")
    plt.show()

'''
description: 主函数
'''
if __name__ == "__main__":
    data = load_wine()
    iris = pd.read_excel("iris_data.xlsx")
    wine = pd.read_excel("winequality_data.xlsx")
    wine = wine.iloc[:150, :]
    x, y = data['data'], data['target']
    # 划分数据集
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    x_train = pd.DataFrame(x_train, columns=data.feature_names, index=None)
    clf = Clf()
    clf.fit(x_train, y_train)
    # 前序遍历树
    clf.preOrder(clf.tree)
    y_pred, scores = clf.pred(x_test, y_test)
    print("Accuracy: {}".format(scores))
    print("真实分类: {}".format(y_test))
    print("预测分类: {}".format(y_pred))
    dicTree = clf.get_dict_tree(clf.tree)
    print(dicTree)
    sns.pairplot(wine, hue='quality label', height=3, diag_kind='kde')
    plt.show()
    createPlot(dicTree)
    confusion(y_test, y_pred)
    

