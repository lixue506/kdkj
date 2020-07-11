from sklearn.model_selection import train_test_split
from sklearn import svm
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import numpy as np 

test = pd.read_excel('iris.data.xlsx','Sheet1',index_col=0)
test.to_csv('test.csv',encoding='utf-8')

data_Set = []
data_Set_x = []
data_Set_y = []

#打开数据集,字符串前加r表示raw string,防止路径字符串中存在的反斜杠带来的转义
data_file = open(r"./test.csv")

#拆分数据集，取前四列为x，第五列为y
for line in data_file.readlines():
    lineArr = line.strip().split(',')
    data_Set.append(lineArr)
    data_Set_x.append(lineArr[0:4])
    data_Set_y.append(lineArr[4])

#按照7:3的比例分割训练集和测试集
data_train_x,data_test_x = train_test_split(data_Set_x,test_size = 0.3,random_state = 55)
data_train_y,data_test_y = train_test_split(data_Set_y,test_size = 0.3,random_state = 55)

"""
分别利用四种核函数进行训练，这些核函数都可以设置参数，例如
decision_function_shape='ovr'时，为one v rest，即一个类别与其他类别进行划分，
decision_function_shape='ovo'时，为one v one，即将类别两两之间进行划分，用二分类的方法模拟多分类的结果。
不设置的话会使用默认参数设置
"""
#linear线性核函数
clf1 = svm.SVC(C=1,kernel='linear', decision_function_shape='ovr').fit(data_train_x,data_train_y)
#使用rbf径向基核函数
clf2 = svm.SVC(C=1, kernel='rbf', gamma=1).fit(data_train_x,data_train_y)
#使用poly多项式核函数
clf3 = svm.SVC(kernel='poly').fit(data_train_x,data_train_y)
#使用sigmoid神经元激活核函数
clf4 = svm.SVC(kernel='sigmoid').fit(data_train_x,data_train_y)

#打印使用不同核函数进行分类时，训练集和测试集分类的准确率
print("linear线性核函数-训练集：",clf1.score(data_train_x, data_train_y))
print("linear线性核函数-测试集：",clf1.score(data_test_x, data_test_y))
print("rbf径向基核函数-训练集：",clf2.score(data_train_x, data_train_y))
print("rbf径向基函数-测试集：",clf2.score(data_test_x, data_test_y))
print("poly多项式核函数-训练集：",clf3.score(data_train_x, data_train_y))
print("poly多项式核函数-测试集：",clf3.score(data_test_x, data_test_y))
print("sigmoid神经元激活核函数-训练集：",clf4.score(data_train_x, data_train_y))
print("sigmoid神经元激活核函数-测试集：",clf4.score(data_test_x, data_test_y))

#使用decision_function()可以查看决策函数
print(clf1.decision_function(data_train_x))
#使用predict()可以查看预测结果
print(clf1.predict(data_train_x))

