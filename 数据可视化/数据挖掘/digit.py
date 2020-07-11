# -*- coding:utf-8 -*-
# -*- author：zzZ_CMing
# -*- 2018/01/24；9:09
# -*- python3.5

import numpy as np
from sklearn.datasets import load_digits
from sklearn.preprocessing import LabelBinarizer #标签二值化
from sklearn.model_selection import train_test_split   #切割数据,交叉验证法

def sigmoid(x):
    return 1/(1+np.exp(-x))

def dsigmoid(x):
    return x*(1-x)

class NeuralNetwork:
    def __init__(self,layers):#(64,100,10)
        #权重的初始化,范围-1到1：+1的一列是偏置值
        self.V = np.random.random((layers[0] + 1, layers[1]+1))*2 - 1
        self.W = np.random.random((layers[1] + 1, layers[2])) * 2 - 1

    def train(self,X,y,lr=0.11,epochs=10000):
        #添加偏置值：最后一列全是1
        temp = np.ones([X.shape[0],X.shape[1]+1])
        temp[:,0:-1] = X
        X = temp

        for n in range(epochs+1):
            #在训练集中随机选取一行(一个数据)：randint()在范围内随机生成一个int类型
            i = np.random.randint(X.shape[0])
            x = [X[i]]
            #转为二维数据：由一维一行转为二维一行
            x = np.atleast_2d(x)

            # L1：输入层传递给隐藏层的值；输入层64个节点，隐藏层100个节点
            # L2：隐藏层传递到输出层的值；输出层10个节点
            L1 = sigmoid(np.dot(x, self.V))
            L2 = sigmoid(np.dot(L1, self.W))

            # L2_delta：输出层对隐藏层的误差改变量
            # L1_delta：隐藏层对输入层的误差改变量
            L2_delta = (y[i] - L2) * dsigmoid(L2)
            L1_delta = L2_delta.dot(self.W.T) * dsigmoid(L1)

            # 计算改变后的新权重
            self.W += lr * L1.T.dot(L2_delta)
            self.V += lr * x.T.dot(L1_delta)

            #每训练1000次输出一次准确率
            if n%1000 == 0:
                predictions = []
                for j in range(X_test.shape[0]):
                    #获取预测结果：返回与十个标签值逼近的距离，数值最大的选为本次的预测值
                    o = self.predict(X_test[j])
                    #将最大的数值所对应的标签返回
                    predictions.append(np.argmax(o))
                #np.equal()：相同返回true，不同返回false
                accuracy = np.mean(np.equal(predictions,y_test))
                print('迭代次数：',n,'准确率：',accuracy)

    def predict(self,x):
        # 添加偏置值：最后一列全是1
        temp = np.ones([x.shape[0] + 1])
        temp[0:-1] = x
        x = temp
        # 转为二维数据：由一维一行转为二维一行
        x = np.atleast_2d(x)

        # L1：输入层传递给隐藏层的值；输入层64个节点，隐藏层100个节点
        # L2：隐藏层传递到输出层的值；输出层10个节点
        L1 = sigmoid(np.dot(x, self.V))
        L2 = sigmoid(np.dot(L1, self.W))
        return L2

#载入数据:8*8的数据集
digits = load_digits()
X = digits.data
Y = digits.target
#输入数据归一化：当数据集数值过大，乘以较小的权重后还是很大的数，代入sigmoid激活函数就趋近于1，不利于学习
X -= X.min()
X /= X.max()

NN = NeuralNetwork([64,100,10])
#sklearn切分数据
X_train,X_test,y_train,y_test = train_test_split(X,Y)
#标签二值化：将原始标签(十进制)转为新标签(二进制)
labels_train = LabelBinarizer().fit_transform(y_train)
labels_test = LabelBinarizer().fit_transform(y_test)

print('开始训练')
NN.train(X_train,labels_train,epochs=20000)
print('训练结束')
