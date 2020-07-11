# 2020-06-26
import math
import numpy as np
import pandas as pd
# （1）随机选取k个聚类中心（一般在样本集中选取，也可以自己随机选取）;

# （2）计算每个样本与k个聚类中心的距离，并将样本归到距离最小的那个类中;

# （3）更新中心，计算属于k类的样本的均值作为新的中心。

# （4）反复迭代（2）（3）,直到聚类中心不发生变化，
#     后者中心位置误差在阈值范围内，或者达到一定的迭代次数
data = [1,5,10,9,26,32,16,21,14]
cent = [1, 5, 10]
distance = np.array([[np.inf for i in range(10)]]*3)
lastnear = np.zeros(10)
n = 0
# 欧氏距离
def Eular(x, y):
    result = np.sqrt((data[i]-cent[j])**2)
    return result

while True:
    
    c = [ [], [], [] ]
    for i in range(len(data)):
        for j in range(len(cent)):
            distance[j,i] = Eular(data[i],cent[j])
        
        x = 1 if distance[0,i] > distance[1,i] else 0
        x = 2 if distance[x,i] > distance[2,i] else x 
        lastnear[n] += distance[x,i]**2
        c[x].append(data[i])
    # 判断条件是否跳出
    if n > 1 and lastnear[n]==lastnear[n-1]:
        break
      
    print("第 ", n+1, " 次循环：")
    print("   平均值点：", cent)
    print("   c1：", c[0])
    print("   c2：", c[1])
    print("   c3：", c[2])
    print("准则函数：", lastnear[n])
    # 更新中心点
    cent[0] = np.sum(c[0]) / (len(c[0]))
    cent[1] = np.sum(c[1]) / (len(c[1]))
    cent[2] = np.sum(c[2]) / (len(c[2]))
    # 判断条件是否跳出
    if n > 1 and lastnear[n]==lastnear[n-1]:
        break
    n += 1
    