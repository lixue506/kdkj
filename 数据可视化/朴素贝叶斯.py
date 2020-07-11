import pandas as pd
import csv

Y = {'15':0,'1':{},'2':{},'3':{},'4':{},'5':{},'6':{},'7':{},'8':{},'9':{},'10':{},'11':{},'12':{},'13':{},'14':{}}
N = {'15':0,'1':{},'2':{},'3':{},'4':{},'5':{},'6':{},'7':{},'8':{},'9':{},'10':{},'11':{},'12':{},'13':{},'14':{}}

data = pd.read_excel('adult.data.xlsx','Sheet1',index_col=0)
data.to_csv('data.csv',encoding='utf-8')
test = pd.read_excel('test.adult.xlsx','Sheet1',index_col=0)
test.to_csv('test.csv',encoding='utf-8')

# 训练
with open('data.csv', encoding='utf-8') as f:
    f_csv = csv.DictReader(f)
    opq = 1
    for row in f_csv:
        print(opq)
        opq += 1
        if row['15'] == '<=50K':
            N['15'] += 1
            for i in range(1,15):
                if i==1 or i==3 or i==5 or i==11 or i==12 or i==13:
                    continue
                t = str(row[str(i)]) # 表示带统计数据的实值
                if t in N[str(i)].keys():
                    N[str(i)][t] += 1
                else:
                     N[str(i)][t] = 1
        else:
            Y['15'] += 1
            for i in range(1,15):
                if i==1 or i==3 or i==5 or i==11 or i==12 or i==13:
                    continue
                t = row[str(i)] # 表示带统计数据的实值
                if t in Y[str(i)].keys():
                    Y[str(i)][t] += 1
                else:
                    Y[str(i)][t] = 1

# 预测
with open('test.csv', encoding='utf-8') as f:
    f_csv = csv.DictReader(f)
    for row in f_csv:
            result1 = 0.7607
            result2 = 0.2393
            for i in range(1,15):
                if i==1 or i==3 or i==5 or i==11 or i==12 or i==13:
                    continue
                t = str(row[str(i)])
                result1 *= N[str(i)][t] / N['15']
                result2 *= Y[str(i)][t] / Y['15']
            if result1 > result2:
                print("预测：不超过50K")
            else:
                print("预测：超过50K")
            
    
