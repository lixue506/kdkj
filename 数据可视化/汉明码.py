# 2020-07-01
import numpy as np
import pandas as pd

# find n
def Find_N(s):
    l = len(s)
    for k in range(l):
        if pow(2,k) >= l + 1:
            return k,l-k
    return -1,-1

def func(l, stay, cc, start, r):
    result = r
    if start < stay:
        result += pow(2, start)
        start += 1
        func(l, stay, cc, start, result)
        func(l, stay, cc, start, r)

    if start == stay:
        result += pow(2, start)
        start += 1
        if result not in cc and result <= l:
            cc.append(result)
        func(l, stay, cc, start, result) 

    if start > stay:
        result += pow(2, start) 
        start += 1
        if result not in cc and result <= l:
            cc.append(result)
        else:
            return
        func(l, stay, cc, start, result)
        func(l, stay, cc, start, r) 

# 计算检验位数
def Check(k,l):
    c = []
    for i in range(k):
        cc = []
        func(l, i, cc, 0, 0)
        c.append(cc)
    return c
# 判断输入字符串是否为二进制
def Yanzheng(s):
    for i in list(s):
        if i != '1' and i != '0':
            return False
    return True

def Save_Check(s):
    if not Yanzheng(s):
        print("输入字符有误！")

    k,n = Find_N(s)

    if k==-1:
        print("输入格式长有误！")
        return False
    
    c = Check(k,k+n)
    a = []

    # 计算异或
    for i in c:
        t = int(s[i[0]-1])
        for j in i[1:]:
            t = t ^ int(s[j-1])
        # print(t)
        a.append(t)

    # 纠正有误位
    le = len(a)
    tmp = 0
    for i in range(le):
        tmp += a[le-i-1] * pow(2, le-i-1)
        
    print("纠错位：",tmp)
    s = list(s)
    s[tmp-1] = str(abs(1-int(s[tmp-1])))
    s = ''.join(s)
    print("正确传输：",s)

if __name__ == "__main__":
    while True:
        s = input("请输入一段二进制编码:\n")
        if s == 'q' or s == 'Q':
            break
        Save_Check(s)
        break