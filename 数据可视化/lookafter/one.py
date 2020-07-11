from scipy.special import gamma
import matplotlib.pyplot as plt
import numpy as np
import math

N = 1000
T = 50   #
h = 0.05 # 步长
alpha = 0.82
D = [1, 1]
A = np.array([[2,-0.11],[-5,2.2]])
B = np.array([[-1.6, -0.238],[-0.18,-2.4]])
I = np.array([0, 0])
Alpha = np.array([[0.02, 0.26],[0, 0.8]])
Beita = np.array([[0.02, 0.01], [0, 0.9]])
Tt = np.array([[0.1, 0],[0, 0.1]])
Ss = np.array([[0.1, 0],[0, 0.1]])
C = np.array([[0.1, 0],[0.1, 0]])
V = np.array([1, 1])
# m = math.ceil(alpha)
# driver system

vn = yn = ynp = [[0] * (N+10), [0] * (N+10)]

def initial_function1(x):
    result = math.cos(x)
    return result

def initial_function2(x):
    result = math.sin(x)
    return result

fai = [initial_function1,initial_function2]

vn[0][0] = yn[0][0] = fai[0](0)
vn[1][0] = yn[1][0] = fai[1](0)


def bjn(n, j):
    result = (pow(h, alpha) / alpha) * (pow(n-j+1, alpha) - pow(n-j, alpha))
    print( pow(n-j+1, alpha))
    print(n-j+1)
    print(n)
    return result

def ajn(n, j):
    result = pow(h, alpha) / (alpha * (alpha + 1))  
    if j == 0:
        t = pow(n,alpha+1) - ((n - alpha) * pow(n+1, alpha))
        return t * result
    elif j == n+1:
        return result
    else:
        t = pow(n-j+2, alpha + 1) + pow(n-j, alpha + 1) - 2 * pow(n-j+1, alpha + 1)
        return result * t

def Ynp(n, i):
    Tn = (n+1) * h
    K = 1
    result = 0
    m = int(Tao(Tn) / h)
    print(Tao(Tn))
    for kk in range(m):
        if kk > 0:
            K = K * kk
        result += fai[i]( kk*h ) * pow(Tn, kk) / K
    t = 0
    for j in range(n+1): 
        tj = j * h
        yj = vj = []
        yj.append(ynp[0][j])
        yj.append(ynp[1][j])
        vj.append(vn[0][j])
        vj.append(vn[1][j])
        t += bjn(n, j) * f(tj, yj, vj, i)
    result += t / gamma(alpha)
    ynp[i][n+1] = result
    return result

# 模拟y(t-tao(t))
def Vn(n, i):
    t = (n+1) * h
    #  Membership [0, 1)
    mm = int(Tao(t) / h)
    deerta = Tao(t) / h - mm  # （x-x0）/(x1-x0)=(y-y0)/(y1-y0)=k
    result = 0
    if mm  > 1:
        result = deerta * yn[i][n-mm+2] + (1 - deerta) * yn[i][n-mm+1]
    else:
        result = deerta * ynp[i][n+1] + (1 - deerta) * yn[i][n]

    vn[i][n+1] = result

    return result

# 时延函数，   τ function >=0
def Tao(t):
    result = 1.38 * abs(math.sin(t))
    return result 
# 激活函数
def Activation(x):
    return 0.5*(abs(x+1)-abs(x-1))
    #  Avoid overflow
    # if x>=0:   
    #     return 1.0/(1+np.exp(-x))
    # else:
    #     return np.exp(x)/(1+np.exp(x))
# 卡普诺函数
def f(t, y, v, i):
    if t<=0:
        return yn[i][0]
    result = I[i] - D[i] * y[i]
    # print(result)
    temp = 0
    TV = float("inf") # 交集取最小
    AlphaF = float("inf") # 交集取最小
    SV = float("-inf") # 并集取最大
    BeitaF = float("-inf") # 并集取最大
    for j in range(2):
        temp += A[i][j] * Activation( y[j] )
        temp += C[i][j] * V[j]
        a = Activation(v[j]) 
        temp += B[i][j] * a
        TV = min(TV, Tt[i][j] * V[j])
        AlphaF = min(AlphaF, Alpha[i][j] * a)
        SV = max(SV, Ss[i][j] * V[j])
        BeitaF = max(BeitaF, Beita[i][j] * a)
        result += temp + TV + AlphaF + SV + BeitaF

    return result

def Yn(n, l):
    Ynp(n, l)
    Vn(n,l)

    Tn = (n+1) * h
    K = 1
    result = 0
    m = int(Tao(Tn) / h)
    for kk in range(m):
        if kk > 0:
            K = K * kk
        result += fai[l]( kk*h ) * pow(Tn, kk) / K

    y = v = []
    y.append(ynp[0][n+1])
    y.append(ynp[1][n+1])
    v.append(vn[0][n+1])
    v.append(vn[1][n+1])

    result += pow(h, alpha) / gamma(alpha+2) * f(Tn, y, v, l)
    t = 0

    for j in range(n+1):
        tj = j * h
        yj = vj = []
        yj.append(ynp[0][j])
        yj.append(ynp[1][j])
        vj.append(vn[0][j])
        vj.append(vn[1][j])
        t += ajn(n, j) * f(tj, yj, vj, l)

    result += t / gamma(alpha)

    yn[l][n+1] = result

def Draw():
    for n in range(0,N+1):
        for l in range(2):
            Yn(n,l)

    t = [n*h for n in range(0,N)]
    # 轨迹图
    plt.title('alpha = %s' % alpha)
    plt.xlabel('t')
    plt.ylabel('x')
    plt.plot(t, yn[0][0:N], label="$x1$", color='green', ls='-.')
    plt.plot(t, yn[1][0:N], label="$x2$", color='red', ls=':') 
    plt.show()
    # 相图
    plt.title('alpha = %s' % alpha)
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.plot(yn[0][0:N], yn[1][0:N]) 
    plt.show()

if __name__ == "__main__":
    t = bjn(1,0)
    print(t)
    # Draw()
