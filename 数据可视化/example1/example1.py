from scipy.special import gamma
import matplotlib.pyplot as plt
import numpy as np
import math

'''
This class is defined as a data node, given the initial parameter information
'''
class Data():
    N = 100
    T = 50
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


# driver system
class cDtX():

    d = Data()
    N = d.N
    T = d.T
    h = T/N
    alpha = d.alpha
    m = math.ceil(alpha)
    #  Application space dimension
    vn = yn = ynp = [[0] * (N+10), [0] * (N+10)]
     
    '''
     initial_function represents 
     a given deterministic function and supports re-editing
    '''
    def initial_function1(x):
        result = math.cos(x)+math.sin(x)
        return result

    def initial_function2(x):
        result = math.sin(x)
        return result
    
    fai = [initial_function1,initial_function2]

    vn[0][0] = yn[0][0] = fai[0](0)
    vn[1][0] = yn[1][0] = fai[1](0)

    '''
     Fractional derivative numerical approximation of some b(j,n+1) values
    '''
    def bjn(self, n, j):
        result = (pow(self.h, self.alpha) / self.alpha) * (pow(n-j+1, self.alpha) - pow(n-j, self.alpha))
        return result
    '''
    Fractional derivative numerical approximation of some a(j,n+1) values
    '''
    def ajn(self, n, j):
        result = pow(self.h, self.alpha) / (self.alpha * (self.alpha + 1))  
        if j == 0:
            t = pow(n,self.alpha+1) - (pow(n+1, self.alpha) * (n - self.alpha))
            return t * result
        elif j == n+1:
            return result
        else:
            t = pow(n-j+2, self.alpha + 1) + pow(n-j, self.alpha + 1) - 2 * pow(n-j+1, self.alpha + 1)
            return result * t
    '''
    To calculate y[n], value used to replace y[n+1].
    '''
    def Ynp(self, n, i):
        Tnk = (n+1) * self.h
        K = 1
        result = 0
        for kk in range(self.m):
            if kk > 0:
                K = K * kk
            result += self.fai[i]( kk*self.h ) * pow(Tnk, kk) / K
        t = 0
        for j in range(n+1): 
            tj = j * self.h
            yj = vj = []
            yj.append(self.ynp[0][j])
            yj.append(self.ynp[1][j])
            vj.append(self.vn[0][j])
            vj.append(self.vn[1][j])
            t += self.bjn(n, j) * self.f(tj, yj, vj, i)
        result += t / gamma(self.alpha)
        self.ynp[i][n+1] = result
        return result
    '''
     Approximate substitution for f (t-τ(t))
    '''
    def Vn(self, n, i):
        
        t = (n+1) * self.h
        #  Membership [0, 1)
        deerta = 0.5  # （x-x0）/(x1-x0)=(y-y0)/(y1-y0)=k

        result = 0
        if self.m  > 1:
            result = deerta * self.yn[i][n-self.m+2] + (1 - deerta) * self.yn[i][n-self.m+1]
        else:
            result = deerta * self.ynp[i][n+1] + (1 - deerta) * self.yn[i][n]
      
        self.vn[i][n+1] = result

        return result
    # τ function >=0
    def Tao(self, t):
        result = 1.38 * abs(math.sin(t))
        return result 
    '''
     Activation function
    '''
    def Activation(self, x):
        return 0.5*(abs(x+1)-abs(x-1))
        #  Avoid overflow
        # if x>=0:   
        #     return 1.0/(1+np.exp(-x))
        # else:
        #     return np.exp(x)/(1+np.exp(x))
    '''
     Capno function
    '''
    def f(self, t, y, v, i):

        if t<=0:
            return self.yn[i][0]

        result = self.d.I[i] - self.d.D[i] * y[i]
        # print(result)
        temp = 0
        TV = float("inf") # 交集取最小
        AlphaF = float("inf") # 交集取最小
        SV = float("-inf") # 并集取最大
        BeitaF = float("-inf") # 并集取最大
        for j in range(len(self.d.D)):
            temp += self.d.A[i][j] * self.Activation( y[j] )
            temp += self.d.C[i][j] * self.d.V[j]
            a = self.Activation(v[j]) 
            temp += self.d.B[i][j] * a
            TV = min(TV, self.d.Tt[i][j] * self.d.V[j])
            AlphaF = min(AlphaF, self.d.Alpha[i][j] * a)
            SV = max(SV, self.d.Ss[i][j] * self.d.V[j])
            BeitaF = max(BeitaF, self.d.Beita[i][j] * a)
        result += temp + TV + AlphaF + SV + BeitaF
        
        return result
    '''
    This function represents the fractional α derivative values
     of the x function, the yi trajectory
    '''
    def Yn(self, n, l):
        # 初始化
        self.Ynp(n, l)
        self.Vn(n,l)
    
        Tnk = (n+1) * self.h
        K = 1
        result = 0

        for kk in range(self.m):
            if kk > 0:
                K = K * kk
            result += self.fai[l]( kk*self.h ) * pow(Tnk, kk) / K

        tn = Tnk
        y = v = []
        y.append(self.ynp[0][n+1])
        y.append(self.ynp[1][n+1])
        v.append(self.vn[0][n+1])
        v.append(self.vn[1][n+1])

        result += pow(self.h, self.alpha) / gamma(self.alpha+2) * self.f(tn, y, v, l)
        t = 0

        for j in range(n+1):
            tj = j * self.h
            yj = vj = []
            yj.append(self.ynp[0][j])
            yj.append(self.ynp[1][j])
            vj.append(self.vn[0][j])
            vj.append(self.vn[1][j])

            t += self.ajn(n, j) * self.f(tj, yj, vj, l)
            
        result += t / gamma(self.alpha)

        self.yn[l][n+1] = result


# response system
class cDtY():

    te = cDtX()
    d = Data()
    N = d.N
    T = d.T
    h = T/N
    yita = [0.02, 0.4]
    yibu = [0.06, 0.3]
    alpha = d.alpha
    m = math.ceil(alpha)
    vn = yn = ynp = [[0] * (N+10), [0] * (N+10)]
    
    def initial_function1(x):
        
        result = math.sin(x)
        return result
    '''
     initial_function represents 
     a given deterministic function and supports re-editing
    '''
    def initial_function2(x):
        result = math.cos(x)
        return result

    fai = [initial_function1,initial_function2]

    vn[0][0] = yn[0][0] = fai[0](0)
    vn[1][0] = yn[1][0] = fai[1](0)

    '''
     Fractional derivative numerical approximation of some b(j,n+1) values
    '''
    def bjn(self, n, j):
        result = (pow(self.h, self.alpha) / self.alpha) * (pow(n-j+1, self.alpha) - pow(n-j, self.alpha))
        return result
    '''
     Fractional derivative numerical approximation of some a(j,n+1) values
    '''
    def ajn(self, n, j):
        result = pow(self.h,self.alpha) / (self.alpha*(self.alpha+1))
        
        if j == 0:
            t = pow(n,self.alpha+1) - pow(n+1,self.alpha)*(n-self.alpha)
            return t * result
        elif j == n+1:
            return result
        else:
            t = pow(n-j+2,self.alpha+1) + pow(n-j,self.alpha+1) - 2 * pow(n-j+1, self.alpha+1)
            return result * t
    '''
    To calculate y[n], value used to replace y[n+1].
    '''
    def Ynp(self, n, i, lis):
        Tnk = (n+1) * self.h
        K = 1
        result = 0
        for k in range(self.m):
            if k > 0:
                K = K * k
            result += self.fai[i]( k*self.h ) * pow(Tnk, k) / K
        t = 0
        
        for j in range(n+1): 
            tj = j * self.h
            vj = yj = []
            yj.append(self.ynp[0][j])
            yj.append(self.ynp[1][j])
            vj.append(self.vn[0][j])
            vj.append(self.vn[1][j])
            t += self.bjn(n, j) * self.f(tj, yj, vj, i, lis)
        result += t / gamma(self.alpha)
        self.ynp[i][n+1] = result
        return result
    '''
     Approximate substitution for f (t-τ(t))
    '''
    def Vn(self, n, i):
        
        t = (n+1) * self.h
        #  Membership [0, 1)
        deerta = 0.5  # （x-x0）/(x1-x0)=(y-y0)/(y1-y0)=k

        result = 0
        if self.m  > 1:
            result = deerta * self.yn[i][n-self.m+2] + (1 - deerta) * self.yn[i][n-self.m+1]
        else:
            result = deerta * self.ynp[i][n+1] + (1 - deerta) * self.yn[i][n]
      
        self.vn[i][n+1] = result

        return result
    # τ function
    def Tao(self, t):
        result = 1.38 * abs(math.sin(t))
        return result 
    '''
     Activation function
    '''
    def Activation(self, x):
        return 0.5*(abs(x+1)-abs(x-1))
        # #  Avoid overflow
        # if x>=0:      
        #     return 1.0/(1+np.exp(-x))
        # else:
        #     return np.exp(x)/(1+np.exp(x))
    '''
     Capno function 
    '''  
    def f(self, t, y, v, i, lis):
        if t<=0:
            return self.yn[i][0]

        result = self.d.I[i] - self.d.D[i] * y[i] + self.Cal_ui(int(t/self.h),i,lis,t)
        # print(result)
        temp = 0
        TV = float("inf") # 交集取最小
        AlphaF = float("inf") # 交集取最小
        SV = float("-inf") # 并集取最大
        BeitaF = float("-inf") # 并集取最大
        for j in range(len(self.d.D)):
            temp += self.d.A[i][j] * self.Activation( y[j] )
            temp += self.d.C[i][j] * self.d.V[j]
            a = self.Activation(v[j]) 
            temp += self.d.B[i][j]* self.Activation(a)
            TV = min(TV, self.d.Tt[i][j] * self.d.V[j])
            AlphaF = min(AlphaF, self.d.Alpha[i][j] * a)
            SV = max(SV, self.d.Ss[i][j] * self.d.V[j])
            BeitaF = max(BeitaF, self.d.Beita[i][j] * a)
        result += temp + TV + AlphaF + SV + BeitaF
        
        return result
    '''
    This function represents the fractional α derivative values
     of the x function, the yi trajectory
    '''
    def Yn(self, n, l, lis):
        # 初始化
        self.Ynp(n, l, lis)
        self.Vn(n,l)
    
        Tnk = (n+1) * self.h
        K = 1
        result = 0

        for kk in range(self.m):
            if kk > 0:
                K = K * kk
            result += self.fai[l]( kk*self.h ) * pow(Tnk, kk) / K

        tn = Tnk
        y = v = []
        y.append(self.ynp[0][n+1])
        y.append(self.ynp[1][n+1])
        v.append(self.vn[0][n+1])
        v.append(self.vn[1][n+1])

        result += pow(self.h, self.alpha) / gamma(self.alpha+2) * self.f(tn, y, v, l, lis)
        t = 0

        for j in range(n+1):
            tj = j * self.h
            yj = vj = []
            yj.append(self.ynp[0][j])
            yj.append(self.ynp[1][j])
            vj.append(self.vn[0][j])
            vj.append(self.vn[1][j])

            t += self.ajn(n, j) * self.f(tj, yj, vj, l, lis)
            
        result += t / gamma(self.alpha)

        self.yn[l][n+1] = result

    #  Calculation Controller
    def Cal_ui(self,n,l,lis,tj):
        t = self.yn[l][n]-lis[l][n]
        result =  (-self.yibu[l]*pow(t, 2) - self.yita[l] * t) * (pow(np.e,2*tj*self.yibu[l]))
        # print(result)
        return result

        

if __name__ == "__main__":
    Temp = cDtX()
    Tmp = cDtY()
    h = Tmp.h
    N = Tmp.N
    # x(t)
    for n in range(0,N+1):
        for l in range(len(Tmp.fai)):
            Temp.Yn(n,l)
    # y(t)
    for n in range(0,N+1):
        for l in range(len(Tmp.fai)):
            Tmp.Yn(n,l,Temp.yn)

    X = Temp.yn
    Y = Tmp.yn
    t = [n*h for n in range(0,N)]
    '''
     Drawing images of x1,x2
    '''
    plt.title('alpha = %s' % Temp.alpha)
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.plot(X[0][0:N], X[1][0:N]) 
    plt.show()


    '''
     Drawing images of x1,y1
    '''
    plt.title('alpha = %s' % Temp.alpha)
    plt.xlabel('t')
    plt.ylabel('y1,x1')
    plt.plot(t, X[0][0:N], label="$x1$", color='green', ls='-.')
    plt.plot(t, Y[0][0:N], label="$y1$", color='red', ls=':') 
    plt.legend()
    plt.show()

    '''
     Drawing images of x2,y2
    '''
    plt.title('alpha = %s' % Temp.alpha)
    plt.xlabel('t')
    plt.ylabel('y2,x2')
    plt.plot(t, X[1][0:N],label="$x2$", color='green', ls='-.')
    plt.plot(t, Y[1][0:N],label="$y2$", color='red', ls=':') 
    plt.legend()
    plt.show()

    '''
    Drawing the error system
    '''
    x = [n*h for n in range(0,N)]
    err = [round(Y[0][i]-X[0][i], 4) for i in range(0,N)]
    plt.xlabel('t')
    plt.ylabel('error')
    plt.plot(x,err)
    plt.show()

    err = [round(Y[1][i]-X[1][i],4) for i in range(0,N)]
    plt.xlabel('t')
    plt.ylabel('error')
    plt.plot(x,err)
    plt.show()
   