from scipy.special import gamma
import matplotlib.pyplot as plt
import numpy as np
import math

'''
This class is defined as a data node, given the initial parameter information
'''
class Data():
    D = [1, 1]
    A = np.array([[-0.56,-0.11],[-0.17,2.2]])
    B = np.array([[-1.6, -0.1],[-0.18,0]])
    I = np.array([0, 0])
    Alpha = np.array([[0.5, 0],[0, 0.5]])
    Beita = np.array([[0.52, 0], [0, 0.12]])
    Tt = np.array([[0.1, 0.82],[0.56, 0.1]])
    Ss = np.array([[0.37, 0.7],[0.16, 0.1]])
    C = np.array([[0.1, 0.02],[0.1, 0.27]])
    V = np.array([1, 1])


# driver system
class cDtX():

    N = 200
    T = 20
    h = T/N
    d = Data()
    alpha = 0.7
    m = math.ceil(alpha)
    #  Application space dimension
    vn = yn = ynp = yt = [0] * (N+10)
    yn[0] = 0.4 
    '''
     initial_function represents 
     a given deterministic function and supports re-editing
    '''
    def initial_function1(x):
        result = abs(math.cos(x))
        return result

    def initial_function2(x):
        result = math.sin(x+1.57)
        return result

    fai = [initial_function1,initial_function2]
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
    def Ynp(self, n, i):
        
        Tnk = (n+1) * self.h
        K = 1
        result = 0

        for k in range(self.m):
            if k > 0:
                K = K * k
            result += self.fai[i]( k*self.h ) * pow(Tnk, k) / K
        
        t = 0
        for j in range(n+1):
            # 有待补充内容
            tj = j * self.h
            yj = self.yn[j]
            vj = self.vn[j]
            # print('yj',yj)
            # print('vj',vj)
            # print("self.f(x)", self.f(tj, yj, vj, j))
            t += self.bjn(n+1, j) * self.f(tj, yj, vj, i)
        
        result += t / gamma(self.alpha)
        # print("  YNP:  ", result)
        self.ynp[n+1] = result

        return result
    '''
     Approximate substitution for f (t-τ(t))
    '''
    def Vn(self, n):
        
        t = (n+1) * self.h
        deerta = self.m - self.Tao(t)/self.h

        result = 0

        if self.m == 1:
            result = deerta * self.ynp[n+1] - (1 - deerta) * self.yn[n]
        else:
            result = deerta * self.yn[n-self.m+2] + (1 - deerta) * self.yn[n-self.m+1]
        # print("  VN:  ", result)
        self.vn[n+1] = result

        return result
    # τ function
    def Tao(self, t):
        result = 1.38 * abs(math.sin(t))
        return result 
    '''
     Activation function
    '''
    def Activation(self, x):
        #  Avoid overflow
        if x>=0:   
            return 1.0/(1+np.exp(-x))
        else:
            return np.exp(x)/(1+np.exp(x))
    '''
     Capno function
    '''
    def f(self, t, y, v, i):

        if t<=0:
            return self.yn[0]
        result = self.d.I[i] - self.d.D[i] * y 
        # print(result)
        temp = 0
        TV = float("inf") # 交集取最小
        AlphaF = float("inf") # 交集取最小
        SV = float("-inf") # 并集取最大
        BeitaF = float("-inf") # 并集取最大
        for j in range(len(self.d.D)):
            temp += self.d.A[i][j] * self.Activation(y)
            temp += self.d.C[i][j] * self.d.V[j]
            temp += self.d.B[i][j]* self.Activation(v)
            a = self.Activation(v) 
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
        self.Vn(n)
    
        Tnk = (n+1) * self.h
        K = 1
        result = 0

        for k in range(self.m):
            if k > 0:
                K = K * k
            if k*self.h <= 0:
                result += self.yn[0] * pow(Tnk, k) / K
            else:
                result += self.fai[l]( k*self.h ) * pow(Tnk, k) / K

        tn = Tnk
        y = self.ynp[n+1]
        v = self.vn[n+1]

        result += pow(self.h, self.alpha) / gamma(self.alpha+2) * self.f(tn, y, v, l)
        t = 0

        for j in range(n+1):
            tj = j * self.h
            yj = self.yn[j]
            vj = self.vn[j]

            t += self.ajn(n, j) * self.f(tj, yj, vj, l)
            
        result += t / gamma(self.alpha)

        self.yn[n+1] = result


# response system
class cDtY():

    te = cDtX()
    d = Data()
    N = te.N
    T = te.T
    h = T/N
    yita = [0.1, 0.16]
    yibu = [0.02, 0.04]
    alpha = te.alpha
    m = math.ceil(alpha)
    vn = yn = ynp = yt = [0] * (N+10)
    yn[0] = 0.35

    
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
    def Ynp(self, n, i,lis):
        
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
            yj = self.yn[j]
            vj = self.vn[j]
            t += self.bjn(n+1, j) * self.f(tj, yj, vj, i,lis)
        
        result += t / gamma(self.alpha)
        self.ynp[n+1] = result

        return result
    '''
     Approximate substitution for f (t-τ(t))
    '''
    def Vn(self, n):
        
        t = (n+1) * self.h
        deerta = self.m - self.Tao(t)/self.h

        result = 0

        if self.m == 1:
            result = deerta * self.ynp[n+1] - (1 - deerta) * self.yn[n]
        else:
            result = deerta * self.yn[n-self.m+2] + (1 - deerta) * self.yn[n-self.m+1]
        # print("  VN:  ", result)
        self.vn[n+1] = result

        return result
    # τ function
    def Tao(self, t):
        result = 1.38 * abs(math.sin(t))
        return result 
    '''
     Activation function
    '''
    def Activation(self, x):
        #  Avoid overflow
        if x>=0:      
            return 1.0/(1+np.exp(-x))
        else:
            return np.exp(x)/(1+np.exp(x))
    '''
     Capno function
    '''  
    def f(self, t, y, v, i,lis):
        if t<=0:
            return self.yn[0]

        result = self.d.I[i] - self.d.D[i] * y + self.Cal_ui(int(t/self.h),i,lis,t)
       
        temp = 0
        TV = float("inf") 
        AlphaF = float("inf") 
        SV = float("-inf") 
        BeitaF = float("-inf") 
        for j in range(len(self.d.D)):
            temp += self.d.A[i][j] * self.Activation(y)
            temp += self.d.C[i][j] * self.d.V[j]
            temp += self.d.B[i][j]* self.Activation(v)
            a = self.Activation(v) 
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
    def Yn(self, n, l,lis):
        # 初始化
        self.Ynp(n, l,lis)
        self.Vn(n)

        Tnk = (n+1) * self.h
        K = 1
        result = 0

        for k in range(self.m):
            if k > 0:
                K = K * k
            if k*self.h <= 0:
                result += self.yn[0] * pow(Tnk, k) / K
            else:
                result += self.fai[l]( k*self.h ) * pow(Tnk, k) / K

        tn = Tnk
        y = self.ynp[n+1]
        v = self.vn[n+1]

        result += pow(self.h, self.alpha) / gamma(self.alpha+2) * self.f(tn, y, v, l,lis)
        t = 0

        for j in range(n+1):
            tj = j * self.h
            yj = self.yn[j]
            vj = self.vn[j]

            t += self.ajn(n, j) * self.f(tj, yj, vj, l,lis)
            
        result += t / gamma(self.alpha)

        self.yn[n+1] = result
    #  Calculation Controller
    def Cal_ui(self,n,l,lis,tj):
        t = round(lis[n]-self.yn[n],3)
        result =  (self.yibu[l]*abs(t) - self.yita[l] * t)*(pow(np.e,2*tj*self.yibu[l]))
        return round(result,3)

        

if __name__ == "__main__":
    Temp = cDtX()
    Tmp = cDtY()
    X,Y = [],[]
    h = Tmp.h
    N = Tmp.N

    for l in range(len(Tmp.fai)):
        for n in range(0,N+1):
            Temp.Yn(n,l)
        for n in range(0,N+1):
            Tmp.Yn(n,l,Temp.yn)

        X.append(Tmp.te.yn[0:N])
        Y.append(Tmp.yn[0:N])
    t = [n*h for n in range(0,N)]

    '''
     Drawing images of x1,x2
    '''
    plt.title('alpha = %s' % Temp.alpha)
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.plot(X[0], X[1]) 
    plt.show()


    '''
     Drawing images of x1,y1
    '''
    plt.title('alpha = %s' % Temp.alpha)
    plt.xlabel('x')
    plt.ylabel('y1,x1')
    plt.plot(t, X[0], label="$x1$", color='green', ls='-.')
    plt.plot(t, Y[0], label="$y1$", color='red', ls=':') 
    plt.legend()
    plt.show()

    '''
     Drawing images of x2,y2
    '''
    plt.title('alpha = %s' % Temp.alpha)
    plt.xlabel('t')
    plt.ylabel('y2,x2')
    plt.plot(t, X[1],label="$x2$", color='green', ls='-.')
    plt.plot(t, Y[1],label="$y2$", color='red', ls=':') 
    plt.legend()
    plt.show()

    '''
    Drawing the error system
    '''
    x = [n*h for n in range(0,N)]
    err = [Y[0][i]-X[0][i] for i in range(0,N)]
    plt.plot(x,err)
    plt.show()

    err = [Y[1][i]-X[1][i] for i in range(0,N)]
    plt.plot(x,err)
    plt.show()
   