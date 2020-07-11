import math
 
userin = input("请输入海明码:")
haiming=[]
#输入的海明码是字符串，下面的代码是将字符串转化成整型列表
haiming=list(haiming)
for x in userin:
	haiming.append(int(x))
 
#计算校验码的个数
for i in range(0,len(haiming)):
	if 2**i>len(haiming):
		break
#校验码个数
num = i
#标记错位
flag = 0
for i in range(0,num):
	b=[]
	if(i==0):
		#利用python的步长。
		a = haiming[(2**i)-1:len(haiming):(2**i)+1]
		if(a.count(1)%2==1):
			flag = flag + 2**i
		print ("a",a)
	else:
		for j in range(2**i,len(haiming)+1):
			if((j/(2**i))%2==1):
				for k in range(j,j+(2**i)):
					if(k>len(haiming)):
						break
					b.append(haiming[k-1])
		if(b.count(1)%2==1):
			flag = flag+2**i
		print ("b",b)
	del(b)
print(flag)
if flag != 0:
    print("flag",flag)
    print("长度：",len(haiming))
    if haiming[flag-1]==0:
            haiming[flag-1]=1
    elif haiming[flag-1]==1:
            haiming[flag-1]=0
    print ("第%d位出错" %(flag))
    print (haiming)
else:
        print ("没有错误")        
input("Press <enter>")