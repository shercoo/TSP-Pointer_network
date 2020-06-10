import numpy as np

x=[20,25,30,35,40,50,60,65,70,75,80,90]
y=[1.81,1.70,1.65,1.55,1.48,1.40,1.30,1.26,1.24,1.21,1.20,1.18]

x=np.array(x)
x2=x**2
print(x)
print(x2)
n=x.shape[0]
X=np.vstack( (np.ones(n),x,x2)).transpose((1,0))
XT=X.transpose((1,0))
Y=np.array(y).transpose()
print(X)
print(Y)
print(np.matmul(np.matmul(np.linalg.inv(np.matmul(XT,X)),XT),Y))

X=X-X.mean(0,keepdims=True)
X=X[:,1:3]
print(X)
Y=(Y-Y.mean(0))
print(Y)
Sxx=np.matmul(X.transpose(1,0),X)
Sxy=np.matmul(X.transpose(1,0),Y)
Syy=np.matmul(Y.transpose(),Y)
Sxy=np.expand_dims(Sxy,1)
print(Sxx)
print(Sxy)
print(Syy)
U=np.matmul(np.matmul(Sxy.transpose((1,0)),np.linalg.inv(Sxx)),Sxy).squeeze(0)
Q=Syy-U
print(U)
print(Q)
from scipy.stats import *
print(U/2/(Q/9))
