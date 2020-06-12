#贪心法
import pandas as pd
import numpy as np
import math
import torch
import time
def getset(citynumber,samples):
    torch.manual_seed(66)
    data_set = []
    for l in range(samples):
        #生成在坐标在0 1 之间的
        x = torch.FloatTensor(2, citynumber*2).uniform_(0, 1)
        data_set.append(x)
    return data_set
trainset=getset(10,100)
data_set=[]
for i in range(100):
    data_set.append(np.array(trainset[i]))
#print(data_set)
print(data_set[0][1])
dist=np.zeros((10,10))
total=0
for p in range(100):
    for i in range(10):
        for j in range(10):
            dist[i][j]=math.sqrt((data_set[p][1][i]-data_set[p][1][j])**2+(data_set[p][0][i]-data_set[p][0][j])**2)
    i=1
    n=10
    j=0
    sumpath=0
    s=[]
    s.append(0)
    start = time.clock()
    while True:
        k=1
        Detemp=10000000
        while True:
            l=0
            flag=0
            if k in s:
                flag = 1
            if (flag==0) and (dist[k][s[i-1]] < Detemp):
                j = k
                Detemp=dist[k][s[i - 1]]
            k+=1
            if k>=n:
                break
        s.append(j)
        i+=1
        sumpath+=Detemp
        if i>=n:
            break
    sumpath+=dist[0][j]
    end = time.clock()
    print("结果：")
    total+=sumpath
    print(sumpath)
    for m in range(n):
        print("%s "%(s[m]),end='')
print(total/100)