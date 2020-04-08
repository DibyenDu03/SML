# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 02:21:47 2020

@author: Dibyendu
@Rollno: MT19034
"""

import math
import numpy as np
import matplotlib.pyplot as plt




def mean(data):
    
    l=len(data)
    sum=0.0
    for i in range(l):
        sum+=data[i][1]
    sum=sum/l

    var=0.0
    for i in range(l):
        var+=(data[i][1]-sum)**2
    var=var/l
    print("\nMean of train smaple- ",sum," Variance of train sample- ",var,"\n")
    
    return [sum,var]

data=[[0,-45],[1,-51],[2,-58],[3,-63],[4,-36],[5,-52],[6,-59],[7,-62],[8,-36],[9,-43],[10,-55],[11,-64]]
train=[1,3,5,7,9,11,12]
test=[2,4,6,8,10]
data_train=[[0,-45],[2,-58],[4,-36],[6,-59],[8,-36],[10,-55],[11,-64]]
sigma=mean(data_train)[1]
#sigma=998.5600000000001

def kernal(x1,x2,l=1):
    
    #sigma=20
    val=sigma*math.exp(-(x1-x2)**2/(2*l*l))
    
    return val
    
            

def gaussian_process(train,test,kmatrix,yout):
    
    s=[]
    yout=np.matrix(yout)
    ymean=[]
    var=[]
    for each in test:
        kstar=[]
        for k in train:
            kstar.append(kernal(each,k))
        kstar=np.matrix(kstar)
        mean=np.matmul(np.matmul(kstar,np.linalg.inv(kmatrix)),yout.T)
        ymean.append(mean.tolist()[0][0])
        var1=kernal(each,each)-np.matmul(np.matmul(kstar,np.linalg.inv(kmatrix)),kstar.T)
        s.append(kstar)
        var.append(var1.tolist()[0][0])
    return ymean,var,s

def KMatrix(train):
    
    kmat=[]
    
    for i in train:
        a=[]
        for k in train:
            s=kernal(i,k)
            a.append(s)
        kmat.append(a)
    kmat=np.matrix(kmat)
    
    return kmat
            
            
xpoin=[]
out=[]

for i in train:
    xpoin.append(data[i-1][0])
    out.append(data[i-1][1])

kmat=KMatrix(xpoin) 
    
pred=[] 
for i in test:
    pred.append(data[i-1][0])

ymean,var,s=gaussian_process(xpoin,pred,kmat,out)

xlow=[]
xhigh=[]
ind=0
y=[]
actual=[]

for i in data:
    actual.append(i[1])

for i in range(len(test)):
    ind=i
    xlow.append(out[i])
    xhigh.append(out[i])
    
    xlow.append((ymean[i]-var[i]/2))
    xhigh.append((ymean[i]+var[i]/2))
    
    y.append(xpoin[i])
    y.append(pred[i])
    

for i in range(ind+1,len(train)):
    xlow.append(out[i])
    xhigh.append(out[i])
    y.append(xpoin[i])

print("Predicted means-\n ",ymean)
print("\nPredicted variance-\n ",var)

plt.plot(y,xhigh,label = "High Bound")
plt.plot(y,actual,label = "Actual line")
plt.scatter(y,actual,label = "Actual Points") 
plt.plot(y,xlow,label = "Low Bound")
plt.xlabel('Distance ->') 
plt.ylabel('Signal Strength ->') 
plt.legend()
plt.show()


