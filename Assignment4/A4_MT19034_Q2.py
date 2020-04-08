# -*- coding: utf-8 -*-
"""
Created on Sat Feb 22 07:17:30 2020

@author: Dibyendu
"""

import os
import codecs
import string
import os
import nltk
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize
import pickle
import math
import numpy as np
from nltk.tree import *
from nltk.stem import WordNetLemmatizer 
import random
import numpy as num
from nltk.tokenize import RegexpTokenizer 
import matplotlib.pyplot as plt

path='data.csv'
fp = codecs.open(path,"r",encoding='utf-8', errors='ignore')
text=fp.read()
data=text.split('\r\n')

data_set=[]
price=[]
data=data[1:201]
random.shuffle(data)

for i in range(0,len(data)):
    str=data[i].split(',')
    data_set.append(float(str[0]))
    price.append(float(str[1]))
 
dataset=[]
tol=200

l=[1,2,4,5,10,15,30]
#l=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30]

x=[]
x_t=[]
for degree in l:
    dataset=[]
    
    x_test=[]
    y_pred1=[]
    y_target1=[]
    
    error=0.0
    error_t=0.0
    print('-----------------------------------------------------------------------------------')
    print("Degree is ",degree)
    for i in range(0,tol):
        fre=[]
        fre1=[]
        fre1.append(1.0)
        fre.append(float(data_set[i]))
        #x.append(fre[0])
        for j in range(0,degree):
            fre1.append(math.pow(fre[0],(j+1)))
            
        dataset.append(fre1[:(1+1*degree)])
    
    k_fold=[]
    k=5
    k_price=[]
    old=0
    for i in range(0,k):
        k_fold.append(dataset[old:int(tol/k*(i+1))])
        k_price.append(price[old:int(tol/k*(i+1))])
        old=int(tol/k*(i+1))
        #print(old)
    
    ''' 
    for i in range(0,k):
        print(len(k_fold[i])," ",len(k_price[i]),end="\n")
    '''
    
    train=[]
    test=[]
    price=np.array(price)
    weight=[]
    for i in range(0,k):
        test=k_fold[i]
        test_p=k_price[i]
        train=[]
        train_p=[]
        for j in range(0,k):
            if(i!=j):
                train.extend(k_fold[j])
                train_p.extend(k_price[j])
        
        test=np.array(test)
        X=np.array(train)
        train_p=np.array(train_p)
        test_p=np.array(test_p)
        train=np.array(train)
        train_p=np.array(train_p)
        
        s=np.dot(X.T,X)
        x1=np.linalg.pinv(s)
        x2=np.dot(x1,X.T)
        
        W=np.dot(x2,train_p)
        W=np.array(W)
        weight.append(W)
        
        y_pred=np.dot(W,test.T)
        out=(test_p-y_pred)*(test_p-y_pred)
        
        y_train=np.dot(W,train.T)
        out_train=(train_p-y_train)*(train_p-y_train)
        sum_t=np.sum(out_train)/len(out_train)
        sum_t=np.sqrt(sum_t)
        
        sum=np.sum(out)/len(out)
        sum=np.sqrt(sum)
        
        #print()
        #print("Test fold- ",(i+1),"-> ")
        #print("Train RMSE- ",sum_t,"  ","\t\tTest RMSE- ",sum,"  ",end="\n")
        error_t+=sum_t
        error+=sum
        for i in range(0,len(test)):
            x_test.append(test[i][1])
            y_pred1.append(y_pred[i])
            y_target1.append(test_p[i])
        
        #print("Train Accuracy- ",(100-sum),"%","\t\tTest Accuracy- ",(100-sum_t),"%")
    print()
    print("train RMSE- ",error_t/k,"\ttest RMSE- ",error/k,"  ",end="\n")
    print()
    x.append(error/k)
    x_t.append(error_t/k)
    plt.scatter(x_test,y_target1,label="Actual")
    plt.scatter(x_test,y_pred1,label="Predict")
    plt.legend() 
    plt.show()
    print('-----------------------------------------------------------------------------------')
plt.plot(l,x_t,label = "Train")
plt.plot(l,x,label = "Validation")
plt.xlabel('Degree ->') 
plt.ylabel('RMSE ->')  
plt.legend() 
plt.show()  
l=np.array(l)
m_rmse=np.max(x)
ind=0
for i in range(0,len(x)):
    if(m_rmse>x[i]):
        ind=i
        m_rmse=x[i]
print("Least mean validation error is ",m_rmse,"\t degree is ",l[ind])  



    