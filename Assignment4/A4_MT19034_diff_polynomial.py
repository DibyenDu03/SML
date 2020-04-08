# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 20:46:52 2020

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
from tqdm import tqdm
from A4_MT19034_multi_polynomial import create_final

path='dataset.txt'
fp = codecs.open(path,"r",encoding='utf-8', errors='ignore')
text=fp.read()
data=text.split('\r\n')
dataset=[]
tol=506
token = RegexpTokenizer('\ ', gaps = True)
price=[]
data_set=[]

for i in range(0,tol):
    str123=data[i*2].strip()
    str123=str123+' '+data[i*2+1].strip()
    data_set.append(str123)
random.shuffle(data_set)

l=[]
x=[]
x_t=[]

tes_per=.8
data_test=data_set[math.ceil(tol*tes_per):tol]
data_set=data_set[:math.ceil(tol*tes_per)]
tol=math.ceil(tol*tes_per)

weight=[]

m=4
index=[1]
print("\nTraining ....\n")
sw=[]
for degree in tqdm(range(1,m+1)):
    l.append(degree)
    dataset=[]
    error=0.0
    error_t=0.0
    
    #print('\n-----------------------------------------------------------------------------------')
    #print()
    
    price=[]
    for i in range(0,tol):
        fre=[]
        fre1=[]
        fre1.append(1.0)
        
        fre=(token.tokenize(data_set[i]))
        fre=[float(i) for i in fre]
        fre2,l1=create_final(fre[:len(fre)-1],degree)
        fre1.extend(fre2)
            
        
        dataset.append(fre1)
        price.append(fre[13])
        
    index.append(l1+1) 
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
    
    
    #print()
    #print("Degree is- ",degree)
    
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
        #print("Train Accuracy- ",(100-sum),"%","\t\tTest Accuracy- ",(100-sum_t),"%")
        
        error_t+=sum_t
        error+=sum
        #print()
    #print()
    #print("Average train RMSE- ",error_t/k,"\tAverage test RMSE- ",error/k,"  ",end="\n")
    #print()
    x.append(error/k)
    x_t.append(error_t/k)
    sw.append(weight)
    #print('-----------------------------------------------------------------------------------')


l=np.array(l)
m_rmse=np.max(x)
ind=0
for i in range(0,len(x)):
    if(m_rmse>x[i]):
        ind=i
        m_rmse=x[i]
print("\nLeast mean validation error is ",m_rmse,"\t degree is ",l[ind])

###########################################_TESTING_##########################################
x_tr=[]

print("\nTesting ....\n")

for degree in tqdm(range(1,m+1)):
    #print("\nFor degree ",degree," Training vs Testing ",tes_per*100," : ",(100-tes_per*100),"\n")
    X_train=[]
    dataset=[]
    
    for i in range(0,tol):
        fre=[]
        fre1=[]
        fre1.append(1.0)
        
        fre=(token.tokenize(data_set[i]))
        fre=[float(i) for i in fre]
        fre2,l1=create_final(fre[:len(fre)-1],degree)
        fre1.extend(fre2)
            
        
        dataset.append(fre1)
    
    for i in dataset:
        X_train.append(i[:index[degree]])
    x_train=np.array(X_train)
    s=np.dot(x_train.T,x_train)
    x1=np.linalg.pinv(s)
    x2=np.dot(x1,x_train.T)
    price1=np.array(price)
    W=np.dot(x2,price1)
    
    dataset1=[]
    data_set1=data_test
    y_target=[]
    for i in range(0,len(data_test)):
            fre=[]
            fre1=[]
            fre1.append(1.0)
            
            fre=(token.tokenize(data_set1[i]))
            fre=[float(i) for i in fre]
            fre2,l1=create_final(fre[:len(fre)-1],degree)
            fre1.extend(fre2)

            dataset1.append(fre1)
            y_target.append(fre[(13)])
    dataset1=np.array(dataset1)
    out=np.dot(W,dataset1.T)
    loss=(y_target-out)*(y_target-out)
    sum=math.sqrt(np.sum(loss)/len(loss))
    #print()
    #print("Final error is ",sum)
    x_tr.append(sum)
    #print()

plt.plot(l,x_t,label = "Training")
plt.plot(l,x,label = "Validation")
plt.xlabel('Degree ->') 
plt.ylabel('RMSE ->')  
plt.legend() 
plt.show()

plt.plot(l,x_tr,label = "Test")
plt.xlabel('Degree ->') 
plt.ylabel('RMSE ->')  
plt.legend() 
plt.show()

deg=[1,l[ind]]

print("\nTraining vs Testing ",tes_per*100," : ",(100-tes_per*100),"\n")

for i in deg:
    print("\nFor degree ",i,"Test error is ",x_tr[i-1])




    
    
    
