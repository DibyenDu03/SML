# -*- coding: utf-8 -*-
"""
Created on Sun Feb  2 09:48:38 2020

@author: Dibyendu
"""

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
from math import exp
import struct as st
from PIL import Image
import idx2numpy
from numpy import linalg
import matplotlib.pyplot as plt
from scipy.linalg import eigh
from sklearn.preprocessing import StandardScaler as SS
from tqdm import tqdm





##################################_LDA_########################################


def generate_Discrimate(X,mU,sigma,prob,dim):
    sigmai=np.linalg.pinv(sigma)
    #lamda=sigma_Add(sigma)
    discrimate= - (1/2)*num.matmul(  num.transpose((X-mU)) , (num.matmul(sigmai,(X-mU))) ) + math.log(prob)           
    return float(discrimate)

def LDA(tresult,labels,result,Tlabel):
    mean=np.mean(result, axis = 0)
    col=len(result[0])
    normalised= np.zeros((len(result),col))
    
    for i in range(0,len(tresult)):
        normalised[i]=result[i]-mean
    
    dataset=[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
    data1=[]
    data2=[]
    data3=[]
    data4=[]
    data5=[]
    data6=[]
    data7=[]
    data8=[]
    data9=[]
    data0=[]
    
    for i in range(0,len(Tlabel)):
        dataset[Tlabel[i]]+=1.0
        if(Tlabel[i]==0):
            data0.append(result[i])
        if(Tlabel[i]==1):
            data1.append(result[i])
        if(Tlabel[i]==2):
            data2.append(result[i])
        if(Tlabel[i]==3):
            data3.append(result[i])
        if(Tlabel[i]==4):
            data4.append(result[i])
        if(Tlabel[i]==5):
            data5.append(result[i])
        if(Tlabel[i]==6):
            data6.append(result[i])
        if(Tlabel[i]==7):
            data7.append(result[i])
        if(Tlabel[i]==8):
            data8.append(result[i])
        if(Tlabel[i]==9):
            data9.append(result[i])
    #print(dataset," ",len(data1))       
    data1=np.array(data1)
    data2=np.array(data2)
    data3=np.array(data3)
    data4=np.array(data4)
    data5=np.array(data5)
    data6=np.array(data6)
    data7=np.array(data7)
    data8=np.array(data8)
    data9=np.array(data9)
    data0=np.array(data0)
    
    
    mean0=np.mean(data0, axis = 0)
    mean1=np.mean(data1, axis = 0)
    mean2=np.mean(data2, axis = 0)
    mean3=np.mean(data3, axis = 0)
    mean4=np.mean(data4, axis = 0)
    mean5=np.mean(data5, axis = 0)
    mean6=np.mean(data6, axis = 0)
    mean7=np.mean(data7, axis = 0)
    mean8=np.mean(data8, axis = 0)
    mean9=np.mean(data9, axis = 0)
    
    
    cov0=np.cov(np.transpose(data0))
    cov1=np.cov(np.transpose(data1))
    cov2=np.cov(np.transpose(data2))
    cov3=np.cov(np.transpose(data3))
    cov4=np.cov(np.transpose(data4))
    cov5=np.cov(np.transpose(data5))
    cov6=np.cov(np.transpose(data6))
    cov7=np.cov(np.transpose(data7))
    cov8=np.cov(np.transpose(data8))
    cov9=np.cov(np.transpose(data9))
    
    count=[]
    val=[]
    cor=0.0
    
    mean1=np.mean(tresult, axis = 0)
    
    for i in range(0,len(tresult)):
        tresult[i]-=mean1
    
    for i in tqdm(range(0,len(tresult))):
        max=-999999999999999999999
        ind=-1
        
        s=[]
        X=tresult[i]
        mU=mean0
        sigma=cov0
        dval=generate_Discrimate(X,mU,sigma,(dataset[0]/60000),len(tresult[0]))
        s.append(dval)
        if(max<dval):
            max=dval
            ind=0
        
        X=tresult[i]
        mU=(mean1)
        sigma=(cov1)
        dval=generate_Discrimate(X,mU,sigma,(dataset[1]/60000),len(result[0]))
        s.append(dval)
        if(max<dval):
            max=dval
            ind=1
            
        X=(tresult[i])
        mU=(mean2)
        sigma=(cov2)
        dval=generate_Discrimate(X,mU,sigma,(dataset[2]/60000),len(tresult[0]))
        s.append(dval)
        if(max<dval):
            max=dval
            ind=2
            
        X=(tresult[i])
        mU=(mean3)
        sigma=(cov3)
        dval=generate_Discrimate(X,mU,sigma,(dataset[3]/60000),len(tresult[0]))
        s.append(dval)
        if(max<dval):
            max=dval
            ind=3
        
        X=(tresult[i])
        mU=(mean4)
        sigma=(cov4)
        dval=generate_Discrimate(X,mU,sigma,(dataset[4]/60000),len(tresult[0]))
        s.append(dval)
        if(max<dval):
            max=dval
            ind=4
            
        X=(tresult[i])
        mU=(mean5)
        sigma=(cov5)
        dval=generate_Discrimate(X,mU,sigma,(dataset[5]/60000),len(tresult[0]))
        s.append(dval)
        if(max<dval):
            max=dval
            ind=5
        
        X=(tresult[i])
        mU=(mean6)
        sigma=(cov6)
        dval=generate_Discrimate(X,mU,sigma,(dataset[6]/60000),len(tresult[0]))
        s.append(dval)
        if(max<dval):
            max=dval
            ind=6
        
        X=(tresult[i])
        mU=(mean7)
        sigma=(cov7)
        dval=generate_Discrimate(X,mU,sigma,(dataset[7]/60000),len(tresult[0]))
        s.append(dval)
        if(max<dval):
            max=dval
            ind=7
            
        X=(tresult[i])
        mU=(mean8)
        sigma=(cov8)
        dval=generate_Discrimate(X,mU,sigma,(dataset[8]/60000),len(tresult[0]))
        s.append(dval)
        if(max<dval):
            max=dval
            ind=8
            
        X=(tresult[i])
        mU=(mean9)
        sigma=(cov9)
        dval=generate_Discrimate(X,mU,sigma,(dataset[9]/60000),len(tresult[0]))
        s.append(dval)
        if(max<dval):
            max=dval
            ind=9
        
        if(ind==labels[i]):
            cor+=1.0
        
        count.append(s)
        val.append(max)
    print("\n Accuracy ",cor/len(tresult)*100)
    return count,val





def PCA(result,energy):
    
    covdata=np.dot(np.transpose(result),result)
    eigen,vector = eigh(covdata)
   
    vector = vector.T
    
    eigensum=np.sum(eigen)
    current=0.0
    index=0
    for i in range(783,-1,-1):
        current+=eigen[i]
        if(current>=energy/100*(eigensum)):
            index=i
            break
   
    pca=[]
    for i in range(783,index-1,-1):
        pca.append(vector[i])
    
    projection=np.dot(result,np.transpose(pca))
    
    plt.scatter(vector[783],vector[782])
    
    #plt.imshow(vector[783].reshape(28,28),cmap='gray')
    
    return projection,pca,covdata,eigen

#############################_FDA_#############################################

def FDA(result,labels):
    
    mean=np.mean(result, axis = 0)
    
    dataset=[0,0,0,0,0,0,0,0,0,0]
    data1=[]
    data2=[]
    data3=[]
    data4=[]
    data5=[]
    data6=[]
    data7=[]
    data8=[]
    data9=[]
    data0=[]
    
    for i in range(0,len(labels)):
        dataset[labels[i]]+=1.0
        if(labels[i]==0):
            data0.append(result[i])
        if(labels[i]==1):
            data1.append(result[i])
        if(labels[i]==2):
            data2.append(result[i])
        if(labels[i]==3):
            data3.append(result[i])
        if(labels[i]==4):
            data4.append(result[i])
        if(labels[i]==5):
            data5.append(result[i])
        if(labels[i]==6):
            data6.append(result[i])
        if(labels[i]==7):
            data7.append(result[i])
        if(labels[i]==8):
            data8.append(result[i])
        if(labels[i]==9):
            data9.append(result[i])
            
    data1=np.array(data1)
    data2=np.array(data2)
    data3=np.array(data3)
    data4=np.array(data4)
    data5=np.array(data5)
    data6=np.array(data6)
    data7=np.array(data7)
    data8=np.array(data8)
    data9=np.array(data9)
    data0=np.array(data0)
    
    
    mean0=np.mean(data0, axis = 0)
    mean1=np.mean(data1, axis = 0)
    mean2=np.mean(data2, axis = 0)
    mean3=np.mean(data3, axis = 0)
    mean4=np.mean(data4, axis = 0)
    mean5=np.mean(data5, axis = 0)
    mean6=np.mean(data6, axis = 0)
    mean7=np.mean(data7, axis = 0)
    mean8=np.mean(data8, axis = 0)
    mean9=np.mean(data9, axis = 0)
    
    
    cov0=np.cov(np.transpose(data0))
    cov1=np.cov(np.transpose(data1))
    cov2=np.cov(np.transpose(data2))
    cov3=np.cov(np.transpose(data3))
    cov4=np.cov(np.transpose(data4))
    cov5=np.cov(np.transpose(data5))
    cov6=np.cov(np.transpose(data6))
    cov7=np.cov(np.transpose(data7))
    cov8=np.cov(np.transpose(data8))
    cov9=np.cov(np.transpose(data9))
    
    Sw=cov0+cov1+cov2+cov3+cov4+cov5+cov6+cov7+cov8+cov9
    
    x0=(mean0-mean)
    x0=np.matrix(x0)
    y0=np.transpose(x0)
    z0=np.dot(y0,x0)
    z0=np.array(z0)
    Sb=z0*dataset[0]
    
    x0=(mean1-mean)
    x0=np.matrix(x0)
    y0=np.transpose(x0)
    z0=np.dot(y0,x0)
    z0=np.array(z0)
    Sb+=z0*dataset[1]
    
    x0=(mean2-mean)
    x0=np.matrix(x0)
    y0=np.transpose(x0)
    z0=np.dot(y0,x0)
    z0=np.array(z0)
    Sb+=z0*dataset[2]
    
    x0=(mean3-mean)
    x0=np.matrix(x0)
    y0=np.transpose(x0)
    z0=np.dot(y0,x0)
    z0=np.array(z0)
    Sb+=z0*dataset[3]
    
    x0=(mean4-mean)
    x0=np.matrix(x0)
    y0=np.transpose(x0)
    z0=np.dot(y0,x0)
    z0=np.array(z0)
    Sb+=z0*dataset[4]
    
    x0=(mean5-mean)
    x0=np.matrix(x0)
    y0=np.transpose(x0)
    z0=np.dot(y0,x0)
    z0=np.array(z0)
    Sb+=z0*dataset[5]
    
    x0=(mean6-mean)
    x0=np.matrix(x0)
    y0=np.transpose(x0)
    z0=np.dot(y0,x0)
    z0=np.array(z0)
    Sb+=z0*dataset[6]
    
    x0=(mean7-mean)
    x0=np.matrix(x0)
    y0=np.transpose(x0)
    z0=np.dot(y0,x0)
    z0=np.array(z0)
    Sb+=z0*dataset[7]
    
    x0=(mean8-mean)
    x0=np.matrix(x0)
    y0=np.transpose(x0)
    z0=np.dot(y0,x0)
    z0=np.array(z0)
    Sb+=z0*dataset[8]
    
    x0=(mean9-mean)
    x0=np.matrix(x0)
    y0=np.transpose(x0)
    z0=np.dot(y0,x0)
    z0=np.array(z0)
    Sb+=z0*dataset[9]
    
    lamda=np.dot(np.linalg.pinv(Sw),Sb)
    Weigen,Wvector = eigh(lamda)
    Wvector=Wvector.T
    index=[]
    
    val=[]
    for i in range(0,9):
        max=0
        ind=0
        for j in range(0,len(Weigen)):
            if(max<Weigen[j] and Weigen[j] not in val):
                max=Weigen[j]
                ind=j
        index.append(ind)
        val.append(max)
    
    top=[]
    temp=Wvector
    for i in index:
        top.append(temp[i])
        
    plt.scatter(top[0],top[1])
    
    projection=np.dot(result,np.transpose(top))
    
    return projection,top


hand_image='train-images.idx3-ubyte'
digit=idx2numpy.convert_from_file(hand_image)
lb = 'train-labels.idx1-ubyte'
labels = idx2numpy.convert_from_file(lb)

test_data='t10k-images.idx3-ubyte'
test_label='t10k-labels.idx1-ubyte'
test=idx2numpy.convert_from_file(test_data)
tlabel=idx2numpy.convert_from_file(test_label)

tresult=[]
for i in range(0,len(test)):
    arr=test[i].flatten()
    tresult.append(arr)

result=[]
for i in range(0,len(digit)):
    arr=digit[i].flatten()
    result.append(arr)
result=np.array(result)

standardized_data = SS().fit_transform(result)
test_standardized_data=SS().fit_transform(tresult)


project,pca,cov,eigen=PCA(standardized_data,70)
#tproject,tpca,tcov,teigen=PCA(test_standardized_data,95)

tproject=np.dot(test_standardized_data,np.transpose(pca))


projectfda,top=FDA(standardized_data,labels)
print("\nAfter Only PCA\n")
lis,v=LDA(tproject,tlabel,project,labels)


#tprojectfda,ttop=FDA(test_standardized_data,tlabel)

print("\nAfter only FDA\n")
tprojectfda=np.dot(test_standardized_data,np.transpose(top))
lis,v=LDA(tprojectfda,tlabel,projectfda,labels)


#lis2,v=LDA(tproject,tlabel,project,labels)

projectfda1,toppca=FDA(project,labels)
print("\nAfter PCA and FDA\n")
tprojectpcafda=np.dot(tproject,np.transpose(toppca))
lis,v=LDA(tprojectpcafda,tlabel,projectfda1,labels)




