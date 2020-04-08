# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 23:11:17 2020

@author: Dibyendu
@Rollno: MT19034   
"""

import numpy as np
import math
import random
from tqdm import tqdm
import codecs



def mean(data):
    
    l=len(data)
    sum=0.0
    for i in range(l):
        sum+=data[i]
    sum=sum/l

    var=0.0
    for i in range(l):
        var+=(data[i]-sum)**2
    var=var/l
    
    
    return var


def read(file):
	fp = codecs.open(file,"r",encoding='utf-8', errors='ignore')
	text = fp.read()
	return text

def RSS(ytar,ypred):
    mse=(ytar-ypred)**2
    return mse

def node_rss(data,tol):
    
    l=len(data)
    mse=0
    pred=0
    for i in range(l):
        pred+=data[i][5]
    if l!=0:
        pred=pred/l
    for i in range(l):
        mse+=RSS(data[i][5],pred)
     
    if tol!=0:
        mse=(mse*l)/tol
    
    return [mse,pred]
        
        

def bulid_tree(data,dept=1,n_freature=11,limit=1000):
    
    ignore=[4]
    fre=[0,1,2,3,5,6,7,8,9,10,11]
    random.shuffle(fre)
    ignore.extend(fre[n_freature:])
    
    tol=len(data)
    mid=node_rss(data,tol)
    mse=mid[0]
    pred=mid[1]
    
    if dept>=limit:
        return [pred]
    
    #print(mse," ",dept)
    
    a=[[] for i in range(1,len(data[0]))]
    for i in range(len(data)):
        for j in range(1,len(a)+1):
            if data[i][j] not in a[j-1]:
                a[j-1].append(data[i][j])
    for j in range(len(a)):
        a[j].sort()
        
    val=[]
    mse1=0.0
    mse2=0.0
    info=0.0
    for i in range(len(a)):
        if i not in ignore:
            for k in a[i]:
                left=[]
                right=[]
                for j in range(len(data)):
                    if data[j][i+1]<=k:
                        left.append(data[j])
                    else:
                        right.append(data[j])
                mse1=node_rss(left,len(left))[0]
                mse2=node_rss(right,len(right))[0]
                info1=mse-(mse1+mse2)
                if info1>info:
                    info=info1
                    val=[i,k]
    if info>0:
        left=[]
        right=[]
        for i in range(len(data)):
            if data[i][val[0]+1]<=val[1]:
                left.append(data[i])
            else:
                right.append(data[i])
        
        rule=[val,pred,mse]
        s=[rule,bulid_tree(left,dept+1,n_freature,limit),bulid_tree(right,dept+1,n_freature,limit)]
        return s
    
    else:
        return [pred]
    
    

def decide(data,s,depth,maxdept):
    
    if len(s)==1:
        m=0
        index=0
        
        mse=s[0]
        
        arr=[]
        for i in range(len(data)):
            arr.append([data[i][0],mse])
        
        return arr
                
            
            
    
    left=[]
    right=[]
    
    for i in range(len(data)):
        if data[i][s[0][0][0]+1] <= s[0][0][1]:
            left.append(data[i])
        else:
            right.append(data[i])
    
    arr=decide(left,s[1],depth+1,maxdept)
    arr1=decide(right,s[2],depth+1,maxdept)
    
    a=[]
    
    a.extend(arr)
    a.extend(arr1)
    
    return a

def decision_tree(train,test,dept):
    
    s=bulid_tree(train,1,limit=dept )
    print("Train is completed")
    pred=decide(test,s,0,0)
    pred.sort()
    cou=0
    tol=0
    for i in range(len(pred)):
        if pred[i][0]==test[i][0]:
            tol+=1
            cou+=RSS(test[i][5],pred[i][1])
    print("MSE- ",cou/tol)
    
def Bagged(train,test,b,part,dept):
    print("Training .....")
    s1=[]
    for i in tqdm(range(b)):
        random.shuffle(train)
        siz=int(len(train)*(part))
        boostrap=[]
        if b>1:
            bt=np.random.randint(low=0,high=len(train),size=siz)
            for k in bt:
                boostrap.append(train[k])
                
        s1.append(bulid_tree(boostrap,1,11,limit=dept))
    print("Test .....")
    
    o1=[]
    for i in tqdm(range(b)):
        o=decide(test,s1[i],0,100)
        o.sort()
        o1.append(o)
    print("Test Complete")
    cou=0
    tol=0
    error=0
    er=[]
    for i in range(len(test)):
        mse_avg=0.0
        tol+=1
        for j in range(b):
            mse_avg+=o1[j][i][1]
        mse_avg=mse_avg/b
        cou+=RSS(test[i][5],mse_avg)
        error+=abs(test[i][5]-mse_avg)
        er.append(abs(test[i][5]-mse_avg))
    print("MSE- ",cou/tol)
    print("Mean Absoulte error- ",error/tol)
    print("SD of error- ",math.sqrt(mean(er)))
    
    
    
def boostrap(train,test,b,part,dept,n_freature=11):
    print("Training ......")
    s1=[]
    o1=[]
    
    fre=[1,2,3,4,6,7,8,9,10,11,12]
    
    for i in tqdm(range(b)):
        random.shuffle(train)
     
        random.shuffle(fre)
        a=fre.copy()
        randomt=[]
        randomtest=[]
        
        for j in range(len(train)):
            data=[train[j][0],train[j][a[0]],train[j][a[1]],train[j][a[2]],train[j][a[3]]]
            data.append(train[j][5])
            for k in range(4,len(a)):
                data.append(train[j][a[k]])
            randomt.append(data)
        siz=int(len(train)*(part))
        
        boostrap=[]
        if b>1:
            bt=np.random.randint(low=0,high=len(train),size=siz)
            for k in bt:
                boostrap.append(randomt[k])
        else:
            boostrap=randomt[:siz ]
        
        for j in range(len(test)):
            data=[test[j][0],test[j][a[0]],test[j][a[1]],test[j][a[2]],test[j][a[3]]]
            data.append(test[j][5])
            for k in range(4,len(a)):
                data.append(test[j][a[k]])
            
            randomtest.append(data)
        
        s2=bulid_tree(boostrap,1,n_freature,limit=dept)
        o=decide(randomtest,s2,0,100)
        o.sort()
        o1.append(o)
        #print(boostrap)
        s1.append(s2)
    #print(len(s1))
    print("Test Complete")
    
    cou=0
    tol=0
    error=0
    er=[]
    for i in range(len(test)):
        mse_avg=0.0
        tol+=1
        for j in range(b):
            mse_avg+=o1[j][i][1]
        mse_avg=mse_avg/b
        cou+=RSS(test[i][5],mse_avg)
        error+=abs(test[i][5]-mse_avg)
        er.append(abs(test[i][5]-mse_avg))
    print("MSE- ",cou/tol)
    print("Mean Absoulte error- ",error/tol)
    print("SD of error- ",math.sqrt(mean(er)))
    
    

data=input("Enter for using default dataset or give the path ") 
path='PRSA_data_2010.1.1-2014.12.31.csv'
if len(data.strip())>1:
    path=data.strip()
    
text=read(path)
data=text.split('\r\n')
data=data[1:len(data)-1]
train=[]
test=[]
val={}
c=1

pm=0
p=0
for i in tqdm( range(len(data)) ):
    a=data[i].split(',')
    if a[5]!='NA':
        pm+=float(a[5])
        p+=1
        
    if a[9] not in val.keys():
        val[a[9]]=c
        a[9]=c
        c+=1
    else:
        a[9]=val[a[9]]
    if a[1]=='2011' or a[1]=='2013':
        test.append(a)
    else:
        train.append(a)

for i in tqdm( range(len(train)) ):
    a=train[i]
    if a[5]=='NA':
        a[5]=pm/p
    a=[float(i) for i in a]
    train[i]=a

for i in tqdm( range(len(test)) ):
    a=test[i]
    if a[5]=='NA':
        a[5]=pm/p
    a=[float(i) for i in a]
    test[i]=a
    
#decision_tree(train,test,8)
print()
#Bagged(train,test,10,0.7,8)

print("Decision Tree Prediction- ")
boostrap(train,test,1,1,8)
print()
print("Bagged Tree Prediction- ")
#boostrap(train,test,10,.7,10)
Bagged(train,test,15,0.7,8)
print()
print("Random Forest Prediction- ")
boostrap(train,test,20,.7,8,4)
print()

