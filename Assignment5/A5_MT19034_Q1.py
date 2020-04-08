# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 19:50:28 2020

@author: Dibyendu
@Rollno: MT19034
"""

import numpy as np
import math
import random
from tqdm import tqdm
import codecs

def gini_index(no,tol):
    
    #print(no," ",tol)
    px=0.0 
    if tol!=0:
        px=no/tol*(tol-no)/tol
    
    return px

def gini_index1(n1,tol):
    
    sum=0.0
    for i in range(12):
        sum+=n1[i]
    px=0.0
    for i in range(12):
        px+=gini_index(n1[i],sum)
    
    px=sum/tol*px
        
    return px

def read(file):
	fp = codecs.open(file,"r",encoding='utf-8', errors='ignore')
	text = fp.read()
	return text

def bulid_tree(data,dept,limit,n_freature=11):
    
    ignore=[1]
    fre=[0,2,3,4,5,6,7,8,9,10,11]
    random.shuffle(fre)
    ignore.extend(fre[n_freature:])
    
    a=[[] for i in range(1,len(data[0]))]
    cl=[0.0 for i in range(12)]
    for i in range(len(data)):
        for j in range(1,len(a)+1):
            if data[i][j] not in a[j-1]:
                a[j-1].append(data[i][j])
        cl[int(data[i][2])-1]+=1.0
        
    for j in range(len(a)):
        a[j].sort()
    
    if dept>=limit:
        return [cl]
        
    #print(a)
        
    tol=0.0
    l=len(data)
    for i in range(len(cl)):
        tol+=gini_index(cl[i],l)
        
    #print(dept,tol)
        
    val=[]
    info=0.0
    for i in range(len(a)):
        if i not in ignore:
            for k in a[i]:
                cl1=[0.0 for s in range(12)]
                for j in range(len(data)):
                    if data[j][i+1]<=k:
                        cl1[int(data[j][2])-1]+=1.0
                cl2=[]
                for s in range(12):
                    cl2.append(cl[s]-cl1[s])
                g1=gini_index1(cl2,l)
                g2=gini_index1(cl1,l)
                info1=tol-(g1+g2)
                if info1>info:
                    info=info1
                    val=[i,k]
                    
    if info==0.0:
        return [cl]
    else:
        left=[]
        right=[]
        for i in range(len(data)):
            if data[i][val[0]+1]<=val[1]:
                left.append(data[i])
            else:
                right.append(data[i])
        
        rule=[val,cl,tol]
        s=[rule,bulid_tree(left,dept+1,limit,n_freature),bulid_tree(right,dept+1,limit,n_freature)]
            
    return s

def decide(data,s,depth,maxdept):
    
    if len(s)==1:
        m=0
        index=0
        cl=s[0]
        #print(cl," ",depth)
        for i in range(len(cl)):
            if m<cl[i]:
                m=cl[i]
                index=i
        arr=[]
        for i in range(len(data)):
            arr.append([data[i][0],index+1])
        
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


### Decision_tree

def Decision_tree(train,test,dept):
    
    print("Training .....")
    s=bulid_tree(train,1,limit=dept)
    print("Test .....")
    o=decide(test,s,0,100)
    o.sort()
    
    
    cou=0
    tol=0
    for i in range(len(o)):
        if o[i][0]==test[i][0]:
            tol+=1
            if o[i][1]==test[i][2]:
                cou+=1
                
    print()
    print(cou/tol*100)
    



#### BAGGING

def Bagged(train,test,b,part,dept):
    print("Training .....")
    s1=[]
    for i in tqdm(range(b)):
        random.shuffle(train)
        siz=int(len(train)*(part))
        boostrap=train[:siz ]
        s1.append(bulid_tree(boostrap,1,limit=dept))
    print("Test .....")
    
    o1=[]
    for i in tqdm(range(b)):
        o=decide(test,s1[i],0,100)
        o.sort()
        o1.append(o)
    print("Test Complete")
    
    cou=0
    
    for i in range(len(test)):
        a=[0 for i in range(12)]
        for j in range(b):
                a[ int(o1[j][i][1])-1]+=1
        m=0
        ind=0
        for j in range(12):
            if a[j]>m:
                m=a[j]
                ind=j
        if ind+1==test[i][2]:
            cou+=1
    
    print(cou/len(test)*100)
    
def boostrap(train,test,b,part,dept,n_freature=11):
    print("Training ......")
    s1=[]
    o1=[]
    fre=[1,3,4,5,6,7,8,9,10,11,12]
    for i in tqdm(range(b)):
        random.shuffle(train)
        random.shuffle(fre)
        a=fre.copy()
        randomt=[]
        randomtest=[]
        
        for j in range(len(train)):
            data=[train[j][0],train[j][a[0]],train[j][2]]
            for k in range(1,len(a)):
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
            data=[test[j][0],test[j][a[0]],test[j][2]]
            for k in range(1,len(a)):
                data.append(test[j][a[k]])
            randomtest.append(data)
        
        s2=bulid_tree(boostrap,1,dept,n_freature)
        o=decide(randomtest,s2,0,100)
        o.sort()
        o1.append(o)
        #print(boostrap)
        s1.append(s2)
    #print(len(s1))
    print("Test Complete")
    
    cou=0
    
    for i in range(len(test)):
        a=[0 for i in range(12)]
        for j in range(b):
                a[ int(o1[j][i][1])-1]+=1
        m=0
        ind=0
        for j in range(12):
            if a[j]>m:
                m=a[j]
                ind=j
        if ind+1==test[i][2]:
            cou+=1
    
    print(cou/len(test)*100)
    
        
                
                
        

    
    
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


#Decision_tree(train,test,8)
print()
#Bagged(train,test,10,0.7,8)

print("Decision Tree Prediction- ")
boostrap(train,test,1,1,8)
print()
print("Bagged Tree Prediction- ")
boostrap(train,test,10,.7,8)
print()
print("Random Forest Prediction- ")
boostrap(train,test,20,.7,8,4)
print()
