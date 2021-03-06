# -*- coding: utf-8 -*-
"""
Created on Fri May  1 16:38:15 2020

@author: Dibyendu
"""

from tqdm import tqdm
import numpy as np
from skimage import io 
from skimage.color import rgb2gray
from imblearn.over_sampling import SMOTE
import pickle
from sklearn.ensemble import BaggingClassifier
from sklearn.neural_network import MLPClassifier
import random
import string


dbfile = open('Dataset_Kaggle', 'rb')      
data_set = pickle.load(dbfile) 
dbfile.close()

train=data_set['Train']
Test=data_set['Test']

x=[]
y=[]

x1=[]
y1=[]

X=[]
Y=[]

ftrain=463

random.shuffle(train)

for i in tqdm(tqdm(range(len(train)))):
    
    Y.append(train[i][2])
    X.append(train[i][1][:ftrain])
    
    
test=[]

for i in tqdm(tqdm(range(len(Test)))):
    
    test.append(Test[i][1][:ftrain])
    
opt=input('Do want to re-trained the model(yes/no)-\t')
if opt.lower()=='yes' or opt.lower()=='y':
    
    le=int(len(X)*0.8)

    for i in tqdm(tqdm(range(le))):
        
        x.append(X[i])
        y.append(Y[i])
    
    for i in tqdm(range(le,len(X))):
    
        x1.append(X[i])
        y1.append(Y[i])
    
    print()
    print('Smote started')
    x,y=SMOTE().fit_resample(x, y)
    print('Smote completed')
    print()
    
    clf = MLPClassifier(alpha=1e-5, hidden_layer_sizes=(150,200,100), random_state=1)
    clf = BaggingClassifier(base_estimator=clf,n_estimators=25,random_state=0,verbose=1)
    
    clf.fit(x,y)
    print((clf.score(x1,y1))*100)
    
    print()
    print('Smote started')
    X,Y=SMOTE().fit_resample(X,Y)
    print('Smote completed')
    print()
    
    clf.fit(X,Y)
    
    o=clf.predict(test)
    
    file='weight_new.sav'
    pickle.dump(clf,open(file,'wb'))
    
else:    

    store_model=pickle.load(open('MT19034_model2.sav','rb'))
    o=store_model.predict(test)


id='id'
label='category'
st=id+","+label+"\n"
for i in range(len(o)):
    st+=Test[i][0]+","+str(o[i])+"\n"

file1 = open("MT19034_Dibyendu_submission.csv","a") 
file1.write(st)
file1.close()


