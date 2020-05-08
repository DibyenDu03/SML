# -*- coding: utf-8 -*-
"""
Created on Fri May  1 20:51:39 2020

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

store_model=pickle.load(open('MT19034_model5.sav','rb'))
o=store_model.predict(test)


id='id'
label='category'
st=id+","+label+"\n"
for i in range(len(o)):
    st+=Test[i][0]+","+str(o[i])+"\n"

file1 = open("MT19034_Dibyendu_submission.csv","a") 
file1.write(st)
file1.close()