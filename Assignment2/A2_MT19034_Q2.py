"""
Created on Sat Feb  1 22:13:58 2020

@author: Dibyendu Roy Chaudhuri
@Roll:   MT19034

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
from math import exp
import struct as st
from PIL import Image
import idx2numpy
from numpy import linalg
import matplotlib.pyplot as plt
import sklearn
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from scipy.linalg import eigh


################################# PCA #########################################


def PCA_SC(result,energy):
    
    mean=np.mean(result, axis = 0)
    
    col=len(result[0])
    normalised= np.zeros((len(result),col))
    
    for i in range(0,len(result)):
        normalised[i]=result[i]-mean
    
    covdata=np.cov(np.transpose(normalised))
    eigen,vector = eigh(covdata)
    
    eigensum=np.sum(eigen)
    current=0.0
    index=0
    for i in range(783,-1,-1):
        current+=eigen[i]
        if(current>=energy/100*(eigensum)):
            index=i
            break
    vector=vector.T
    index=180
    pca=vector[(index):]
    projection=np.dot(normalised,np.transpose(pca))
   
    
    return projection,pca,covdata


hand_image='train-images.idx3-ubyte'
digit=idx2numpy.convert_from_file(hand_image)
lb = 'train-labels.idx1-ubyte'
labels = idx2numpy.convert_from_file(lb)

result=[]
for i in range(0,len(digit)):
    arr=digit[i].flatten()
    result.append(arr)
result=np.array(result)

mean=0
lamda=80
noise=np.random.normal(mean,lamda,[len(result),len(result[0])])
result=result+noise
project,vector,c=PCA_SC(result,95)

p=np.dot(vector,np.transpose(vector))
x=np.dot(np.linalg.pinv(p),vector)
x=np.dot(project,x)


################################# IMAGE DISPLAY ###############################

img = Image.fromarray(x[200].reshape(28,28))
img1 = Image.fromarray(result[200].reshape(28,28))

#plt.imshow(result[238].reshape(28,28),cmap='gray')
#plt.imshow(x[238].reshape(28,28),cmap='gray')

img.show()
img1.show()
