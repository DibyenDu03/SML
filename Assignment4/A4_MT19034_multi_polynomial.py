# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 03:17:55 2020

@author: Dibyendu
"""

def sort_l(str1):
    s=str1.split('_')
    s.sort()
    s1=''
    for i in range(0,len(s)):
        if(i!=0):
            s1=s1+"_"+s[i]
        else:
            s1=s[i]
    return s1


def create_final(A,p):
    pow1=p
    str1=[]
    strc=[]
    str11=[]
    dic={}
    for i1 in range(0,len(A)):
        str2='a'+str(i1)
        str1.append(str2)
        strc.append(str2)
        str11.append(str2)
        dic[str2]=A[i1]
        
    
    for i1 in range(1,pow1):
        final=[]
        for j in range(0,len(str11)):
            for k in range(0,len(strc)):
                st2=str11[j]+"_"+strc[k]
                final.append(st2)
        str1.extend(final)
        str11=final
    
    final=[]
    for i in str1:
        final.append(sort_l(i))
    final.sort()
    final=list(set(final))
    final.sort(key=len)
    res=[]
    #print(final)
    for i1 in final:
        sf=i1.split('_')
        m=1.0
        for j in sf:
            m=m*dic[j]
        res.append(m)
    return res,len(final)
    
A=[1,2,3,4,5,6,7,8,9,0,1,2,3]
r,l=create_final(A,3)
#print(r,"\n",l)

    
    
