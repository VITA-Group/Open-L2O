#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 18 17:10:52 2019

@author: Yue Cao
"""
import numpy as np
#import pdb




def data_loader():
    
    scoor_init=[]
    sq=[]
    se=[]
    sr=[]
    sbasis=[]
    seval=[]
    protein_list = np.loadtxt("train_list", dtype='str')

    for i in range(len(protein_list)):

        #if(i+1 ==len(protein_list)):
        #    n=9
        #else:
        #    n=6
        n=6

        for j in range(1,n):
            x = np.loadtxt("data/"+protein_list[i]+'_'+str(j)+"/coor_init")
            q = np.loadtxt("data/"+protein_list[i]+'_'+str(j)+"/q")
            e = np.loadtxt("data/"+protein_list[i]+'_'+str(j)+"/e")
            r = np.loadtxt("data/"+protein_list[i]+'_'+str(j)+"/r")
            basis = np.loadtxt("data/"+protein_list[i]+'_'+str(j)+"/basis")
            eigval = np.loadtxt("data/"+protein_list[i]+'_'+str(j)+"/eigval")
            
            #print (x.shape, q.shape, e.shape, r.shape, basis.shape, eigval.shape)
            

            q=np.tile(q, (1, 1))
            e=np.tile(e, (1,1))

            q = np.matmul(q.T, q)
            e = np.sqrt(np.matmul(e.T, e))
            r = (np.tile(r, (len(r), 1)) + np.tile(r, (len(r), 1)).T)/2


            scoor_init.append(x)
            sq.append(q)
            se.append(e)
            sr.append(r)
            sbasis.append(basis)
            seval.append(eigval)

   

    scoor_init = np.array(scoor_init)
    sq = np.array(sq)
    se = np.array(se)
    sr = np.array(sr)
    sbasis = np.array(sbasis)
    seval = np.array(seval)
    print (sq.shape, se.shape, seval.shape)
    return scoor_init, sq, se, sr, sbasis, seval

if __name__ == "__main__":
    data_loader()




