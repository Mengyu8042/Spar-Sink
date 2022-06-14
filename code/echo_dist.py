# -*- coding: utf-8 -*-
"""
Compute the WFR distance matrix using Spar-Sink

Three cases: health, heart failure, and arrhythmia
"""

import cv2
import numpy as np
from all_funcs import wfr_kernel, spar_sinkhorn_unbalanced
import json
import os
from scipy.io import savemat

import random
random.seed(1234)

import warnings
warnings.filterwarnings("ignore")


def makeLRGrid(a,b):
    temp = []
    for i in range(a):
        for j in range(b):
            temp.append([i,j])
    return np.array(temp)

def qusimc(L):
    res = np.zeros((2,L))
    for i in range(L):
        res[0,i] = np.cos(np.pi/L*i)
        res[1,i] = np.sin(np.pi/L*i)

    return res

def qusimc_slice(X_region, L):
    slice_dir = qusimc(L)
    slice_point = X_region@slice_dir
    res1 = np.zeros((slice_point.shape[0],L),dtype=np.double)
    res2 = np.zeros((slice_point.shape[0],L),dtype=np.int)

    for i in range(L):
        perm = np.argsort(slice_point[:,i])
        res1[:,i] = slice_point[:,i][perm]
        res2[:,i] = perm
    
    return res1, res2


data_path = os.path.join(os.path.dirname(os.getcwd()), 'data/')

a = open(os.path.join(data_path, 'Name.txt'), 'r', encoding='UTF-8')
Name = a.read()
Name = json.loads(Name)

n = 1
case = 'health'  # 'health' or 'failure' or 'arrhythmia'

if case == 'health':
    Name = ['0X1A05DFFFCAFB253B.avi']
elif case == 'failure':
    Name = ['0X1DBFE6B5E123C051.avi']
else:
    Name = ['0X1EFF5D4EF0A8CAF8.avi']



# =============================================================================
# 
# Read avi
# 
# =============================================================================
n_resize = 1
video_temp = [None]*len(Name)
videopath = os.path.join(data_path, 'three_case/')


for k in range(len(Name)):
    cap = cv2.VideoCapture(videopath+Name[k])
    frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_temp[k] = np.zeros((frameCount, int(frameWidth/n_resize), int(frameWidth/n_resize)), dtype='uint8')

    fc = 0
    ret = True

    while (fc < frameCount  and ret):
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_shrink = cv2.resize(gray, None, fx = 1/n_resize, fy = 1/n_resize)
        video_temp[k][fc] = gray_shrink
        fc += 1


vp = int(112/n_resize)

X_region = makeLRGrid(vp, vp)



# =============================================================================
# 
# Compute WFR via Spar-Sink
# 
# =============================================================================
epsilon = 0.01
eta = 15
reg_kl = 1

K = wfr_kernel(X_region, X_region, eta, epsilon)
M = -epsilon*np.log(K)
M /= np.max(M[K!=0])
K[K!=0] = np.exp(-M[K!=0]/epsilon)
M[K==0] = -epsilon*np.log(1e-100)

nn = K.shape[0]
sparsity = np.sum(K==0)/(nn**2)
nonzero = np.sum(K!=0)
nnzSqrt = int(np.sqrt(nonzero))
s = int((2**5)*nnzSqrt)


video_sample = video_temp[0]
len_resize = 3
video_len = int(video_sample.shape[0]/len_resize)


dist_mat = np.zeros((video_len, video_len))

for i in range(0, video_len-1):
    print('i: ', i)

    a = video_sample[len_resize*i].flatten()
    a = np.array(a, dtype='uint16') + 1
    w1 = a/np.sum(a)

    for j in range(i+1, video_len):
    
        b = video_sample[len_resize*j].flatten() 
        b = np.array(b, dtype='uint16') + 1
        w2 = b/np.sum(b)
        
        WFR = spar_sinkhorn_unbalanced(w1, w2, M, K, s, epsilon, reg_kl)

        dist_mat[i,j] = WFR


dist_mat += dist_mat.T - np.diag(dist_mat.diagonal())


mdict = {'dist_mat': dist_mat}
output_path = os.path.join(os.path.dirname(os.getcwd()), 'output/')
savemat(os.path.join(output_path, 'dist_mat_%s.mat'%case), mdict)