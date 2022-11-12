# -*- coding: utf-8 -*-
"""
Predict the ED time point using Sinkhorn, Spar-Sink, Rand-Sink, and Nys-Sink

    a) for original scale (112*112), set n_resize = 1 
    
    b) for mean-pooling (56*56), set n_resize = 2

"""

import ot
import time
import cv2
import numpy as np
import pandas as pd
import ot.plot
from all_funcs import wfr_kernel, wfr_distance, spar_sinkhorn_unbalanced
from all_funcs import unifNys, nys_sinkhorn_unbalanced, nys_sinkhorn_unbalanced2
import json
import os

import random
random.seed(123)

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

n = 20
Name = random.sample(Name, n)


# =============================================================================
# 
# Read avi
# 
# =============================================================================
n_resize = 1
video_temp = [None]*len(Name)
videopath = os.path.join(data_path, '200data/')


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


# =============================================================================
# 
# Read ground truth
# 
# =============================================================================
gt = pd.read_csv(os.path.join(data_path, 'VolumeTracings.csv'))

gt2 = np.zeros((n,2))

for i in range(n):

    v_name = Name[i]
    gtt = gt.query('FileName==@v_name')
    v_beat = gtt.Frame.values
    gt2[i,0] = v_beat[0]
    gt2[i,1] = v_beat[-1]
    
    
thres = 30

X_region = makeLRGrid(vp,vp)



# =============================================================================
# 
# Compute WFR and predict ED
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
s0 = 0.001*nn*np.log(nn)**4
s_list = np.array([1, 2, 2**2, 2**3]) * s0  # subsample size
s_list = s_list.astype(np.int)


#%% Sinkhorn
accu_sink = np.zeros(n)
time_sink = np.zeros(n)

for i in range(n):
    print('Sinkhorn: sample', i)
    stime = time.process_time() 

    bg = int(gt2[i,0])
    ed = int(gt2[i,1])

    a = video_temp[i][bg].flatten()
    a = np.array(a, dtype='uint16') + 1
    w1 = a/np.sum(a)
    high = np.zeros(thres)

    for j in range(bg, min(video_temp[i].shape[0],(bg+thres))):

        b = video_temp[i][j].flatten()
        b = np.array(b, dtype='uint16') + 1
        w2 = b/np.sum(b)
        
        G = ot.unbalanced.sinkhorn_unbalanced(w1, w2, M, epsilon, reg_kl)
        WFR = wfr_distance(w1, w2, M, K, G, epsilon, reg_kl)

        high[int(j-bg)] = WFR
    
    high_id = np.argmax(high)
    accu_sink[i] = abs(high_id/(ed-bg)-1)
    time_sink[i] = time.process_time() - stime
    print('acc', accu_sink[i], 'time', time_sink[i])

print('Sinkhorn: acc_mean', np.nanmean(accu_sink))
print('Sinkhorn: acc_std', np.nanstd(accu_sink))
print('Sinkhorn: time_mean', np.nanmean(time_sink))



#%% Spar-Sink
accu_spar = np.zeros([n, len(s_list)])
time_spar = np.zeros([n, len(s_list)])

for i in range(n):
    print('Spar-Sink: sample', i)
    
    bg = int(gt2[i,0])
    ed = int(gt2[i,1])

    a = video_temp[i][bg].flatten()
    a = np.array(a, dtype='uint16') + 1
    w1 = a/np.sum(a)
    high = np.zeros(thres)

    k = 0
    for s in s_list:
        print('s', s)
        stime = time.process_time() 

        for j in range(bg, min(video_temp[i].shape[0],(bg+thres))):
    
            b = video_temp[i][j].flatten() 
            b = np.array(b, dtype='uint16') + 1
            w2 = b/np.sum(b)
            
            WFR, _ = spar_sinkhorn_unbalanced(w1, w2, M, K, s, epsilon, reg_kl, 'spar-sink')
    
            high[int(j-bg)] = WFR
        
        high_id = np.argmax(high)
        accu_spar[i,k] = abs(high_id/(ed-bg)-1)
        time_spar[i,k] = time.process_time() - stime
        print('acc', accu_spar[i,k], 'time', time_spar[i,k])
        k += 1

print('Spar-Sink: acc_mean', np.nanmean(accu_spar, axis=0))
print('Spar-Sink: acc_std', np.nanstd(accu_spar, axis=0))
print('Spar-Sink: time_mean', np.nanmean(time_spar, axis=0))



#%% Rand-Sink
accu_rand = np.zeros([n, len(s_list)])
time_rand = np.zeros([n, len(s_list)])

for i in range(n):
    print('Rand-Sink: sample', i)
    
    bg = int(gt2[i,0])
    ed = int(gt2[i,1])

    a = video_temp[i][bg].flatten()
    a = np.array(a, dtype='uint16') + 1
    w1 = a/np.sum(a)
    high = np.zeros(thres)

    k = 0
    for s in s_list:
        print('s', s)
        stime = time.process_time() 

        for j in range(bg, min(video_temp[i].shape[0],(bg+thres))):
    
            b = video_temp[i][j].flatten() 
            b = np.array(b, dtype='uint16') + 1
            w2 = b/np.sum(b)
            
            WFR, _ = spar_sinkhorn_unbalanced(w1, w2, M, K, s, epsilon, reg_kl, 'rand-sink')
    
            high[int(j-bg)] = WFR
        
        high_id = np.argmax(high)
        accu_rand[i,k] = abs(high_id/(ed-bg)-1)
        time_rand[i,k] = time.process_time() - stime
        print('acc', accu_rand[i,k], 'time', time_rand[i,k])
        k += 1

print('Rand-Sink: acc_mean', np.nanmean(accu_rand, axis=0))
print('Rand-Sink: acc_std', np.nanstd(accu_rand, axis=0))
print('Rand-Sink: time_mean', np.nanmean(time_rand, axis=0))



#%% Nys-Sink
accu_nys = np.zeros([n, len(s_list)])
time_nys = np.zeros([n, len(s_list)])

for i in range(n):
    print('Nys-Sink: sample', i)
    
    bg = int(gt2[i,0])
    ed = int(gt2[i,1])

    a = video_temp[i][bg].flatten()
    a = np.array(a, dtype='uint16') + 1
    w1 = a/np.sum(a)
    high = np.zeros(thres)

    k = 0
    for s in s_list:
        print('s', s)
        stime = time.process_time() 

        for j in range(bg, min(video_temp[i].shape[0],(bg+thres))):
    
            b = video_temp[i][j].flatten() 
            b = np.array(b, dtype='uint16') + 1
            w2 = b/np.sum(b)
            
            R, A = unifNys(K, s)
            WFR, _ = nys_sinkhorn_unbalanced(a, b, M, R, A, epsilon, reg_kl)
            
            if np.isnan(WFR):
                K_nys = np.dot(np.dot(R, A), R.T)
                WFR, _ = nys_sinkhorn_unbalanced2(a, b, M, K_nys, epsilon, reg_kl)

            high[int(j-bg)] = WFR
        
        high_id = np.argmax(high)
        accu_nys[i,k] = abs(high_id/(ed-bg)-1)
        time_nys[i,k] = time.process_time() - stime
        print('acc', accu_nys[i,k], 'time', time_nys[i,k])
        k += 1

print('Nys-Sink: acc_mean', np.nanmean(accu_nys, axis=0))
print('Nys-Sink: acc_std', np.nanstd(accu_nys, axis=0))
print('Nys-Sink: time_mean', np.nanmean(time_nys, axis=0))

