# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import ot
import ot.plot
from ot.datasets import make_1D_gauss as gauss
import time
from scipy.stats import uniform
from all_funcs import spar_sinkhorn, nys_sinkhorn, nys_sinkhorn2, unifNys

import warnings
warnings.filterwarnings("ignore")

import os
os.getcwd()
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"



#%% Generate the synthetic data
n_list = np.array([800, 1600, 3200])
eps_list = np.array([0.05, 0.01, 0.005, 0.001])
d = 10  # dimension of support points
nloop = 6  # number of replications for subsampling methods
nloop_sink = 3  # number of replications for Sinkhorn

time_sink = np.zeros([nloop_sink, len(n_list), len(eps_list)])
time_nys = np.zeros([nloop, len(n_list), len(eps_list)])
time_spar = np.zeros([nloop, len(n_list), len(eps_list)])


for i in range(nloop):
    for j in range(len(n_list)):
        n1 = n_list[j]  # sample size (number of support points)
        n2 = n_list[j]
        s = int(0.005*n1*np.log(n1)**4)  # subsample size
        print('cycle', i, 'n =', n1)
    
        ## Uniform
        xs = np.empty([n1, d])
        for jj in range(d):
            xs[:,jj] = uniform.rvs(size=n1)
        xt = xs.copy()
        
        ## Gaussian distribution
        m1=n1/3; s1=n1/20
        m2=n2/2; s2=n2/20
        a = gauss(n1, m=m1, s=s1)
        b = gauss(n2, m=m2, s=s2)
        a = a/np.sum(a)
        b = b/np.sum(b)
        
        
        for k in range(len(eps_list)):
            epsilon = eps_list[k] # entropy parameter
            print('epsilon =', epsilon)
            
            #%% Compute cost and kernel matrices
            M = ot.dist(xs, xt)
            M /= np.max(M)
            K = np.exp(-M/epsilon)
            
            
            #%% Sinkhorn
            if i < nloop_sink:
                start1 = time.process_time()
                W = ot.sinkhorn2(a, b, M, epsilon, method='sinkhorn', stopThr=1e-10)
                W = np.sqrt(W)
                time_sink[i,j,k] = time.process_time() - start1
            

            #%% Nys-Sink
            start1 = time.process_time()
            R, A = unifNys(K, s)
            W_nys, _ = nys_sinkhorn(a, b, M, R, A)
            if np.isnan(W_nys):
                K_nys = np.dot(np.dot(R, A), R.T)
                W_nys, _ = nys_sinkhorn2(a, b, M, K_nys)
            
            time_nys[i,j,k] = time.process_time() - start1


            #%% Spar-Sink
            start1 = time.process_time()
            W_spar, _ = spar_sinkhorn(a, b, M, K, s, method='spar-sink', sampling='marginal')
            time_spar[i,j,k] = time.process_time() - start1
            


plt.figure(1, figsize=(5, 5))
plt.title('C1', fontsize=24)

plt.plot(np.log10(n_list), np.log10(np.nanmean(time_sink[:,:,0], axis=0)), color='grey', marker='.', linestyle='-',
         label=r'Sinkhorn ($\varepsilon$=0.05)')
plt.plot(np.log10(n_list), np.log10(np.nanmean(time_sink[:,:,1], axis=0)), color='grey', marker='v', linestyle='--',
         label=r'Sinkhorn ($\varepsilon$=0.01)')
plt.plot(np.log10(n_list), np.log10(np.nanmean(time_sink[:,:,2], axis=0)), color='grey', marker='x', linestyle='-.',
         label=r'Sinkhorn ($\varepsilon$=0.005)')
plt.plot(np.log10(n_list), np.log10(np.nanmean(time_sink[:,:,3], axis=0)), color='grey', marker='o', linestyle=':',
          label=r'Sinkhorn ($\varepsilon$=0.001)')

plt.plot(np.log10(n_list), np.log10(np.nanmean(time_nys[:,:,0], axis=0)), color='y', marker='.', linestyle='-',
         label=r'Nys-Sink ($\varepsilon$=0.05)')
plt.plot(np.log10(n_list), np.log10(np.nanmean(time_nys[:,:,1], axis=0)), color='y', marker='v', linestyle='--',
         label=r'Nys-Sink ($\varepsilon$=0.01)')
plt.plot(np.log10(n_list), np.log10(np.nanmean(time_nys[:,:,2], axis=0)), color='y', marker='x', linestyle='-.',
         label=r'Nys-Sink ($\varepsilon$=0.005)')
plt.plot(np.log10(n_list), np.log10(np.nanmean(time_nys[:,:,3], axis=0)), color='y', marker='o', linestyle=':',
          label=r'Nys-Sink ($\varepsilon$=0.001)')

plt.plot(np.log10(n_list), np.log10(np.nanmean(time_spar[:,:,0], axis=0)), color='red', marker='.', linestyle='-',
         label=r'Spar-Sink ($\varepsilon$=0.05)')
plt.plot(np.log10(n_list), np.log10(np.nanmean(time_spar[:,:,1], axis=0)), color='red', marker='v', linestyle='--',
         label=r'Spar-Sink ($\varepsilon$=0.01)')
plt.plot(np.log10(n_list), np.log10(np.nanmean(time_spar[:,:,2], axis=0)), color='red', marker='x', linestyle='-.',
         label=r'Spar-Sink ($\varepsilon$=0.005)')
plt.plot(np.log10(n_list), np.log10(np.nanmean(time_spar[:,:,3], axis=0)), color='red', marker='o', linestyle=':',
          label=r'Spar-Sink ($\varepsilon$=0.001)')

# plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=20)

plt.xlabel(r'$\log_{10}$(n)', fontsize=20)
plt.ylabel(r'$\log_{10}$(time)', fontsize=20)
plt.tick_params(labelsize=20)
plt.show()
# plt.savefig('time_ot.png', dpi=200, bbox_inches='tight')
# plt.close('all')

