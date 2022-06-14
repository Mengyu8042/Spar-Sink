# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pylab as pl
import matplotlib.pyplot as plt
import ot
import ot.plot
from ot.datasets import make_1D_gauss as gauss
import time
from scipy.stats import uniform
from all_funcs import spar_sinkhorn, nys_sinkhorn, nys_sinkhorn2, unifNys

import warnings
warnings.filterwarnings("ignore")

#%% Generate the synthetic data
n_list = np.array([400, 800, 1600, 3200, 6400, 12800])
eps_list = np.array([0.05, 0.01, 0.005, 0.001])
d = 10
nloop = 10

time_full = np.zeros([len(n_list), len(eps_list)])
time_nys = np.zeros([nloop, len(n_list), len(eps_list)])
time_spar = np.zeros([nloop, len(n_list), len(eps_list)])


for j in range(len(n_list)):
    n1 = n_list[j]
    n2 = n_list[j]

    ## Uniform
    xs = np.empty([n1, d])
    for jj in range(d):
        xs[:,jj] = uniform.rvs(size=n1)
    xt = xs.copy()
    
    ## Gaussian distribution
    m1=n1/3; s1=n1/20
    m2=n2/2; s2=n2/20
    a = gauss(n1, m=m1, s=s1)  # m= mean, s= std
    b = gauss(n2, m=m2, s=s2)  # m= mean, s= std

    a = a/np.sum(a)
    b = b/np.sum(b)
    
    
    for k in range(len(eps_list)):
        epsilon = eps_list[k] # entropy parameter = reg
        
        #%% Compute cost and kernel matrices
        M = ot.dist(xs, xt)
        M /= np.max(M)
        K = np.exp(-M/epsilon)
        
        
        # Sinkhorn
        start1 = time.process_time()
        W = ot.sinkhorn2(a, b, M, epsilon, method='sinkhorn')
        W = np.sqrt(W)
        time_full[j,k] = time.process_time() - start1
        print("Sinkhorn: W", W)
        print("Sinkhorn: time", time_full[j,k])
        
        
        #%% Element-wise sampling
        nonzero = np.sum(K!=0)
        s = int((2**4)*np.sqrt(nonzero))
        print('nnz:',nonzero,'s:',s)
        
        
        for i in range(nloop):
            #%% Nys-Sink
            print("nys-sink: n =", n1, "eps =", epsilon, "cycle", i)
            start1 = time.process_time()
            
            R, A = unifNys(K, s)
            W_nys = nys_sinkhorn(a, b, M, R, A, stopThr=1e-4)
            
            if np.isnan(W_nys):
                K_nys = np.dot(np.dot(R, A), R.T)
                W_nys = nys_sinkhorn2(a, b, M, K_nys, stopThr=1e-4)
            
            time_nys[i,j,k] = time.process_time() - start1
            print(time_nys[i,j,k])



            #%% Spar-Sink
            print("spar-sink: n =", n1, "eps =", epsilon, "cycle", i)
            start1 = time.process_time()
            
            W_spar = spar_sinkhorn(a, b, M, K, s, stopThr=1e-4)
            
            time_spar[i,j,k] = time.process_time() - start1
            print(time_spar[i,j,k])




#%% Plot
pl.figure(5, figsize=(5, 5))
plt.title('C1', fontsize=20)

plt.plot(np.log10(n_list), np.log10(time_full[:,0]), color='grey', marker='.', linestyle='-',
         label=r'Sinkhorn ($\varepsilon$=0.05)')
plt.plot(np.log10(n_list), np.log10(time_full[:,1]), color='grey', marker='v', linestyle='--',
         label=r'Sinkhorn ($\varepsilon$=0.01)')
plt.plot(np.log10(n_list), np.log10(time_full[:,2]), color='grey', marker='x', linestyle='-.',
         label=r'Sinkhorn ($\varepsilon$=0.005)')
plt.plot(np.log10(n_list), np.log10(time_full[:,3]), color='grey', marker='o', linestyle=':',
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

# plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=17)

plt.xlabel(r'$\log_{10}$(n)', fontsize=17)
plt.ylabel(r'$\log_{10}$(time)', fontsize=17)
plt.tick_params(labelsize=17)
plt.show()
