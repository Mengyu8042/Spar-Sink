# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pylab as pl
import matplotlib.pyplot as plt
import ot
import ot.plot
from ot.datasets import make_1D_gauss as gauss
from scipy.stats import uniform, t
from all_funcs import wfr_kernel, wfr_distance, spar_sinkhorn_unbalanced, rand_sinkhorn_unbalanced
from all_funcs import unifNys, nys_sinkhorn_unbalanced, nys_sinkhorn_unbalanced2

import warnings
warnings.filterwarnings("ignore")

import os
os.getcwd()
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


#%% Generate the synthetic data
n1 = 500  # nb bins
n2 = 500
d_list = np.array([5, 10, 20, 50])
nloop = 20
ns = 4

case_name = 'C1'  # 'C1' or 'C2' or 'C3'
eta_name = 'R3'  # 'R1' or 'R2' or 'R3'

if case_name == 'C1':
    case = ['uniform', 'gaussian']
    if eta_name == 'R1':
        eta_list = np.array([0.33,0.45,0.62,0.96])
    elif eta_name == 'R2':
        eta_list = np.array([0.28,0.40,0.57,0.91])
    else:
        eta_list = np.array([0.23,0.35,0.52,0.86])
elif case_name == 'C2':
    case = ['gaussian', 'gaussian']
    if eta_name == 'R1':
        eta_list = np.array([1.1, 1.5, 2.1, 3.3]) 
    elif eta_name == 'R2':
        eta_list = np.array([0.9, 1.3, 1.9, 3.1])
    else:
        eta_list = np.array([0.7, 1.1, 1.7, 2.9])
else:
    case = ['uniform', 't']
    if eta_name == 'R1':
        eta_list = np.array([0.33,0.45,0.62,0.96])
    elif eta_name == 'R2':
        eta_list = np.array([0.28,0.40,0.57,0.91])
    else:
        eta_list = np.array([0.23,0.35,0.52,0.86])
## Uniform: (5d: 0.33,0.28,0.23) (10d: 0.45,0.4,0.35) (20d: 0.62,0.57,0.52) (50d: 0.96,0.91,0.86)
## Gaussian: (5d: 1.1,0.9,0.7) (10d: 1.5,1.3,1.1) (20d: 2.1,1.9,1.7) (50d: 3.3,3.1,2.9)


epsilon = 0.1 # entropy parameter
reg_kl = 1.  # Unbalanced KL relaxation parameter


err_nys = np.zeros([nloop, ns, len(d_list)])
err_spar = np.zeros([nloop, ns, len(d_list)])
err_rand = np.zeros([nloop, ns, len(d_list)])

k = 0

for k in range(len(d_list)):
    d = d_list[k]
    eta = eta_list[k]

    if case[0] == 'uniform':
        # Uniform
        xs = np.empty([n1, d])
        for jj in range(d):
            xs[:,jj] = uniform.rvs(size=n1)
    else:
        ## Gaussian
        mu = np.tile(0, d)
        cov = np.eye(d)
        for ii in range(d):
            for jj in range(d):
                cov[ii,jj] = 0.6**(abs(ii-jj))
        xs = np.random.multivariate_normal(mu, cov, n1)
    xt = xs.copy()
    
    m1 = n1/3; s1 = n1/20
    m2 = n2/2; s2 = n2/20
    if case[1] == 'gaussian':
        # Gaussian distribution
        a = gauss(n1, m=m1, s=s1)  # m=mean, s=std
        b = gauss(n2, m=m2, s=s2)  # m=mean, s=std
    else:
        # t distribution
        a = t.pdf(np.arange(n1), df=5, m=m1, s=s1)
        b = t.pdf(np.arange(n2), df=5, m=m2, s=s2)
    
    a = a/np.sum(a)
    b = b/np.sum(b)
    a *= 5.
    b *= 3.
    
    # =============================================================================
    # pl.figure(1, figsize=(5, 5))
    # plt.scatter(xs[:,0],xs[:,1], c='b', label="a")
    # plt.scatter(xt[:,0],xt[:,1], c='r', marker='x', label="b")
    # plt.legend()
    # plt.show()
    # =============================================================================
    
    
    #%% Compute cost and kernel matrices
    K = wfr_kernel(xs, xt, eta, epsilon)
    M = -epsilon*np.log(K)
    # M /= np.max(M[K!=0])
    M[K==0] = -epsilon*np.log(1e-100)
    sparsity = np.sum(K==0)/(n1*n2)
    nonzero = np.sum(K!=0)
    print("sparsity:",sparsity)
    
    
    # Sinkhorn
    G = ot.unbalanced.sinkhorn_unbalanced(a, b, M, epsilon, reg_kl)
    WFR = wfr_distance(a, b, M, K, G, epsilon, reg_kl)
    print("Sinkhorn WFR", WFR)


    #%% Element-wise sampling
    nonzero = np.sum(K!=0)
    print('nnz: ', nonzero)
    nnzSqrt = int(np.sqrt(nonzero))
    s_list = np.floor([(2**2)*nnzSqrt, (2**3)*nnzSqrt, (2**4)*nnzSqrt, (2**5)*nnzSqrt])
    s_list = s_list.astype(np.int)
    print('s:',s_list)
    
    
    
    #%% Nys-Sink
    for i in range(nloop):
        print("nys-sink: d =", d, "cycle", i)
        j = 0
        for s in s_list:
            R, A = unifNys(K, s)
            WFR_nys = nys_sinkhorn_unbalanced(a, b, M, R, A, epsilon, reg_kl)
            
            if np.isnan(WFR_nys):
                K_nys = np.dot(np.dot(R, A), R.T)
                WFR_nys = nys_sinkhorn_unbalanced2(a, b, M, K_nys, epsilon, reg_kl)
                
            err_nys[i,j,k] = abs(WFR-WFR_nys)/WFR
            
            j += 1
    
    
    
    #%% Spar-Sink
    for i in range(nloop):
        print("spar-sink: d =", d, "cycle", i)
        j = 0
        for s in s_list:
            WFR_spar = spar_sinkhorn_unbalanced(a, b, M, K, s, epsilon, reg_kl)
            err_spar[i,j,k] = abs(WFR-WFR_spar)/WFR
            j += 1



    #%%% Rand-Sink
    for i in range(nloop):
        print("rand-sink: d =", d, "cycle", i)
        j = 0
        for s in s_list:
            WFR_rand = rand_sinkhorn_unbalanced(a, b, M, K, s, epsilon, reg_kl)
            err_rand[i,j,k] = abs(WFR-WFR_rand)/WFR
            j += 1
    

err_rand[err_rand==np.inf] = np.nan
err_spar[err_spar==np.inf] = np.nan
err_nys[err_nys==np.inf] = np.nan


print("d =",d_list[0])
print("nys-sink:",np.nanmean(err_nys[:,:,0], axis=0))
print("rand-sink:",np.nanmean(err_rand[:,:,0], axis=0))
print("spar-sink:",np.nanmean(err_spar[:,:,0], axis=0))

print("d =",d_list[1])
print("nys-sink:",np.nanmean(err_nys[:,:,1], axis=0))
print("rand-sink:",np.nanmean(err_rand[:,:,1], axis=0))
print("spar-sink:",np.nanmean(err_spar[:,:,1], axis=0))

print("d =",d_list[2])
print("nys-sink:",np.nanmean(err_nys[:,:,2], axis=0))
print("rand-sink:",np.nanmean(err_rand[:,:,2], axis=0))
print("spar-sink:",np.nanmean(err_spar[:,:,2], axis=0))

print("d =",d_list[3])
print("nys-sink:",np.nanmean(err_nys[:,:,3], axis=0))
print("rand-sink:",np.nanmean(err_rand[:,:,3], axis=0))
print("spar-sink:",np.nanmean(err_spar[:,:,3], axis=0))


#%% Plot
pl.figure(5, figsize=(5, 5))
plt.title(r'%s, %s'%(case_name, eta_name), fontsize=20)

plt.errorbar(np.log10(s_list), np.nanmean(np.log10(err_rand[:,:,0]), axis=0), np.nanstd(np.log10(err_rand[:,:,0]), axis=0),
              fmt='b.-', label='Rand-Sink (d=5)')
plt.errorbar(np.log10(s_list), np.nanmean(np.log10(err_rand[:,:,1]), axis=0), np.nanstd(np.log10(err_rand[:,:,1]), axis=0),
              fmt='bv--', label='Rand-Sink (d=10)')
plt.errorbar(np.log10(s_list), np.nanmean(np.log10(err_rand[:,:,2]), axis=0), np.nanstd(np.log10(err_rand[:,:,2]), axis=0),
              fmt='bx-.', label='Rand-Sink (d=20)')
plt.errorbar(np.log10(s_list), np.nanmean(np.log10(err_rand[:,:,3]), axis=0), np.nanstd(np.log10(err_rand[:,:,3]), axis=0),
              fmt='bo:', label='Rand-Sink (d=50)')


plt.errorbar(np.log10(s_list), np.nanmean(np.log10(err_nys[:,:,0]), axis=0), np.nanstd(np.log10(err_nys[:,:,0]), axis=0),
              fmt='y.-', label='Nys-Sink (d=5)')
plt.errorbar(np.log10(s_list), np.nanmean(np.log10(err_nys[:,:,1]), axis=0), np.nanstd(np.log10(err_nys[:,:,1]), axis=0),
              fmt='yv--', label='Nys-Sink (d=10)')
plt.errorbar(np.log10(s_list), np.nanmean(np.log10(err_nys[:,:,2]), axis=0), np.nanstd(np.log10(err_nys[:,:,2]), axis=0),
              fmt='yx-.', label='Nys-Sink (d=20)')
plt.errorbar(np.log10(s_list), np.nanmean(np.log10(err_nys[:,:,3]), axis=0), np.nanstd(np.log10(err_nys[:,:,3]), axis=0),
              fmt='yo:', label='Nys-Sink (d=50)')


plt.errorbar(np.log10(s_list), np.nanmean(np.log10(err_spar[:,:,0]), axis=0), np.nanstd(np.log10(err_spar[:,:,0]), axis=0),
              fmt='r.-', label='Spar-Sink (d=5)')
plt.errorbar(np.log10(s_list), np.nanmean(np.log10(err_spar[:,:,1]), axis=0), np.nanstd(np.log10(err_spar[:,:,1]), axis=0),
              fmt='rv--', label='Spar-Sink (d=10)')
plt.errorbar(np.log10(s_list), np.nanmean(np.log10(err_spar[:,:,2]), axis=0), np.nanstd(np.log10(err_spar[:,:,2]), axis=0),
              fmt='rx-.', label='Spar-Sink (d=20)')
plt.errorbar(np.log10(s_list), np.nanmean(np.log10(err_spar[:,:,3]), axis=0), np.nanstd(np.log10(err_spar[:,:,3]), axis=0),
              fmt='ro:', label='Spar-Sink (d=50)')

# plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=17)

plt.xlabel(r'$\log_{10}$(s)', fontsize=17)
plt.ylabel(r'$\log_{10}$(RMAE)', fontsize=17)
plt.tick_params(labelsize=17)
plt.show()
