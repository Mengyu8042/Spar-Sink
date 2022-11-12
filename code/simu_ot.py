# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import ot
import ot.plot
from ot.datasets import make_1D_gauss as gauss
from scipy.stats import uniform, t
from all_funcs import spar_sinkhorn, nys_sinkhorn, nys_sinkhorn2, unifNys

import warnings
warnings.filterwarnings("ignore")

import os
os.getcwd()
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"



#%% Generate the synthetic data
n1 = 1000  # sample size (number of support points)
n2 = 1000
d_list = np.array([5, 10, 20, 50])  # dimension of support points
nloop = 50  # number of replications

s0 = 0.001*n1*np.log(n1)**4
s_list = np.array([2, 2**2, 2**3, 2**4]) * s0  # subsample size
s_list = s_list.astype(np.int)
ns = len(s_list)
print('s:', s_list)
print('s/n^2:', s_list / (n1*n2))


for case_name in ['C1', 'C2', 'C3']:
    for epsilon in [0.1, 0.01, 0.001]:  # entropy parameter
        print(case_name, 'epsilon =', epsilon)
    
    
        if case_name == 'C1':
            case = ['uniform', 'gaussian']
        elif case_name == 'C2':
            case = ['gaussian', 'gaussian']
        elif case_name == 'C3':
            case = ['uniform', 't']
            

        if case_name == 'C3' and epsilon == 0.001:
            stable = 1e-35
        else:
            stable = 1e-50
        
        
        err_nys = np.zeros([nloop, ns, len(d_list)])
        err_spar = np.zeros([nloop, ns, len(d_list)])
        err_rand = np.zeros([nloop, ns, len(d_list)])

        
        for k in range(len(d_list)):
            d = d_list[k]
            print('d =', d)
            
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
                        cov[ii,jj] = 0.5**(abs(ii-jj))
                xs = np.random.multivariate_normal(mu, cov, n1)
            xt = xs.copy()
            
            m1 = n1/3; s1 = n1/25
            m2 = n2/2; s2 = n2/25
            if case[1] == 'gaussian':
                # Gaussian distribution
                a = gauss(n1, m=m1, s=s1)  # m=mean, s=std
                b = gauss(n2, m=m2, s=s2)  # m=mean, s=std
            else:
                # t distribution
                a = t.pdf(np.arange(n1), df=5, loc=m1, scale=s1)
                b = t.pdf(np.arange(n2), df=5, loc=m2, scale=s2)
            a = a/np.sum(a)
            b = b/np.sum(b)

            
            #%% Compute cost and kernel matrices
            M = ot.dist(xs, xt)
            M /= np.max(M)
            K = np.exp(-M/epsilon)
            
            
            #%% Shinkhorn
            W = ot.sinkhorn2(a, b, M, epsilon, method='sinkhorn')
            W = np.sqrt(W)

            
            #%% Spar-Sink
            s_spar_list = np.zeros([nloop, ns])
            for i in range(nloop):
                for j, s in enumerate(s_list):
                    W_spar, s_spar = spar_sinkhorn(a, b, M, K, s, 'spar-sink', stable=stable)
                    err_spar[i,j,k] = abs((W-W_spar)/W)
                    s_spar_list[i,j] = s_spar
                    
            s_spar_list = np.floor(np.nanmean(s_spar_list, axis=0))
            s_spar_list = s_spar_list.astype(np.int)
            
            
            #%%% Rand-Sink
            for i in range(nloop):
                for j, s in enumerate(s_spar_list):
                    W_rand, _ = spar_sinkhorn(a, b, M, K, s, 'rand-sink', stable=stable)
                    err_rand[i,j,k] = abs((W-W_rand)/W)

            
            #%% Uniform Nys-Sink
            for i in range(nloop):
                for j, s in enumerate(s_spar_list):
                    R, A = unifNys(K, s)
                    W_nys, _ = nys_sinkhorn(a, b, M, R, A, stable=stable)
                    # if np.isnan(W_nys):
                    #     K_nys = np.dot(np.dot(R, A), R.T)
                    #     W_nys, _ = nys_sinkhorn2(a, b, M, K_nys, stable=stable)
                    err_nys[i,j,k] = abs((W-W_nys)/W)

        
        err_rand[err_rand==np.inf] = np.nan
        err_spar[err_spar==np.inf] = np.nan
        err_nys[err_nys==np.inf] = np.nan
        
        for j in range(ns):
            for k in range(len(d_list)):
                err_rand[:,j,k][err_rand[:,j,k] <= np.nanquantile(err_rand[:,j,k], 0.1)] = np.nan
                err_rand[:,j,k][err_rand[:,j,k] >= np.nanquantile(err_rand[:,j,k], 0.9)] = np.nan
                err_spar[:,j,k][err_spar[:,j,k] <= np.nanquantile(err_spar[:,j,k], 0.1)] = np.nan
                err_spar[:,j,k][err_spar[:,j,k] >= np.nanquantile(err_spar[:,j,k], 0.9)] = np.nan
                err_nys[:,j,k][err_nys[:,j,k] <= np.nanquantile(err_nys[:,j,k], 0.1)] = np.nan
                err_nys[:,j,k][err_nys[:,j,k] >= np.nanquantile(err_nys[:,j,k], 0.9)] = np.nan

        
        
        #%% Plot
        plt.figure(1, figsize=(5, 5))
        plt.title(r'%s ($\varepsilon$=%.3f)'%(case_name, epsilon), fontsize=24)

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
        
        # plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=20)
        
        plt.xlabel(r'$\log_{10}$(s)', fontsize=20)
        plt.ylabel(r'$\log_{10}$(RMAE)', fontsize=20)
        plt.tick_params(labelsize=20)
        plt.show()
        # plt.savefig('s_dist_{}_{}.png'.format(case_name, epsilon), dpi=200, bbox_inches='tight')
        # plt.close('all')
