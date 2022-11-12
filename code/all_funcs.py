# -*- coding: utf-8 -*-

"""
(Unbalanced) optimal transport distance approximation methods 

Basic functions include:
1) Computing the WFR kernel matrix and the WFR distance:
    `wfr_kernel` and `wfr_distance`
2) Implementing our proposed Spar-Sink method for OT and UOT:
    `spar_sinkhorn` and `spar_sinkhorn_unbalanced` with method='spar-sink'
3) Implementing the Rand-Sink method for OT and UOT:
    `spar_sinkhorn` and `spar_sinkhorn_unbalanced` with method='rand-sink'
4) Implementing the Nys-Sink method for OT and UOT:
    Sampling func: `unifNys` and `levNys`
    Sinkhorn func: `nys_sinkhorn`, `nys_sinkhorn_unbalanced`, 
                   `nys_sinkhorn2`, and `nys_sinkhorn_unbalanced2`
"""

import numpy.matlib 
import numpy as np
from sklearn import decomposition
from scipy.spatial.distance import cdist
from scipy import sparse
import warnings
import torch


#%% 1) WFR functions

def wfr_kernel(X, Y, eta=100, reg=0.1):
    '''
    Compute the kernel matrix of WFR distance between X and Y
    '''
    Ksub = cdist(X, Y, metric="euclidean")/(2*eta)
    Ksub[Ksub > np.pi/2] = np.pi/2
    Ksub = np.cos(Ksub)**(2/reg)

    return Ksub



def wfr_distance(a, b, M, K, G, reg, reg_kl):
    '''
    Compute the WFR distance between a and b given the transport plan G
    '''
    M[G==0] = 0
    res = np.sum(M*G)
    a_star = np.sum(G, axis=1)
    b_star = np.sum(G, axis=0)
    kl1 = sum(np.multiply(a_star,np.log(a_star)-np.log(a)) - a_star + a)
    kl2 = sum(np.multiply(b_star,np.log(b_star)-np.log(b)) - b_star + b)
    res += reg_kl*kl1
    res += reg_kl*kl2
    res = np.sqrt(res)
    
    return res



#%% 2) Spar-Sink

def spar_sinkhorn(a, b, M, K, s, method='spar-sink', sampling='poisson', plan=False, 
                  stable=1e-50, numItermax=1000, stopThr=1e-6, verbose=False):
    '''
    Our porposed SPAR-SINK method for OT
    '''
    ns, nt = K.shape
    
    if sampling == 'poisson':
        if method == 'spar-sink':
            prob = np.sqrt(np.outer(a, b)) * (K > 0)
            prob /= np.sum(prob)
        elif method == 'rand-sink':
            prob = np.ones((a.shape[0], b.shape[0])) * (K > 0)
            prob /= np.sum(prob)
            
        prob *= s
        prob[prob>1] = 1
        
        mask = torch.bernoulli(torch.from_numpy(prob)).numpy()
        nnz = np.sum(mask)
        
        K_spar = np.zeros((a.shape[0], b.shape[0]))
        K_spar[mask!=0] = K[mask!=0]/prob[mask!=0]
        
    else:
        if method == 'spar-sink':
            a_sqrt = np.sqrt(a)
            b_sqrt = np.sqrt(b)
        elif method == 'rand-sink':
            a_sqrt = np.ones(ns)
            b_sqrt = np.ones(nt)
            
        prob = np.vstack((a_sqrt, b_sqrt))
        ind = torch.multinomial(torch.tensor(prob), int(s), replacement=True, generator=None)
        ind = ind.numpy().T
        ind = ind[:,0]*nt + ind[:,1]
        p = np.outer(a_sqrt, b_sqrt)
        p = p / np.sum(p)
        p = np.reshape(s*p, (1,-1))[0]
        
        K_flatten = np.reshape(K, (1,-1))[0]
        K_spar = np.zeros(ns * nt)
        K_spar[ind] = K_flatten[ind] / p[ind]
        K_spar = np.reshape(K_spar, (ns, nt))
        nnz = np.sum(K_spar!=0)
        
        
    id_row = np.squeeze(np.asarray(np.where(np.sum(K_spar, axis=1) != 0)))
    id_col = np.squeeze(np.asarray(np.where(np.sum(K_spar, axis=0) != 0)))
    K_spar = K_spar[id_row, ][:, id_col]
    K = K[id_row, ][:, id_col]
    M = M[id_row, ][:, id_col]
    a = a[id_row]
    b = b[id_col]

    dim_a, dim_b = K_spar.shape

    u = np.ones(dim_a) / dim_a
    v = np.ones(dim_b) / dim_b
    
    if min(dim_a, dim_b) >= 200:
        K_spar = sparse.csr_matrix(K_spar)

    err = 1.
    
    for ii in range(numItermax):
        uprev = u
        vprev = v

        Kv = K_spar.dot(v)
        u = a / Kv + stable
        Ktu = K_spar.T.dot(u)
        v = b / Ktu + stable            

        if (np.any(Ktu == 0)
                or np.any(np.isnan(u)) or np.any(np.isnan(v))
                or np.any(np.isinf(u)) or np.any(np.isinf(v))):
            # we have reached the machine precision come back to NaN and quit loop
            warnings.warn('Warning: numerical errors at iteration %d' % ii)
            if method != 'rand-sink':
                u = np.full((dim_a, ), np.nan)
                v = np.full((dim_b, ), np.nan)
            else:
                u = uprev
                v = vprev
            break
        
        if ii % 10 == 0:
            # we can speed up the process by checking for the error only all the 10th iterations
            err_u = abs(u - uprev).max() / max(abs(u).max(), abs(uprev).max(), 1.)
            err_v = abs(v - vprev).max() / max(abs(v).max(), abs(vprev).max(), 1.)
            err = 0.5 * (err_u + err_v)

            if err < stopThr:
                break
            
            if verbose:
                if ii % 200 == 0:
                    print(
                        '{:5s}|{:12s}'.format('It.', 'Err') + '\n' + '-' * 19)
                print('{:5d}|{:8e}|'.format(ii, err))

    
    if isinstance(K_spar, np.ndarray):
        G_temp = u[:, None] * K_spar * v[None, :]  # transport plan
    else:
        G_temp = u[:, None] * K_spar.toarray() * v[None, :]  # transport plan
    W = np.sum(G_temp[G_temp!=0] * M[G_temp!=0])
        
    if plan == False:
        return np.sqrt(W), nnz
    
    else:
        G_temp2 = np.zeros((ns, dim_b))
        G_temp2[id_row, ] = G_temp
        G = np.zeros((ns, nt))
        G[:, id_col] = G_temp2

        return np.sqrt(W), nnz, G
    



def spar_sinkhorn_unbalanced(a, b, M, K, s, reg, reg_kl, method='spar-sink', plan=False,
                             numItermax=1000, stopThr=1e-6):
    '''
    Our porposed SPAR-SINK method for UOT
    '''
    ns, nt = K.shape
    
    if method == 'spar-sink':
        scale1 = reg / (2*reg_kl + reg)
        scale2 = reg_kl / (2*reg_kl + reg)
        prob = K**scale1 * np.outer(a, b)**scale2
    elif method == 'rand-sink':
        prob = np.ones((a.shape[0], b.shape[0])) * (K > 0)

    prob /= np.sum(prob)
    prob *= s
    prob[prob>1] = 1
    
    mask = torch.bernoulli(torch.from_numpy(prob)).numpy()
    nnz = np.sum(mask)
    
    K_spar = np.zeros((a.shape[0], b.shape[0]))
    K_spar[mask!=0] = K[mask!=0]/prob[mask!=0]


    id_row = np.squeeze(np.asarray(np.where(np.sum(K_spar, axis=1) != 0)))
    id_col = np.squeeze(np.asarray(np.where(np.sum(K_spar, axis=0) != 0)))
    K_spar = K_spar[id_row, ][:, id_col]
    K = K[id_row, ][:, id_col]
    M = M[id_row, ][:, id_col]
    a = a[id_row]
    b = b[id_col]

    dim_a, dim_b = K_spar.shape
    
    u = np.ones(dim_a) / dim_a
    v = np.ones(dim_b) / dim_b

    if min(dim_a, dim_b) >= 200:
        K_spar = sparse.csr_matrix(K_spar)

    fi = reg_kl / (reg_kl + reg)

    err = 1.

    for ii in range(numItermax):
        uprev = u
        vprev = v

        Kv = K_spar.dot(v)
        u = (a / Kv) ** fi
        Ktu = K_spar.T.dot(u)
        v = (b / Ktu) ** fi

        if (np.any(Ktu == 0.)
                or np.any(np.isnan(u)) or np.any(np.isnan(v))
                or np.any(np.isinf(u)) or np.any(np.isinf(v))):
            # we have reached the machine precision come back to NaN and quit loop
            warnings.warn('Warning: numerical errors at iteration %d' % ii)
            if method != 'rand-sink':
                u = np.full((dim_a, ), np.nan)
                v = np.full((dim_b, ), np.nan)
            else:
                u = uprev
                v = vprev
            break
        
        if ii % 10 == 0:
            err_u = abs(u - uprev).max() / max(abs(u).max(), abs(uprev).max(), 1.)
            err_v = abs(v - vprev).max() / max(abs(v).max(), abs(vprev).max(), 1.)
            err = 0.5 * (err_u + err_v)
            
            if err < stopThr:
                break
          
            
    if isinstance(K_spar, np.ndarray):
        G_temp = u[:, None] * K_spar * v[None, :]  # transport plan
    else:
        G_temp = u[:, None] * K_spar.toarray() * v[None, :]  # transport plan
    
    res = np.sum(G_temp[G_temp!=0] * M[G_temp!=0])
    a_star = np.sum(G_temp, axis=1)
    b_star = np.sum(G_temp, axis=0)
    kl1 = sum(np.multiply(a_star, np.log(a_star)-np.log(a)) - a_star + a)
    kl2 = sum(np.multiply(b_star, np.log(b_star)-np.log(b)) - b_star + b)
    res += reg_kl*kl1
    res += reg_kl*kl2 
        
    if plan == False:
        return np.sqrt(res), nnz
    
    else:
        G_temp2 = np.zeros((ns, dim_b))
        G_temp2[id_row, ] = G_temp
        G = np.zeros((ns, nt))
        G[:, id_col] = G_temp2
    
    return np.sqrt(res), nnz, G



#%% 3) Nys-Sink

def unifNys(X, s, stable=1e-20):
    '''
    Uniform column sampling to construct a low-rank approximation of X
    '''
    n = X.shape[0]
    r = int(np.ceil(s/n))
    id = np.random.choice(range(n), r, replace=True) 
    id = np.unique(id)
    r = len(id)
    
    R = X[:,id]
    A = X[id,:][:,id] + stable * np.eye(r)
    A = np.linalg.inv(A)

    return [R, A]



def levNys(X, s, order=2, stable=1e-20):
    '''
    Leveraging column sampling to construct a low-rank approximation of X
    '''
    n = X.shape[0]
    r = int(np.ceil(s/n))
    # U,D,VT = np.linalg.svd(X, full_matrices=False)
    U,D,VT = decomposition.randomized_svd(X, order)
    prob = np.sum(U*U, axis=1)
    prob /= np.sum(prob)
    id = np.random.choice(range(n), r, replace=False, p=prob) 
    
    R = X[:,id]
    A = X[id,:][:,id] + stable * np.eye(r)
    A = np.linalg.inv(A)

    return [R, A]



def nys_sinkhorn(a, b, M, R, A, numItermax=1000, stopThr=1e-6, stable=1e-50):
    '''
    NYS-SINK method for OT
    
    Input the Nystrom decomposition of the kernel matrix K, i.e., K = R * A * R.T
    '''
    dim_a = R.shape[0]

    u = np.ones(dim_a) / dim_a
    v = np.ones(dim_a) / dim_a

    err = 1.
    
    for ii in range(numItermax):
        uprev = u
        vprev = v

        Kv = np.dot(R, A.dot(np.dot(R.T,v)))
        u = a / Kv 
        Ktu = np.dot(R, A.dot(np.dot(R.T,u)))
        v = b / Ktu

        if (np.any(Ktu == 0)
                or np.any(np.isnan(u)) or np.any(np.isnan(v))
                or np.any(np.isinf(u)) or np.any(np.isinf(v))):
            # we have reached the machine precision come back to previous solution and quit loop
            warnings.warn('Warning: numerical errors at iteration %d' % ii)
            u = uprev
            v = vprev
            break
        
        if ii % 10 == 0:
            # we can speed up the process by checking for the error only all the 10th iterations
            err_u = abs(u - uprev).max() / max(abs(u).max(), abs(uprev).max(), 1.)
            err_v = abs(v - vprev).max() / max(abs(v).max(), abs(vprev).max(), 1.)
            err = 0.5 * (err_u + err_v)

            if err < stopThr:
                break

    u[u<0] = 0; v[v<0] = 0
    G = u.reshape((-1, 1)) * (np.dot(np.dot(R,A),R.T)) * v.reshape((1, -1))  # transport plan
    G[G<0] = 0
    W = np.sum(G[G!=0] * M[G!=0])
        
    return np.sqrt(W), G



def nys_sinkhorn_unbalanced(a, b, M, R, A, reg, reg_kl, numItermax=1000, stopThr=1e-6):
    '''
    NYS-SINK method for UOT
    
    Input the Nystrom decomposition of the kernel matrix K, i.e., K = R * A * R.T
    '''
    dim_a = R.shape[0]

    u = np.ones(dim_a) / dim_a
    v = np.ones(dim_a) / dim_a
    
    fi = reg_kl / (reg_kl + reg)
    
    err = 1.
    
    for ii in range(numItermax):
        uprev = u
        vprev = v

        Kv = np.dot(R, A.dot(np.dot(R.T,v)))
        u = (a / Kv) ** fi
        Ktu = np.dot(R, A.dot(np.dot(R.T,u)))
        v = (b / Ktu) ** fi

        if (np.any(Ktu == 0)
                or np.any(np.isnan(u)) or np.any(np.isnan(v))
                or np.any(np.isinf(u)) or np.any(np.isinf(v))):
            warnings.warn('Warning: numerical errors at iteration %d' % ii)
            u = uprev
            v = vprev
            break
        
        err_u = abs(u - uprev).max() / max(abs(u).max(), abs(uprev).max(), 1.)
        err_v = abs(v - vprev).max() / max(abs(v).max(), abs(vprev).max(), 1.)
        err = 0.5 * (err_u + err_v)
        if err < stopThr:
            break
    
    u[u<0] = 0; v[v<0] = 0
    G = u.reshape((-1, 1)) * (np.dot(np.dot(R,A),R.T)) * v.reshape((1, -1))  # transport plan
    G[G<0] = 0
    
    res = np.sum(G[G!=0] * M[G!=0])
    a_star = np.sum(G, axis=1)
    b_star = np.sum(G, axis=0)
    kl1 = sum(np.multiply(a_star, np.log(a_star)-np.log(a)) - a_star + a)
    kl2 = sum(np.multiply(b_star, np.log(b_star)-np.log(b)) - b_star + b)
    res += reg_kl*kl1
    res += reg_kl*kl2
    
    return np.sqrt(res), G



def nys_sinkhorn2(a, b, M, K, numItermax=1000, stopThr=1e-6, stable=1e-50):
    '''
    NYS-SINK method for OT
    
    Input the entire kernel matrix K in case of numerical issues
    '''
    ns, nt = K.shape
    id_row = np.squeeze(np.asarray(np.where(np.sum(K, axis=1) != 0)))
    id_col = np.squeeze(np.asarray(np.where(np.sum(K, axis=0) != 0)))
    K = K[id_row, ][:, id_col]
    M = M[id_row, ][:, id_col]
    a = a[id_row]
    b = b[id_col]

    dim_a, dim_b = K.shape

    u = np.ones(dim_a) / dim_a
    v = np.ones(dim_b) / dim_b

    err = 1.
    
    for ii in range(numItermax):
        uprev = u
        vprev = v

        Kv = K.dot(v)
        u = a / Kv
        Ktu = K.T.dot(u)
        v = b / Ktu

        if (np.any(Ktu == 0)
                or np.any(np.isnan(u)) or np.any(np.isnan(v))
                or np.any(np.isinf(u)) or np.any(np.isinf(v))):
            warnings.warn('Warning: numerical errors at iteration %d' % ii)
            u = uprev
            v = vprev
            break
        
        if ii % 10 == 0:
            err_u = abs(u - uprev).max() / max(abs(u).max(), abs(uprev).max(), 1.)
            err_v = abs(v - vprev).max() / max(abs(v).max(), abs(vprev).max(), 1.)
            err = 0.5 * (err_u + err_v)

            if err < stopThr:
                break
    
    u[u<0] = 0; v[v<0] = 0
    G_temp = u[:, None] * K * v[None, :]  # transport plan
    G_temp[G_temp<0] = 0
    W = np.sum(G_temp[G_temp!=0] * M[G_temp!=0])
     
    G_temp2 = np.zeros((ns, dim_b))
    G_temp2[id_row, ] = G_temp
    G = np.zeros((ns, nt))
    G[:, id_col] = G_temp2
    
    return np.sqrt(W), G



def nys_sinkhorn_unbalanced2(a, b, M, K, reg, reg_kl, numItermax=1000, stopThr=1e-6):
    '''
    NYS-SINK method for UOT
    
    Input the entire kernel matrix K in case of numerical issues
    '''
    ns, nt = K.shape
    id_row = np.squeeze(np.asarray(np.where(np.sum(K, axis=1) != 0)))
    id_col = np.squeeze(np.asarray(np.where(np.sum(K, axis=0) != 0)))
    K = K[id_row, ][:, id_col]
    M = M[id_row, ][:, id_col]
    a = a[id_row]
    b = b[id_col]
    
    dim_a, dim_b = K.shape
    
    u = np.ones(dim_a) / dim_a
    v = np.ones(dim_b) / dim_b

    fi = reg_kl / (reg_kl + reg)

    err = 1.

    for ii in range(numItermax):
        uprev = u
        vprev = v

        Kv = K.dot(v)
        u = (a / Kv) ** fi
        Ktu = K.T.dot(u)
        v = (b / Ktu) ** fi

        if (np.any(Ktu == 0.)
                or np.any(np.isnan(u)) or np.any(np.isnan(v))
                or np.any(np.isinf(u)) or np.any(np.isinf(v))):
            warnings.warn('Warning: numerical errors at iteration %d' % ii)
            u = uprev
            v = vprev
            break
        
        if ii % 10 == 0:
            err_u = abs(u - uprev).max() / max(abs(u).max(), abs(uprev).max(), 1.)
            err_v = abs(v - vprev).max() / max(abs(v).max(), abs(vprev).max(), 1.)
            err = 0.5 * (err_u + err_v)
            if err < stopThr:
                break
    
    u[u<0] = 0; v[v<0] = 0
    G_temp = u[:, None] * K * v[None, :]  # transport plan
    G_temp[G_temp<0] = 0
    
    res = np.sum(G_temp[G_temp!=0] * M[G_temp!=0])
    a_star = np.sum(G_temp, axis=1)
    b_star = np.sum(G_temp, axis=0)
    kl1 = sum(np.multiply(a_star, np.log(a_star)-np.log(a)) - a_star + a)
    kl2 = sum(np.multiply(b_star, np.log(b_star)-np.log(b)) - b_star + b)
    res += reg_kl*kl1
    res += reg_kl*kl2
    
    G_temp2 = np.zeros((ns, dim_b))
    G_temp2[id_row, ] = G_temp
    G = np.zeros((ns, nt))
    G[:, id_col] = G_temp2
    
    return np.sqrt(res), G
