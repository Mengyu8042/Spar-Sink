# -*- coding: utf-8 -*-
"""
Cycle illustration via MDS using the WFR distance matrix computed by Spar-Sink

Three cases: health, heart failure, and arrhythmia
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.io
from sklearn.manifold import MDS
import os


output_path = os.path.join(os.path.dirname(os.getcwd()), 'output/')
case_list = ['health', 'failure', 'arrhythmia']


for case in case_list:

    if case == 'health':
        dist_mat = scipy.io.loadmat(os.path.join(output_path, 'dist_mat_health.mat'))['dist_mat']
    elif case == 'failure':
        dist_mat = scipy.io.loadmat(os.path.join(output_path, 'dist_mat_failure.mat'))['dist_mat']
        ind = range(9,50)
        dist_mat = dist_mat[ind,][:,ind]  
    else:
        dist_mat = scipy.io.loadmat(os.path.join(output_path, 'dist_mat_arrhythmia.mat'))['dist_mat']
        ind = range(0,47)
        dist_mat = dist_mat[ind,][:,ind]
    
    
    # Normalize the distance matrix
    np.fill_diagonal(dist_mat, dist_mat[0,1])
    dist_mat = (dist_mat-np.min(dist_mat))/(np.max(dist_mat)-np.min(dist_mat))
    np.fill_diagonal(dist_mat, 0)
    
    plt.figure(1, (5,5))
    f, ax = plt.subplots()
    im = ax.imshow(dist_mat)
    f.colorbar(im)
    plt.title('WFR distance matrix', fontsize=20)
    plt.show()
    
    
    # MDS
    mds = MDS(
        n_components=2,
        metric=True,
        max_iter=1000,
        eps=1e-12,
        dissimilarity="precomputed",
        random_state=821,
        n_init=4,
    )
    X_transform = mds.fit_transform(dist_mat)  # Get the embeddings
    
    time_len = np.array(range(X_transform.shape[0]))
    
    
    plt.figure(2, (5,5))
    f, ax = plt.subplots()
    ax.plot(X_transform[:,0], X_transform[:,1], c = 'grey')
    cmap = sns.cubehelix_palette(as_cmap=True)
    points = ax.scatter(X_transform[:,0], X_transform[:,1], c = time_len, cmap = cmap)
    cbar = f.colorbar(points)
    cbar.set_label('time', fontsize=17)
    plt.title('MDS in 2D', fontsize=20)
    plt.xticks([])
    plt.yticks([])
    plt.show()
