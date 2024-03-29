# Importance Sparsification for Sinkhorn Algorithm
This repository includes the implementation of our work **"Importance Sparsification for Sinkhorn Algorithm"**.


## Introduction
A brief introduction about the folders and files:
* `data/`: the used echocardiogram video dataset downloaded from [EchoNet-Dynamic](https://echonet.github.io/dynamic/).
* `code/`: methods and implementations.
    * `all_funcs.py`: the proposed Spar-Sink method and baselines for OT and UOT.
    * `simu_ot.py` and `simu_uot.py`: simulation code.
    * `echo_dist.py`, `echo_mds.py`, and `echo_pred.py`: echocardiogram analysis code.
* `output/`: outputs of `echo_dist.py`.
    * `dist_mat_health.mat`, `dist_mat_failure.mat`, and `dist_mat_arrhythmia.mat`: WFR distance matrices approximated by Spar-Sink for individuals in the state of health, heart failure, and arrhythmia, respectively.


## Reproducibility
For simulations in Section 5,
* you can run `simu_ot.py` to reproduce the results in Figure 2;
* you can run `simu_uot.py` to reproduce the results in Figure 3.


For echocardiogram analysis in Section 6,
* you can run `echo_dist.py` and `echo_mds.py` to reproduce the results in Figure 7; in `echo_dist.py`, choose `case` among **'health', 'failure', 'arrhythmia'**. Considering that running `echo_dist.py` takes several hours, we provide its outputs in `output/`, and you can directly run `echo_mds.py` using the outputs to reproduce Figure 7.
* you can run `echo_pred.py` to reproduce the results in Table 1; set `n_resize = 1` for panel (a), and set `n_resize = 2` for panel (b). Considering that running the script takes tens of hours, you can decrease the number of samples (e.g., set `n = 20` or `n = 10`) to test it within a limited time.


## Main Dependencies
Environment: Python 3.8

Install the following requirements using the `pip` or `conda` command:
* cv2
* json
* matplotlib
* numpy
* os
* pandas
* POT
* random
* scikit-learn
* scipy
* seaborn
* torch


## Acknowledgements

This toolbox has been created and is maintained by

* [Mengyu Li](https://github.com/Mengyu8042): limengyu516@ruc.edu.cn
* Jun Yu: yujunbeta@bit.edu.cn
* [Tao Li](https://github.com/sherlockLitao): 2019000153lt@ruc.edu.cn
* [Cheng Meng](https://github.com/ChengzijunAixiaoli): chengmeng@ruc.edu.cn

Feel free to contact us if any questions.

## Main References
Rémi Flamary, Nicolas Courty, Alexandre Gramfort, Mokhtar Z. Alaya, Aurélie Boisbunon, Stanislas Chambon, Laetitia Chapel, Adrien Corenflos, Kilian Fatras, Nemo Fournier, Léo Gautheron, Nathalie T.H. Gayraud, Hicham Janati, Alain Rakotomamonjy, Ievgen Redko, Antoine Rolet, Antony Schutz, Vivien Seguy, Danica J. Sutherland, Romain Tavenard, Alexander Tong, and Titouan Vayer. “[POT Python Optimal Transport library](https://pythonot.github.io/),” Journal of Machine Learning Research, 22(78):1−8, 2021.

David Ouyang, Bryan He, Amirata Ghorbani, Neal Yuan, Joseph Ebinger, Curt P. Langlotz, Paul A. Heidenreich, Robert A. Harrington, David H. Liang, Euan A. Ashley, and James Y. Zou. “[EchoNet-Dynamic: A Large New Cardiac Motion Video Data Resource for Medical Machine Learning](https://echonet.github.io/dynamic/),” 2020.
