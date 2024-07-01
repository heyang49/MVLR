# -*- coding: utf-8 -*-
"""
Created on Sat Jun  8 10:46:04 2024

@author: HY
"""


import numpy as np
from model import MVLR


def experiment(args):
    
    print('\n Experiment:')
    print('Dataset: %s, missing rate: %s' %(args.dataset, args.smr))
    ## data loading
    dense_tensor = np.load('../MVLR/inputs/dense_tensor_%s.npy' %(args.dataset))
    binary_tensor = np.load('../MVLR/inputs/binary_tensor_%s_%s_smr%.1f_ibr%.1f.npy'%(args.dataset, args.mode, args.smr, args.ibr))
    obs_tensor =  np.multiply(dense_tensor, binary_tensor)
    ## model implementation
    rec_tensor, Z, E, L, R, mape_Q, rmse_Q, mape_O, rmse_O, mape_V, rmse_V, MAPE = MVLR(dense_tensor, obs_tensor, args.rho, args.θ, args.Θ, args.w, args.α, args.μ, args.β, args.λ, args.γ, args.t, args.c, args.day_intervals, args.maxiter)

        

            
            

    
    
    
