# -*- coding: utf-8 -*-
"""
Created on Sat Jun  8 10:46:04 2024

@author: HY
"""

import argparse
import datetime
import pandas as pd

from solution import experiment


def init_args():
    parser = argparse.ArgumentParser(description='some arguments')
    
    
    parser.add_argument('--dataset', type=str, default= 'D8' )  ## D4, D8, Seattle
    parser.add_argument('--mode', type=str, default= 'RM' )    ## RM, NM
    parser.add_argument('--smr', type=int, default= 0.2)  ##  sensor missing rate: 0.2, 0.
    parser.add_argument('--ibr', type=int, default= 0.0 )  ## permanent missing ratio
    parser.add_argument('--day_intervals', type=int, default= 288)   
    
    
    ## MVLR parameter
    parser.add_argument('--rho', type=list, default= [1e-6, 1e-5, 1e-5])  
    parser.add_argument('--θ', type=list, default= [0.1, 0.05, 0.3 ])
    parser.add_argument('--Θ', type=int, default= 0.05)    ## truncated percentage of tensor Z
    parser.add_argument('--w', type=list, default= [ 1, 1, 1])  ## weights for tensor L
    parser.add_argument('--α', type=list, default= [1/3, 1/3, 1/3]) ## weights for mode k
    parser.add_argument('--μ', type=int, default= 1e-4)  ## 乘子项Y的惩罚系数
    parser.add_argument('--β', type=int, default= 1e-2)  ## R-Z 的惩罚系数
    parser.add_argument('--λ', type=int, default= 100)  ## 子空间张量R的权衡系数 1e-1
    parser.add_argument('--γ', type=int, default= [1, 1, 1])  ## penalty for E term  1e-6, 1e-3, 1e-3
    parser.add_argument('--t', type=list, default= [1.3, 1.2, 1.2]) ##
    parser.add_argument('--c', type=int, default= 1e-4)
    parser.add_argument('--maxiter', type=int, default= 200)
    
    
    args = parser.parse_args()
    
    
    return args




args = init_args()


def run():
    """
    implement experiments
    """
    args = init_args()
    experiment(args)
    
    # accuracy_plot(args)
                

        
        
# main function
if __name__ == '__main__':
    s_time = datetime.datetime.now()
    print(s_time, 'start')
    run()
    e_time = datetime.datetime.now()
    print(e_time, 'running time:', e_time - s_time)
