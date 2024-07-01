# -*- coding: utf-8 -*-
"""
Created on Sat Jun  8 11:27:02 2024

@author: HY
"""

import numpy as np
from numpy.linalg import inv as inv

import matplotlib.pyplot as plt
import time
from numpy.linalg import svd

def ten2mat(tensor, mode):
    return np.reshape(np.moveaxis(tensor, mode, 0), (tensor.shape[mode], -1), order = 'F')

def mat2ten(mat, tensor_size, mode):
    index = list()
    index.append(mode)
    for i in range(tensor_size.shape[0]):
        if i != mode:
            index.append(i)
    return np.moveaxis(np.reshape(mat, list(tensor_size[index]), order = 'F'), 0, mode)

def compute_mape(actual, pred): 
    actual, pred = np.array(actual), np.array(pred)
    return np.mean(np.abs((actual - pred) / actual)) * 100

def compute_rmse(actual, pred): 
    actual, pred = np.array(actual), np.array(pred)
    return np.sqrt(np.mean((actual - pred)**2)) 

def svt_tnn(mat, alpha, rho, theta):
    tau = alpha / rho
    [m, n] = mat.shape
    if 2 * m <= n:
        u, s, v = np.linalg.svd(mat @ mat.T, full_matrices = 0)
        s = np.sqrt(s)
        idx = np.sum(s > tau)
        mid = np.zeros(idx)
        mid[:theta] = 1
        mid[theta:idx] = (s[theta:idx] - tau) / s[theta:idx]
        return (u[:, :idx] @ np.diag(mid)) @ (u[:, :idx].T @ mat)
    elif m > 2 * n:
        return svt_tnn(mat.T, tau, theta).T

def prox_l21(F, α):
    '''apply l2,1 norm minimization'''
    T, Col = F.shape
    E = np.empty((T, Col)) 
    for c in range(Col):
        ## calculate the l2 norm of column c
        Q = np.linalg.norm(F[:, c])   
        if Q > α:
            E[:, c] = (Q-α)/Q * F[:, c]
        else:
            E[:, c] = 0     
    return E

def MVLR(dense_tensor, obs_tensor, ρ, θ, Θ, w, α, μ, β, λ, γ, t, c, day_intervals, maxiter):
    V, T, M  = dense_tensor.shape
    t1, t2, t3 = t
    
    dim = np.array([M, day_intervals , int(T/day_intervals)])
    
    pos_missing_Q = np.where(obs_tensor[0] == 0)
    pos_missing_K = np.where(obs_tensor[1] == 0)
    pos_missing_V = np.where(obs_tensor[2] == 0)
    
    pos_obs_Q = np.where(obs_tensor[0] != 0)
    pos_obs_K = np.where(obs_tensor[1] != 0)
    pos_obs_V = np.where(obs_tensor[2] != 0)

    ## test set - for validation
    pos_test_Q = np.where((dense_tensor[0] != 0) & (obs_tensor[0] == 0))
    pos_test_K = np.where((dense_tensor[1] != 0) & (obs_tensor[1] == 0))
    pos_test_V = np.where((dense_tensor[2] != 0) & (obs_tensor[2] == 0))
    
    ## P(X) = P(M)
    X = obs_tensor.copy()

    # V, T, M  = X.shape
    ## auxilary tensor for tensor X
    F = np.zeros([V, T, M])
    ## subspace tensor,                    
    Z = np.zeros([V, M, M])
    ## auxilary tensor for tensor X
    L = np.zeros([V, T, M])
    ## auxilary tensor for tensor L
    J = np.zeros([3, T, M])
    ## auxilary tensor for tensor Z
    R = np.zeros([V, M, M])
    ## error tensor
    E = np.zeros([V, T, M])

    S = np.zeros([3, V, M, M])    

    ## multiplier tensor                    V*n2*n2
    Y1 = np.zeros([V, T, M])
    ## multiplier tensor                    V*n2*n2
    Y2 = np.zeros([V, T, M])
    ## multiplier tensor                    V*n2*n2
    Y3 = np.zeros([V, M, M])

    MAPE_Q = np.zeros(maxiter)
    MAPE_K = np.zeros(maxiter)
    MAPE_V = np.zeros(maxiter)


    it = 0

    while True:
        for v in range(V): 
            ρ[v] = min(ρ[v]*t1 , 1e6)
        μ = min(μ*t2 , 1e6)
        β = min(β*t3 , 1e6)
        
        
        # Update R
        ## TNN
        dim2 = np.array([V, M, M])
        ten = Z - Y3/β
        for k in range(3):
            mat = ten2mat(ten, k)
            tmat = svt_tnn(mat, α[k]*λ, β, np.int(np.ceil(Θ * dim2[k])))
            S[k]= mat2ten(tmat, dim2, k)
        R = np.mean(S, axis = 0)
        

        ## Update Z
        for v in range(V): 
            Z[v] = inv(μ* X[v].T @ X[v] +  β*np.identity(M)) @ ( X[v].T @ Y2[v] + Y3[v] + μ * X[v].T @ X[v] - μ * X[v].T @ E[v] + β*R[v])


        ## Update X
        for v in range(V): 
            F[v] = (ρ[v]*L[v] + Y1[v] + Y2[v] @ Z[v].T - Y2[v] + μ*(E[v] - E[v] @ Z[v].T) ) @ inv((ρ[v] + μ)*np.identity(M) - μ*Z[v].T - μ*Z[v] + μ*Z[v] @ Z[v].T) 
        X[0][pos_missing_Q] = F[0][pos_missing_Q]
        X[1][pos_missing_K] = F[1][pos_missing_K]
        X[2][pos_missing_V] = F[2][pos_missing_V] 



        ## Update L
        for v in range(V): 
            # TNN
            ten = mat2ten((X[v] - Y1[v]/ρ[v]).T, dim, 0)
            for k in range(3):
                mat = ten2mat(ten, k)
                tmat = svt_tnn(mat, α[k]*w[v], ρ[v], np.int(np.ceil(θ[v] * dim[k])))
                tten = mat2ten(tmat, dim, k)
                J[k] = ten2mat(tten, 0).T
            L[v] = np.mean(J, axis = 0)
            

        ## Update E
        for v in range(V): 
            E[v] = prox_l21(X[v] - X[v] @ Z[v] + Y2[v]/μ, γ[v]/μ)

        ## Update Y1, Y2, Y3
        for v in range(V):
            Y1[v] += ρ[v]*(L[v]-X[v])
            Y2[v] += μ*(X[v] - X[v] @ Z[v] - E[v])
        Y3 += β*(R-Z)

        ## cheak convergence
        tolZ = np.max(R-Z)

        # tolX = np.sqrt(np.sum((tensor_hat - last_tensor) ** 2)) / snorm
        tolL = np.max(L-X)
        

        mape_Q = compute_mape(dense_tensor[0][pos_test_Q], (X[0])[pos_test_Q])
        mape_K = compute_mape(dense_tensor[1][pos_test_K], (X[1])[pos_test_K])
        mape_V = compute_mape(dense_tensor[2][pos_test_V], (X[2])[pos_test_V])

        rmse_Q = compute_rmse(dense_tensor[0][pos_test_Q], (X[0])[pos_test_Q])
        rmse_K = compute_rmse(dense_tensor[1][pos_test_K], (X[1])[pos_test_K])
        rmse_V = compute_rmse(dense_tensor[2][pos_test_V], (X[2])[pos_test_V])

        MAPE_Q[it] = mape_Q
        MAPE_K[it] = mape_K
        MAPE_V[it] = mape_V

        if (it + 1) % 2 == 0:
            print('Iter: {}'.format(it + 1))
            print('MAPE_Q: {:.5}, RMSE_Q: {:.5}'.format(mape_Q, rmse_Q))
            print('MAPE_K: {:.5}, RMSE_K: {:.5}'.format(mape_K, rmse_K))
            print('MAPE_V: {:.5}, RMSE_V: {:.5}'.format(mape_V, rmse_V))
            print()


        if (it>1) and ((tolZ < c) or (tolL < c) or (it+1 >= maxiter)):
            break

        ## update iteration
        it +=1
        
    print('Iter: {}'.format(it + 1))
    print('MAPE_Q: {:.5}, RMSE_Q: {:.5}'.format(mape_Q, rmse_Q))
    print('MAPE_K: {:.5}, RMSE_K: {:.5}'.format(mape_K, rmse_K))
    print('MAPE_V: {:.5}, RMSE_V: {:.5}'.format(mape_V, rmse_V))

    
    
    MAPE = np.zeros((3, it))
    MAPE[0] = MAPE_Q[:it]
    MAPE[1] = MAPE_K[:it]
    MAPE[2] = MAPE_V[:it]
    
    return X, Z, E, L, R , mape_Q, rmse_Q, mape_K, rmse_K, mape_V, rmse_V, MAPE