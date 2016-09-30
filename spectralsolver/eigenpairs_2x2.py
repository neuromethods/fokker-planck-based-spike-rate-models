# -*- coding: utf-8 -*-
"""
Numba-compatible helper functions for two-dimensional matrix and matrix-vector operations.

Created on Fri Aug 21 17:24:53 2015

@author: Moritz Augustin
"""

import cmath
import numpy as np
import scipy.linalg

# choose whether to use numba (in particular its nopython mode) or not
numba_on = True

if numba_on:
    from numba import jit, njit
else:
    jit = lambda f: f
    njit = jit

# function design: input arrays can used for output as well

# possible improvements:
# a) instead of matrix inversion times vector solve of the equivalent linear system
# b) exploit structure of specific coefficient matrix, e.g., sig!=0 no cases nec. in eig

# check if np.dot is available in numba (Z = np.dot(X,Y) instead)
# numba-compatible version of 2d matrix multiplication
# Z = X * Y

# this function computes the complex matrix exponential exp(X*t) where t is a real scalar
# this algorithm does not require eig/inv etc. and is based on the following wikipedia article:
# https://en.wikipedia.org/wiki/Matrix_exponential#Evaluation_by_Laurent_series
# input: real scalar t, complex 2x2 arrays X (argument) and exp_Xt (result) -- ident. refs. allowed
@njit
def exp_mat2_full_scalar(X, t, exp_Xt):
    s = 0.5 * (X[0,0] + X[1,1])
    det_X_minus_sI = (X[0,0]-s) * (X[1,1]-s)  -  X[0,1] * X[1,0]
    q = cmath.sqrt(-det_X_minus_sI)
    # we have to distinguish the case q=0 (the other exceptional case t=0 should not occur)
    if abs(q) < 1e-15: # q is (numerically) 0
        sinh_qt_over_q = t
    else:
        sinh_qt_over_q = cmath.sinh(q*t) / q
    cosh_qt = cmath.cosh(q*t)
    cosh_q_t_minus_s_sinh_qt_over_qt = cosh_qt - s*sinh_qt_over_q
    exp_st = cmath.exp(s*t)
    
    # abbreviations for the case of exp_Xt referencing the same array as X
    E_00 = exp_st * (cosh_q_t_minus_s_sinh_qt_over_qt   +   sinh_qt_over_q * X[0,0])
    E_01 = exp_st * (sinh_qt_over_q * X[0,1])
    E_10 = exp_st * (sinh_qt_over_q * X[1,0])
    E_11 = exp_st * (cosh_q_t_minus_s_sinh_qt_over_qt   +   sinh_qt_over_q * X[1,1])
    
    exp_Xt[0,0] = E_00
    exp_Xt[0,1] = E_01
    exp_Xt[1,0] = E_10
    exp_Xt[1,1] = E_11

# matrix-vector product Z = X Y
@njit
def mult_mat2_full_vec(X, Y, Z):
    Z0 = X[0,0] * Y[0] + X[0,1] * Y[1]
    Z1 = X[1,0] * Y[0] + X[1,1] * Y[1]
    Z[0] = Z0
    Z[1] = Z1

# full matrix product Z = X Y
@njit
def mult_mat2_full(X, Y, Z):
    Z_00 = X[0,0] * Y[0,0] + X[0,1] * Y[1,0]
    Z_01 = X[0,0] * Y[0,1] + X[0,1] * Y[1,1]
    Z_10 = X[1,0] * Y[0,0] + X[1,1] * Y[1,0]
    Z_11 = X[1,0] * Y[0,1] + X[1,1] * Y[1,1]
    Z[0,0] = Z_00
    Z[0,1] = Z_01
    Z[1,0] = Z_10
    Z[1,1] = Z_11

# similar but Y is a diagonal matrix (array with diag elements)
# (todo: check if * operator is available in numba (D*V instead))
@njit
def mult_mat2_full_diag(X, Y, Z):
    Z[0,0] = X[0,0] * Y[0]
    Z[0,1] = X[0,1] * Y[1]
    Z[1,0] = X[1,0] * Y[0]
    Z[1,1] = X[1,1] * Y[1]
    
# inverse matrix (has to exist) Z = X^{-1}
@njit
def inv_mat2_full(X, Z):
    detXinv = 1.0 / (X[0,0]*X[1,1] - X[0,1]*X[1,0])
    Z_00 = X[1,1] * detXinv
    Z_01 = -X[0,1] * detXinv
    Z_10 = -X[1,0] * detXinv
    Z_11 = X[0,0] * detXinv
    Z[0,0] = Z_00
    Z[0,1] = Z_01
    Z[1,0] = Z_10
    Z[1,1] = Z_11

# full eigenvalue eigenvector decomposition of X = V D V^{-1}
# (the if-else cases can be get rid of if the structure of X is known)
@njit
def eig_mat2_full(X, D, V):
    # abbreviations
    detX = X[0,0]*X[1,1] - X[0,1]*X[1,0]
    trX = X[0,0] + X[1,1]
    root = cmath.sqrt(trX*trX/4.0-detX)
    
    # eigenvals    
    D[0] = trX/2.0 + root
    D[1] = trX/2.0 - root
    
    # eigenvecs
    if X[1,0] != 0:
        # eigenvec1
        V_00 = D[0] - X[1,1]
        V_10 = X[1,0]
        # eigenvec2
        V_01 = D[1] - X[1,1]
        V_11 = X[1,0]
        
    elif X[0,1] != 0:
        # eigenvec1
        V_00 = X[0,1]
        V_10 = D[0] - X[0,0]
        # eigenvec2
        V_01 = X[0,1]
        V_11 = D[1] - X[0,0]
        
    else: # X diagonal => unity eigenvecs
        # eigenvec1
        V_00 = 1.0
        V_10 = 0.0
        # eigenvec2
        V_01 = 0.0
        V_11 = 1.0
    
    V[0,0] = V_00
    V[1,0] = V_10
    V[0,1] = V_01
    V[1,1] = V_11
        
@njit
def exp_mat2_diag(D, E):
    E[0] = cmath.exp(D[0])
    E[1] = cmath.exp(D[1])


# the main section contains only some manual tests using the functions
if __name__ == '__main__':
    print('analytic calculation of the 2x2 matrix exponential '+
          'through exploting the eigenvalue-/vector decomposition')   
          
          
          
    # testing...
    
    N = 2
    
    print('1) NUMPY FUNCTIONS')
    A = np.random.rand(N, N) + 1j*np.random.rand(N, N)
    D, V = np.linalg.eig(A)
    invV = np.linalg.inv(V)
    Aspec = np.dot(D*V, invV) # V * D * inv(V) = A
    
    print('||A - A_spec|| = {}'.format(np.linalg.norm(A - Aspec)))
    
    
    expA = scipy.linalg.expm(A) # exp(A)
    expA_spec = np.dot(np.exp(D)*V, invV)  # exp(A) = V * exp(D) * inv(V) 
                                          # np.dot(np.dot(V, np.diag(np.exp(D))), )
    print('||expA - expA_spec|| = {}'.format(np.linalg.norm(expA - expA_spec)))
    
    if N != 2:
        print('the following code can only be applied to 2x2 matrices')
    
    print('2) CUSTOM FUNCTIONS')
    
    # matrix storage
    V_custom = np.zeros(A.shape, dtype=np.complex)
    D_custom = np.zeros(V_custom.shape[0], dtype=np.complex)
    Aspec_custom = np.zeros_like(V_custom) # not required for exp(A) calc
    VD_custom = np.zeros_like(V_custom)
    invV_custom = np.zeros_like(V_custom)
    
    eig_mat2_full(A, D_custom, V_custom)
    inv_mat2_full(V_custom, invV_custom)
    mult_mat2_full_diag(V_custom, D_custom, VD_custom)
    mult_mat2_full(VD_custom, invV_custom, Aspec_custom)
    print('||A - Aspec_custom|| = {}'.format(np.linalg.norm(A - Aspec_custom)))
    
    expD_custom = np.zeros_like(D_custom)
    V_expD_custom = np.zeros_like(V_custom)
    expA_spec_custom = np.zeros_like(V_custom)
    
    exp_mat2_diag(D_custom, expD_custom)
    mult_mat2_full_diag(V_custom, expD_custom, V_expD_custom)
    mult_mat2_full(V_expD_custom, invV_custom, expA_spec_custom)
    print('||expA - expA_spec_custom|| = {}'.format(np.linalg.norm(expA - expA_spec_custom)))
    
    # matrix times vector    
    b = np.random.rand(N) + 1j*np.random.rand(N)
    Ab = np.dot(A, b)
    
    Ab_custom = np.zeros_like(b, dtype=np.complex)
    mult_mat2_full_vec(A, b, Ab_custom)
    print('||A*b - A*b_custom||={}'.format(np.linalg.norm(Ab - Ab_custom)))

    print('3) BEST EXAMPLE. LINEAR COMBINATION OF THE MATRIX AND IDENTIY BASED ON CAYLEY-HAMILTON THM')
    expA_spec_best = np.zeros_like(A)
#    A[0,0] = 0
#    A[0,1] = 0
#    A[1,0] = 0
    exp_mat2_full_scalar(A, 1.0, expA_spec_best)  
    print('||expA - expA_spec_best|| = {}'.format(np.linalg.norm(scipy.linalg.expm(A) - expA_spec_best))) 
    