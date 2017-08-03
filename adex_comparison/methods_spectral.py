# -*- coding: utf-8 -*-

'''
Solver framework for the eigenvalue problem associated with the Fokker-Planck operator of 
(linear or non-linear) integrate-and-fire model neurons subject to white noise input.

Work started in January 2014; current version: July 2017 

Please cite the publication which has introduced the solver if you want to use it: 
    Augustin, Ladenbauer, Baumann, Obermayer (2017) PLOS Comput Biol

@author: Moritz Augustin <augustin@ni.tu-berlin.de>
'''

# TODO: 
# - remove richardson & odeint code + parameters [keep backup and visualize the effect]
# - comment functions
# - comment parameters in params.py
# - remove old stuff
# - verbosity: 2-3 levels. default not much, middle: more info but not every iteration, 3rd: all root results per step
# - cleanup params
# - warn upon tref=0 (refer to paper)
# - catch the following exception (only show them in verbose mode)
    #/usr/lib/python2.7/dist-packages/scipy/optimize/minpack.py:236: RuntimeWarning: The iteration is not making good progress, as measured by the 
    #  improvement from the last ten iterations.
    #  warnings.warn(msg, RuntimeWarning)




from __future__ import division # omit unwanted integer division
import math
import cmath
from warnings import warn, simplefilter
import numpy as np
from scipy.optimize import root
import scipy.linalg
import matplotlib.pyplot as plt
import multiprocessing
import itertools
import time
import tables
# for plotting
import matplotlib.colors as colors
import matplotlib.cm as cmx
import os

import misc.utils

enforce_numba = False # when true: if numba is not installed define dummy jit 
                      # decorators to allow (=> slow calculations not requiring numba)
try:
    from numba import jit, njit
except:
        if enforce_numba == False:
            print(('Please install the numba-package for fast computation (strongly '
                 +'recommended) or set "enforce_numba" to False in {}! Exiting.').format(os.path.basename(__file__)))
            exit()
        else:
            warn('numba missing, skipping just in time compilation')
            jit = lambda func: func
            njit = jit


def compute_eigenvalue_curve_given_sigma(arg_tuple):
        
    
    params, mu_arr, sigma_arr, j, N_eigvals, eigenval_init_real_grid = arg_tuple

    N_mu = mu_arr.shape[0]    
    
    comp_single_start = time.time()

    specsolv = SpectralSolver(params.copy())

    mu_dummy =  np.concatenate([mu_arr.reshape(-1, 1), np.zeros((N_mu, 1))], axis=1)
    mu_refined_dummy = specsolv.refine_mu_sigma_curve(mu_dummy) # refines the mu_sigma within specsolv also
    mu_refined = mu_refined_dummy[:, 0].reshape(-1, 1)
    N_mu_refined = mu_refined_dummy.shape[0]
    
    lambda_j = np.zeros((N_eigvals, N_mu_refined), dtype=np.complex128) # mu x modes
    
    
    mu_sigma = np.concatenate([mu_arr.reshape(-1, 1), sigma_arr[j]*np.ones((N_mu, 1))], 
                              axis=1)
    
    mu_sigma_refined = specsolv.refine_mu_sigma_curve(mu_sigma) # refines the mu_sigma within specsolv also

    mu_init, sigma_init = mu_sigma_refined[0, :]
    specsolv.params['mu'] = mu_init
    specsolv.params['sigma'] = sigma_init

    lambda_real_init, ode_real_init, ode_real_qlb = specsolv.real_eigenvalues(eigenval_init_real_grid, min_ev=N_eigvals)
    
    lambda_real_init = lambda_real_init[:N_eigvals]
    ode_real_init = ode_real_init[:N_eigvals]
    
    
    print('after having real inits compute N_eigvals={} eigenvalue curves for sigma={}'.format(N_eigvals, sigma_arr[j]))    
    
    for m in range(N_eigvals): 
        lambda_init_complex = complex(lambda_real_init[m])
        specsolv.params['init_q_r'] = ode_real_init[m] # to initialize odeint close to normalized eigenfunctions
        
        if params['verboselevel'] >= 1:
            print('======================= computing eigenvalue curve for mode {m} and sigma {sigma_init}'.format(**locals()))
            
        try:
            lambda_j[m, :] = specsolv.eigenvalue_curve(lambda_init_complex, mu_sigma_refined, m)
        except Exception as exce:
            error('convergence failed for sigma={sigma}, m={m}, exception: {exce}'.format(sigma=sigma_arr[j], 
                  m=m, exce=exce))
    
    
    comp_single_duration = time.time() - comp_single_start
    
    # collapse lambda using only the prescribed mu values (not the refined ones)
    inds_collapsed = [i for i in range(N_mu_refined) if mu_refined[i] in mu_arr]
    lambda_j_collapsed = lambda_j[:, inds_collapsed]
    
    return (j, lambda_j_collapsed, comp_single_duration)


def  compute_quantities_given_sigma(arg_tuple):
    # params, quant_names, lambda_1, lambda_2, mu_arr, sigma_arr, j = arg_tuple
    params, quant_names, lambda_all, lambda_1, lambda_2, N_eigvals, mu_arr, sigma_arr, j = arg_tuple

    assert N_eigvals <= lambda_all.shape[0]
    N_mu = mu_arr.shape[0]
    sigma_j = sigma_arr[j]
    dmu = params['dmu_couplingterms']
    dsigma = params['dsigma_couplingterms']
    
    quant_j = {}

    # start computation time
    comp_single_start = time.time()

    for q in quant_names:

        # real quantities
        # do not depend on the number of eigenvalues
        if q in ['r_inf', 'dr_inf_dmu', 'dr_inf_dsigma', 
                 'V_mean_inf', 'dV_mean_inf_dmu', 'dV_mean_inf_dsigma']:
            quant_j[q] = np.zeros(N_mu)

        elif q in ['f', 'psi_r', 'c_mu', 'c_sigma']:
            # matrix for each j_sigma curve
            quant_j[q] = np.zeros((N_eigvals, N_mu)) + 0j # complex type

        elif q in ['C_mu', 'C_sigma']:
                # 3-dim. array for C_mu & C_sigma
                quant_j[q] = np.zeros((N_eigvals, N_eigvals, N_mu)) +0j # complex type

        # quantity is not known
        else:
            error('unknown/unsupported quantity {}'.format(q))
    
    params_quants = params.copy()
    
    specsolv = SpectralSolver(params_quants)


# optional code useful for incorporating a spike shape from Josef/Richardson: 
# p0 integrates to unity, r0 respective r_inf
#r0ref = r0/(1+r0*Tref);
#p0 = r0ref * p0/r0;
#%prob. density only reflecting nonrefr. proportion which integrates to r0ref/r0
#Vmean_sps = dV*sum(V.*p0) + (1-r0ref/r0)*(Vs+Vr)/2;  %remark: (1-r0ref/r0)==r0ref*Tref


    def phi0_rinf_noref(mu, sigma):
        # use global caching variables to prevent redundant computation of stationary mode
        global phi_0_noref_cache, r_inf_noref_cache, V_arr_cache 
        if phi_0_noref_cache[(mu, sigma)] is None: # only compute stationary mode if not already stored temporarily            
            V_arr_cache, phi_0_noref, q_0 = specsolv.eigenfunction(0., mu, sigma) # 0 is always an eigenvalue
            phi_0_noref = phi_0_noref.real
            norm_fac = normalize(phi_0_noref, V_arr_cache) # ensure that phi_0(V) integrates to unity
            r_inf_noref = q_0[-1].real/norm_fac # the initialiation of the backwards integration is the threshflux
            # store into cache            
            phi_0_noref_cache[(mu, sigma)] = phi_0_noref
            r_inf_noref_cache[(mu, sigma)] = r_inf_noref
        else:
            # load from cache
            phi_0_noref = phi_0_noref_cache[(mu, sigma)]
            r_inf_noref = r_inf_noref_cache[(mu, sigma)]
        
        
        
        return V_arr_cache, phi_0_noref, r_inf_noref
        
    # assuming existence of: V_arr, phi_0_noref, r_inf_noref
    def rinf_ref(mu, sigma):
        # get stationary distribution neclecting refractory period
        V_arr, phi_0_noref, r_inf_noref = phi0_rinf_noref(mu, sigma)
        
        # respect the refractory period which is assumed to be nonnegative
        tau_ref = params['tau_ref']
        assert tau_ref >= 0
        r_inf = 1.0 / (1.0/r_inf_noref + tau_ref)
        
        return r_inf
        
    def Vmeaninf_noref(mu, sigma):
        # get stationary distribution neclecting refractory period
        V_arr, phi_0_noref, r_inf_noref = phi0_rinf_noref(mu, sigma)
        
        # use midpoint integration for integral(V*phi0(V)*dV)
        V_center = 0.5 * (V_arr[:-1] + V_arr[1:])
        phi0_center = 0.5 * (phi_0_noref[:-1] + phi_0_noref[1:])
        V_mean_inf_noref =  np.sum(V_center * phi0_center * np.diff(V_arr))

        return V_mean_inf_noref
        
    
    # computing eigenfunction psi_1 of adjoint operator, lambda_1 should be an eigenvalue
    def psi1(mu, sigma, lambda_1, cache=True):
        # use global caching variable to prevent redundant computation
        global psi_1_cache, V_arr_cache
        if not cache or psi_1_cache is None: # only compute stationary mode if not already stored temporarily            
            V_arr_cache, psi_1_cache, dpsi = specsolv.eigenfunction(lambda_1, mu, sigma, 
                                                              adjoint=True) 
        return V_arr_cache, psi_1_cache
        
    
    # computing eigenfunction psi_2 of adjoint operator, lambda_2 should be an eigenvalue
    def psi2(mu, sigma, lambda_2, cache=True):
        # use global caching variable to prevent redundant computation
        global psi_2_cache, V_arr_cache
        if not cache or psi_2_cache is None: # only compute stationary mode if not already stored temporarily            
            V_arr_cache, psi_2_cache, dpsi = specsolv.eigenfunction(lambda_2, mu, sigma, 
                                                              adjoint=True) 
        return V_arr_cache, psi_2_cache

    
    def phi1(mu, sigma, lambda_1):
        V_arr, psi_1 = psi1(mu, sigma, lambda_1)
        V_arr, phi_1, q_1 = specsolv.eigenfunction(lambda_1, mu, sigma)
        # binormalize phi_1 s.t. the inner product (psi_1, phi_1) = 1
        inprod1 = inner_prod(psi_1, phi_1, V_arr)
        if params['verboselevel'] > 0:
            print('inner product (phi1)={}'.format(inprod1))
        phi_1 /= inprod1
        q_1 /= inprod1
        
        return V_arr, phi_1, q_1
        
    def phi2(mu, sigma, lambda_2):
        V_arr, psi_2 = psi2(mu, sigma, lambda_2)
        V_arr, phi_2, q_2 = specsolv.eigenfunction(lambda_2, mu, sigma)
        # binormalize phi_2 s.t. the inner product (psi_2, phi_2) = 1
        inprod2 = inner_prod(psi_2, phi_2, V_arr)
        if params['verboselevel'] > 0:
            print('inner product (phi2)={}'.format(inprod2))
        phi_2 /= inprod2
        q_2 /= inprod2
        
        return V_arr, phi_2, q_2


    # general helpers (N)

    # extend for caching later
    def psiN(mu, sigma, lambda_N):
        V_arr, psi_N, dpsi = specsolv.eigenfunction(lambda_N, mu, sigma,
                                                    adjoint=True)
        return V_arr, psi_N


    def phiN(mu, sigma, lambda_N):
        V_arr, psi_N = psiN(mu, sigma, lambda_N)
        V_arr, phi_N, q_N = specsolv.eigenfunction(lambda_N, mu, sigma)
        inprodN = inner_prod(psi_N, phi_N, V_arr)
        # if params['verboselevel'] > 0:
            # print('inner product (phi{})={}'.format(N, inprodN))
        phi_N /= inprodN
        q_N /= inprodN
        return V_arr, phi_N, q_N

    def fn(mu, sigma, lambda_n):
        V_arr, phi_n, q_n = phiN(mu, sigma, lambda_n)
        f_n = q_n[-1]
        return f_n


    specsolve = SpectralSolver(params.copy())
                             
    # method for obtaining an eigenvalue at mu, sigma specified mu_sigma_target
    # note that a simple call of specsolve.eigenvalue() is not  sufficient due 
    # to sensitive areas (=> 'robust')
    def eigenvalue_robust(mu_sigma_init, mu_sigma_target, lambda_init):
        mu_init, sigma_init = mu_sigma_init
        mu_target, sigma_target = mu_sigma_target
        mu_sigma_curve = np.array([[mu_init, sigma_init],
                                   [mu_target, sigma_target]])
        # alternative to the following: try/except -- interpolate coarse grid of lambda_all if error
        lambda_curve = specsolve.eigenvalue_curve(lambda_init, mu_sigma_curve)
        lambda_target = lambda_curve[-1]
        
        return lambda_target

    # loop over N_mu
    for i in range(N_mu):
        # abbreviations
        mu_i = mu_arr[i]
        # lambda_1_ij = lambda_1[i, j]
        # lambda_2_ij = lambda_2[i, j]

        # TODO: move this to verbose 1 (but at least the initialization of the curve should be indicated)
        print('======================= mu={}, sigma={} == computing quantities'.format(mu_i, sigma_j))

        # create variable for temporarily stored functions such as phi0 
        # which can be accessed from the following helper functions
        # all mu/sigma combinations required as for each pair r_inf and V_mean_inf depend on them
        global phi_0_noref_cache
        phi_0_noref_cache = {(mu_i, sigma_j):       None,
                             (mu_i+dmu, sigma_j):   None,
                             (mu_i-dmu, sigma_j):   None,
                             (mu_i, sigma_j+dsigma): None,
                             (mu_i, sigma_j-dsigma): None,}
        # steady state rate is better not computed via finite difference of phi0 but rather 
        # by the threshold flux which is only available after computing phi0, therefore cache r_inf
        global r_inf_noref_cache
        r_inf_noref_cache = phi_0_noref_cache.copy() # note deepcopy would be needed for nested dict
    
        # assuming same V grid for all mu sigma and all eigenfunctions
        global V_arr_cache
        V_arr_cache = None 
        
        # psi1,2 are required for (i) f_1,2 due to phi normalization => biorth, and (ii) psi_r_* itself
        # (however, we disable this caching for computation of c_k_x quantities 
        # where psi(mu+-dmu,sigma+-dsigma) are required)
        global psi_1_cache
        psi_1_cache = None
        global psi_2_cache
        psi_2_cache = None

        # calculate quantities 
        for q in quant_names:
            
            # compute quantities which do not depend on lambda
            if q == 'r_inf':
                r_inf = rinf_ref(mu_i, sigma_j)
                quant_j[q][i] = r_inf
            
            if q == 'dr_inf_dmu':
                r_inf_plus_mu = rinf_ref(mu_i+dmu, sigma_j)
                r_inf_minus_mu = rinf_ref(mu_i-dmu, sigma_j)
                # central difference quotient
                quant_j[q][i]  = (r_inf_plus_mu - r_inf_minus_mu) / (2*dmu)
            
            if q == 'dr_inf_dsigma':
                r_inf_plus_sigma = rinf_ref(mu_i, sigma_j+dsigma)
                r_inf_minus_sigma = rinf_ref(mu_i, sigma_j-dsigma)
                # central difference quotient
                quant_j[q][i] = (r_inf_plus_sigma - r_inf_minus_sigma) / (2*dsigma)
            
            if q == 'V_mean_inf':
                quant_j[q][i] = Vmeaninf_noref(mu_i, sigma_j)
            
            if q == 'dV_mean_inf_dmu':
                V_mean_inf_plus_mu = Vmeaninf_noref(mu_i+dmu, sigma_j)
                V_mean_inf_minus_mu = Vmeaninf_noref(mu_i-dmu, sigma_j)
                # central difference quotient
                quant_j[q][i] = (V_mean_inf_plus_mu - V_mean_inf_minus_mu) / (2*dmu)
            
            if q == 'dV_mean_inf_dsigma':
                V_mean_inf_plus_sigma = Vmeaninf_noref(mu_i, sigma_j+dsigma)
                V_mean_inf_minus_sigma = Vmeaninf_noref(mu_i, sigma_j-dsigma)
                # central difference quotient
                quant_j[q][i] = (V_mean_inf_plus_sigma - V_mean_inf_minus_sigma) / (2*dsigma)


            # f_k is the flux of the k-th eigenfunction phi_k evaluated at the threshold V_s
            # (V_s is the right most point in the V_arr grid cell where we use backwards finite
            # differences respecting the absorbing boudary condition)
            # note that in contrast to the manuscript we normalize phi that yields q_Nv != 1
            # in general and thus f_1, f_2 != 1

            # loop over the N lambdas
            # test with eigenfluxes 'f'
            if q in ['f', 'psi_r', 'c_mu', 'c_sigma']:
                for n in xrange(N_eigvals):
                    # get the eigenvalue n for the respective mu, sigma
                    lambda_n_ij = lambda_all[n, i, j]
                    # vector of f's
                    if q == 'f':
                        V_arr, phi_n, q_n = phiN(mu_i, sigma_j, lambda_n_ij)
                        f_n = q_n[-1]

                        # save in quant_j-dict
                        quant_j[q][n][i] = f_n

                    # vector of psi_r's
                    # psi_r_k is just the eigenfunction psi_k evaluated at the reset
                    if q == 'psi_r':
                        V_arr, psi_n = psiN(mu_i, sigma_j, lambda_n_ij)
                        k_r = np.argmin(np.abs(V_arr-params['V_r']))
                        psi_r_n = psi_n[k_r]
                        #save in quant_j-dict
                        quant_j[q][n][i] = psi_r_n

                    # vector of c_mu
                    # inner product between (discretized) partial derivative of psi w.r.t mu and
                    # the stationary distribution of active (i.e. non refractory) neurons
                    if q == 'c_mu':
                        # we need to evaluate psi_n(mu+-dmu) and thus lambda_n(mu+-dmu)
                        lambda_n_plus_mu = eigenvalue_robust((mu_i, sigma_j), (mu_i+dmu, sigma_j), lambda_n_ij)
                        V_arr, psi_n_plus_mu = psiN(mu_i+dmu, sigma_j, lambda_n_plus_mu)

                        lambda_n_minus_mu = eigenvalue_robust((mu_i, sigma_j), (mu_i-dmu, sigma_j), lambda_n_ij)
                        V_arr, psi_n_minus_mu = psiN(mu_i-dmu, sigma_j, lambda_n_minus_mu)

                        # discretization of the partial derivative of psi w.r.t mu
                        dpsi_n_dmu = (psi_n_plus_mu - psi_n_minus_mu)/(2*dmu)
                        V_arr, phi_0_noref, r_inf_noref = phi0_rinf_noref(mu_i, sigma_j)

                        # compute inner product
                        c_mu_n = inner_prod(dpsi_n_dmu, phi_0_noref, V_arr)

                        # save in quant_j-dict
                        quant_j[q][n][i] = c_mu_n

                    # vector of c_sigma
                    if q == 'c_sigma':

                        # we need to evaluate psi_n(sigma+-dsigma) and thus lambda_n(sigma+-dsigma)
                        lambda_n_plus_sigma = eigenvalue_robust((mu_i, sigma_j), (mu_i, sigma_j+dsigma), lambda_n_ij)
                        V_arr, psi_n_plus_sigma = psiN(mu_i, sigma_j+dsigma, lambda_n_plus_sigma)

                        lambda_n_minus_sigma = eigenvalue_robust((mu_i, sigma_j), (mu_i, sigma_j-dsigma), lambda_n_ij)
                        V_arr, psi_n_minus_sigma = psiN(mu_i, sigma_j-dsigma, lambda_n_minus_sigma)

                        # discretization of the partial derivative of psi w.r.t sigma
                        dpsi_n_dsigma = (psi_n_plus_sigma - psi_n_minus_sigma)/(2*dsigma)
                        V_arr, phi_0_noref, r_inf_noref = phi0_rinf_noref(mu_i, sigma_j)

                        c_sigma_n = inner_prod(dpsi_n_dsigma, phi_0_noref, V_arr)

                        #save in quant_j-dict
                        quant_j[q][n][i] = c_sigma_n

            if q in ['C_mu', 'C_sigma']:
                for k in range(N_eigvals):
                    for l in range(N_eigvals):
                        lambda_k_ij = lambda_all[k, i, j]
                        lambda_l_ij = lambda_all[l, i, j]
                        if q == 'C_mu':
                            lambda_k_plus_mu = eigenvalue_robust((mu_i, sigma_j), (mu_i+dmu, sigma_j), lambda_k_ij)
                            V_arr, psi_k_plus_mu = psiN(mu_i+dmu, sigma_j, lambda_k_plus_mu)

                            lambda_k_minus_mu = eigenvalue_robust((mu_i, sigma_j), (mu_i-dmu, sigma_j), lambda_k_ij)
                            V_arr, psi_k_minus_mu = psiN(mu_i-dmu, sigma_j, lambda_k_minus_mu)

                            # compute finite-difference (partial derivative)
                            dpsi_k_dmu = (psi_k_plus_mu-psi_k_minus_mu)/(2*dmu)

                            ## Phi
                            V_arr, phi_l, q_l = phiN(mu_i, sigma_j, lambda_l_ij)
                            # compute inner product
                            C_mu_kl = inner_prod(dpsi_k_dmu, phi_l, V_arr)

                            # save in quant_j-dict
                            quant_j[q][k][l][i] = C_mu_kl


                        if q == 'C_sigma':
                            # todo: check this again
                            lambda_k_plus_sigma = eigenvalue_robust((mu_i, sigma_j), (mu_i, sigma_j+dsigma), lambda_k_ij)
                            V_arr, psi_k_plus_sigma = psiN(mu_i, sigma_j+dsigma, lambda_k_plus_sigma)

                            lambda_k_minus_sigma = eigenvalue_robust((mu_i, sigma_j), (mu_i, sigma_j-dsigma), lambda_k_ij)
                            V_arr, psi_k_minus_sigma = psiN(mu_i, sigma_j-dsigma, lambda_k_minus_sigma)

                            # compute finite-difference (partial derivative)
                            dpsi_k_dsigma = (psi_k_plus_sigma-psi_k_minus_sigma)/(2*dsigma)

                            ## Phi
                            V_arr, phi_l, q_l = phiN(mu_i, sigma_j, lambda_l_ij)
                            # compute inner product
                            C_sigma_kl = inner_prod(dpsi_k_dsigma, phi_l, V_arr)

                            # save in quant_j-dict
                            quant_j[q][k][l][i] = C_sigma_kl


    comp_single_duration = time.time() - comp_single_start

    return j, quant_j, comp_single_duration




def spectrum_enforce_complex_conjugation(lambda_all, mu_arr, sigma_arr, 
                                         tolerance_conjugation=1e-5, merge_dist=2, 
                                         conjugation_first_imag_negative=False):
    
    # merge_dist: no of mu points in which the real parts must be close enough in terms of the following tolerance
    
    # tolerance_conjugation:    # eigenvalues differing this amount in their real part  
                                    # are assumed to be complex conjugates of each other from
                                    # that point
                                    # if this is true for at least merge_dist points
    
    N_eigvals = lambda_all.shape[0]
    N_sigma = lambda_all.shape[2]

    print('START: enforcing complex conjugated spectrum in raw eigenvalue data: lambda_all')
                                                                   
    # also done: 
    # imaginary parts below the detected 2xreal->complex conj. pair 
    # merge point are considered to be zero 

    #if two eigenvalues merge and become a complex conj. pair then it is ensured that both the eigenvalue with positiv
    #and negative part exist; merging = real parts are close and imag part > 0 (right of merge point which is not in grid)
    for j in range(N_sigma):
        for tup in itertools.combinations(range(N_eigvals), 2):
            diff = lambda_all[tup[0],:,j] - lambda_all[tup[1],:,j]
            idx_zero = np.where((np.abs(np.real(diff))<tolerance_conjugation) &
                                (np.abs(lambda_all[tup[0],:,j].imag) > tolerance_conjugation))[0]
            if len(idx_zero) > merge_dist:
                # tup[0] and tup[1] have merged at (the left of) mu-index idx_zero[0]
                i_merge = idx_zero[0]
                
                print('detected complex conjugated pair for lambda inds ({}, {}) left of mu={} with sigma={}'
                .format(tup[0], tup[1], mu_arr[i_merge], sigma_arr[j]))
                
                # cleanup: imaginary parts before the merge index
                lambda_all[tup[0], :i_merge, j] = lambda_all[tup[0], :i_merge, j].real
                lambda_all[tup[1], :i_merge, j] = lambda_all[tup[1], :i_merge, j].real

                # cleanup 2: make the first eigval of the complex conjugate pair having a 
                # positive imaginary part and the 2nd the complex conjugate of it (neg. imag.).
                # point-wise might not be most efficient but repairs curve switching, too
                # added switch to allow first imag negative as maurizio
                for idx in idx_zero: 
                    
                    if not conjugation_first_imag_negative:
                        if lambda_all[tup[0],idx,j].imag < 0:
                            lambda_all[tup[0],idx,j] += 2*abs(lambda_all[tup[0],idx,j].imag)*1j
                    else:
                        if lambda_all[tup[0],idx,j].imag > 0:
                            lambda_all[tup[0],idx,j] -= 2*abs(lambda_all[tup[0],idx,j].imag)*1j
                    
                    lambda_all[tup[1],idx,j] = np.conj(lambda_all[tup[0],idx,j])
                    
                    
                # how we should only have true real or true complex conjugates, 
                # i.e., we can detect the transition by imag>0
    
    
    # no return required. inplace change
    print('DONE: enforcing complex conjugated spectrum done')
    print('')



class SpectralSolver(object):
    
    def __init__(self, params):
        if params['verboselevel'] >= 1:
            print('constructing spectral solver object with params:')
            print(params)
        self.params = params        
        self.ode_steps = []


    def compute_eigenvalue_rect(self, mu_arr, sigma_arr, N_eigvals, eigenval_init_real_grid, 
                                N_procs=multiprocessing.cpu_count()):
                                
        print('START computing full spectrum with '+str(N_eigvals)+' eigenvalues on mu, sigma grid')
    
        
        N_mu = mu_arr.shape[0]
        N_sigma = sigma_arr.shape[0]
        lambda_all = np.zeros((N_eigvals, N_mu, N_sigma), dtype=np.complex128)
    
        
        arg_tuple_list = [(self.params, mu_arr, sigma_arr, j, N_eigvals, eigenval_init_real_grid) 
                            for j in range(N_sigma)]
        
        comp_total_start = time.time()

        if N_procs <= 1:
            # single processing version, i.e. loop
            pool = False
            result = (compute_eigenvalue_curve_given_sigma(arg_tuple) for arg_tuple in arg_tuple_list)
            
        else:
            # multiproc version
            pool = multiprocessing.Pool(N_procs, maxtasksperchild=5)
            result = pool.imap_unordered(compute_eigenvalue_curve_given_sigma, arg_tuple_list)
        
        finished = 0
        for j, lambda_j, runtime in result:
            finished += 1
            print(('mu-curve of {eigs} eigenvalues computed for sigma={sig} in {rt:.2}s, completed: {comp}/{tot}').
                  format(eigs=N_eigvals, sig=sigma_arr[j], rt=runtime, comp=finished, tot=N_sigma))
            lambda_all[:, :, j] = lambda_j
        
        if pool:
            pool.close()
        
        print('total time for computation (N_eigvals={Neig}, N_mu={Nmu}, N_sigma={Nsig}): {rt:.2}s'.
              format(rt=time.time()-comp_total_start, Neig=N_eigvals, Nmu=N_mu, Nsig=N_sigma))
    
        print('DONE: computing full spectrum')    
    
        return lambda_all
    

    def compute_quantities_rect(self, quantities_dict, 
                quant_names=['r_inf', 'dr_inf_dmu', 'dr_inf_dsigma',
                             'V_mean_inf', 'dV_mean_inf_dmu', 'dV_mean_inf_dsigma',
                             'f', 'psi_r', 'c_mu', 'c_sigma',
                             'C_mu', 'C_sigma'], N_eigvals = 2,
                              N_procs=multiprocessing.cpu_count()):


        print('START: computing quantity rect: {}'.format(quant_names))

        # lambda_1 & lambda_2 --> lambda_all
        lambda_1 = quantities_dict['lambda_1']
        lambda_2 = quantities_dict['lambda_2']
        lambda_all = quantities_dict['lambda_all']


        mu_arr = quantities_dict['mu']
        sigma_arr = quantities_dict['sigma']
        
        N_mu = mu_arr.shape[0]
        N_sigma = sigma_arr.shape[0]

        
        # zero quantities_dict arrays
        for q in quant_names:
            
            # real quantities
            if q in ['r_inf', 'dr_inf_dmu', 'dr_inf_dsigma', 
                     'V_mean_inf', 'dV_mean_inf_dmu', 'dV_mean_inf_dsigma']:
                quantities_dict[q] = np.zeros((N_mu, N_sigma))

            elif q in ['f', 'psi_r', 'c_mu', 'c_sigma']:
                quantities_dict[q] = np.zeros((N_eigvals, N_mu, N_sigma)) +0j # complex type


            # complex quants
            elif q in ['C_mu', 'C_sigma']:
                quantities_dict[q] = np.zeros((N_eigvals, N_eigvals, N_mu, N_sigma)) + 0j # complex dtype

        arg_tuple_list = [(self.params, quant_names, lambda_all, lambda_1, lambda_2, N_eigvals, mu_arr, sigma_arr, j)
                          for j in range(N_sigma)]
        
        comp_total_start = time.time()
        if N_procs <= 1:
            # single processing version, i.e. loop
            pool = False
            result = (compute_quantities_given_sigma(arg_tuple) for arg_tuple in arg_tuple_list)


        else:
            # multiproc version
            pool = multiprocessing.Pool(N_procs, maxtasksperchild=5)
            result = pool.imap_unordered(compute_quantities_given_sigma, arg_tuple_list)
        
        finished = 0
        for j, quantities_j_dict, runtime in result:

            finished += 1
            print(('mu-curve of quantities computed for sigma={sig} in {rt:.2}s, completed: {comp}/{tot}').
                  format(sig=sigma_arr[j], rt=runtime, comp=finished, tot=N_sigma))
            for q in quantities_j_dict.keys():
                if q in ['C_mu', 'C_sigma']:
                    quantities_dict[q][:, :, :, j] = quantities_j_dict[q]
                elif q in ['f', 'psi_r', 'c_mu', 'c_sigma']:
                    quantities_dict[q][:, :, j] = quantities_j_dict[q]
                elif q in ['r_inf']: # add the stationary quantities as well
                    quantities_dict[q][:, j] = quantities_j_dict[q]


        
        if pool:
            pool.close()
        
        print('total time for quantitiy computation (N_mu={Nmu}, N_sigma={Nsig}): {rt:.2}s'.
              format(rt=time.time()-comp_total_start, Nmu=N_mu, Nsig=N_sigma))
    
        
        
        
        print('DONE: computing quantity rect for quantities: {}'.format(quant_names))
        print('')
        
        
        return quantities_dict
        
        
    
    def load_quantities(self, filepath, 
                        quantities_dict,
                        # todo change default loading quantities
                        quantities=['lambda_all', 'mu', 'sigma', 
                                    'lambda_1', 'lambda_2', 
                                    'r_inf', 'dr_inf_dmu', 'dr_inf_dsigma',
                                    'V_mean_inf', 'dV_mean_inf_dmu', 'dV_mean_inf_dsigma',
                                    'f_1', 'f_2', 'psi_r_1', 'psi_r_2',
                                    'c_mu_1', 'c_mu_2', 'c_sigma_1', 'c_sigma_2'],
                        load_params=True):
    
        print('loading quantities {} from file {}'.format(quantities, filepath))
        try:
            h5file = tables.open_file(filepath, mode='r')
            root = h5file.root
            
            for q in quantities:
                quantities_dict[q] = h5file.get_node(root, q).read()            
                       
            if load_params:
                params = self.params
                # loading params
                # only overwrite what is in the file. not start params from scratch. otherwise: uncomment following line
                #params = {} 
                for child in root.params._f_walknodes('Array'):
                    params[child._v_name] = child.read()[0]
                for group in root.params._f_walk_groups():
                    if group != root.params: # walk group first yields the group itself then its children
                        params[group._v_name] = {}
                        for subchild in group._f_walknodes('Array'):
                            params[group._v_name][subchild._v_name] = subchild.read()[0]            
            
            h5file.close()
        
        except IOError:
            error('could not load quantities from file '+filepath)
        except:
            h5file.close()
        
        print('')

    
    # for spectrum include the keys 'mu', 'sigma', 'lambda_all' into the dictionary
    # for quantities include keys 'lambda_1', 'f_2', etc. (see also load_quantities)
    def save_quantities(self, filepath, quantities_dict, save_params=True, save_rawspec=True):
        print('saving quantities {} into file {}'.format(quantities_dict.keys(), filepath))
    
        try:
            h5file = tables.open_file(filepath, mode='w')
        
            root = h5file.root
            
            
            for q in quantities_dict.keys():
                print(q)
                if q != 'lambda_all' or save_rawspec: # only save lambda_all if save_rawspec
                    h5file.create_array(root, q, quantities_dict[q])
                    print('created array {}'.format(q))
                
            if save_params:
                params = self.params
                h5file.create_group(root, 'params', 'Neuron model and numerics parameters')
                for name, value in params.items():
                    # for python2/3 compat.:
                    # if isinstance(name, str):
                    # name = name.encode(encoding='ascii') # do not allow unicode keys
                    if isinstance(value, (int, float, bool)):
                        h5file.create_array(root.params, name, [value])
                    elif isinstance(value, str):
                        h5file.create_array(root.params, name, [value.encode(encoding='ascii')])
                    elif isinstance(value, dict):
                        params_sub = h5file.create_group(root.params, name)
                        for nn, vv in value.items():
                            # for python2/3 compat.:
                            # if isinstance(nn, str):
                            # nn = nn.encode(encoding='ascii') # do not allow unicode keys
                            if isinstance(vv, str):
                                h5file.create_array(params_sub, nn, [vv.encode(encoding='ascii')])
                            else:
                                h5file.create_array(params_sub, nn, [vv])
            h5file.close()
        
        except IOError:
            warn('could not write quantities into file {}'.format(filepath))
        except:
            h5file.close()
    
        print('')
    
    
    # eigen_ind: only due to debug outputs
    def eigenvalue_curve(self, lambda_init, mu_sigma, eigen_ind='unknown', real=False, adjoint=False):
        # TODO: check inputs (lambda_init complex and mu_sigma 2 cols)
        # TODO: debug outputs if debug is on
    
        if adjoint:
            warn('adjoint normalization not working properly, better use non-adjoint here or large abs tol if really necessary')
        
        len_mu_sigma = mu_sigma.shape[0]
        lambda_arr = np.zeros(len_mu_sigma, dtype=np.complex128)
        
        # for better pydev code analysis
        lastmu = None
        lastsigma = None
        lastlambda = None
        lastmu_diff = None
        lastsigma_diff = None
        lastlambda_diff = None


        # whenever convergence fails despite of the nested solver structure we set 
        # the eigenvalue to the previous eigenvalue and increase the following counter
        # which is only reset when a successful eigenvalue computation happened
        workaround_usage = 0
        self.params['solver_xtol_orig'] = self.params['root_options']['xtol']
        self.params['solver_abstol_orig'] = self.params['solver_abstol']
        
        for i in range(len_mu_sigma):
            
            
            mu, sigma = mu_sigma[i, :]
            
            converged_outer = False # true if the eigenvalue for mu, sigma has been found
            
            # converged_outer flags whether for the target mu/sigma (i.e. those of mu_sigma)
            #    the corresponding eigenvalue could be successfully computed
            #    if no convergence achieavable mu_calc/sigma_calc are chosen to be closer to the 
            #    previously mu/sigma of the last step with for sure successfully computed eigenvalue
            #    the latter is then repeated maxrefinements times and success indicated by 
            # variable converged_inner
            while not converged_outer:
                
                root_extra_options = {}
                converged_inner = False # true if the eigenvalue for mu_calc, sigma_calc has been found
                maxrefinements = self.params['solver_maxrefinements']
                k_refinement = 0
                mu_calc = mu
                sigma_calc = sigma
                while not converged_inner and k_refinement<maxrefinements:
                    
                    self.ode_steps = [] # count only for current mu,sigma the ode steps
                    
                    self.params['mu'] = mu_calc
                    self.params['sigma'] = sigma_calc
                        
                    # the case i==0 is handled by the function's argument
                    if i==1: # first initial value is given to the function (for i==0)
                        lambda_init = lastlambda
#                        lastmu = mu # only for quantities required
                    if i>1:
                        if self.params['solver_init_approx_type'] == 'linear':
                            # linear approximation in the direction of last_mu_sigma_diff (and NOT mu_sigma_diff i.e. this
                            # approximate directional derivative introduces a small error)
                            norm_mu_sigma_diff = np.sqrt((mu_calc-lastmu)**2 + (sigma_calc-lastsigma)**2)
                            norm_last_mu_sigma_diff = np.sqrt(lastmu_diff**2 + lastsigma_diff**2)
                            lambda_init = lastlambda + lastlambda_diff * norm_mu_sigma_diff/norm_last_mu_sigma_diff
                        else: # i.e. solver_init_approx_type == 'constant'
                            lambda_init = lastlambda                    
                    
                    # minpack's hybrid solver for nonlinear systems behaves badly when one 
                    # variable has values close to zero but not exactly zero. in our case this 
                    # is happens for the imaginary part of the initial eigenvalue. therefore we ensure it is zero or 
                    # large enough (i.e. the square root of parameter eps)
                    if i>0 and abs(lambda_init.imag) < (self.params['root_options']['eps'])**0.5:
                        if self.params['verboselevel']>=1:
                            print('hack (due to issue in minpack): adding imaginary perturbation to lambda_init (setting it to zero or sqrt(eps))'+
                              ' to enforce reasonable finite diff step for jacobian (case 0 is already handled in minpack)')
                        zero_one_toggle = 0 if k_refinement % 2 == 0 else 1  # vary with setting the perturb to 0 and sqrt(eps)
                        lambda_init = lambda_init.real + 1j * zero_one_toggle * math.sqrt(self.params['root_options']['eps']) # i.e. always positive imag part
                    
                    try:
                        if self.params['verboselevel']>=1:
                            print('eigen_ind={eigen_ind}, mu={mu}, sigma={sigma}, lambda_init={lambda_init}'.format(**locals()))
                        lambda_calc =  self.eigenvalue(lambda_init, real=real, root_extra_options=root_extra_options, adjoint=adjoint)
                        
                        # continuity check
                        if i>0 and lastlambda!=0 and abs((lambda_calc-lastlambda)/lastlambda) > self.params['solver_smoothtol']:
                            if self.params['verboselevel']>=1:
                                print('continuity condition failed. not accepting converged eigenvalue.')
                            converged_inner = False
                        else:
                            converged_inner = True # if the eigenvalue call did not throw an exception we converged
                    except NotConverged as nc:
                            solver_result = nc.solver_result
                            if self.params['verboselevel']>=1:
                                print('NOT CONVERGED')
                                print(solver_result)
                                print(nc)
                    
                    if self.params['verboselevel']>=1:
                        print('ode steps: {0}'.format(self.ode_steps))
                        
                    
                    if converged_inner:
                        if self.params['verboselevel']>=1:
                            print('CONVERGED')
                        # update last eigenvalue, mu and sigma (last converged data) and their diffs to the next last one)
                        if i>0:
                            lastmu_diff = mu_calc - lastmu
                            lastsigma_diff = sigma_calc - lastsigma
                            lastlambda_diff = lambda_calc - lastlambda
                        lastmu = mu_calc
                        lastsigma = sigma_calc
                        lastlambda = lambda_calc
                        # update also real q (non adjoint) / psi (adjoint) init
                        self.params['init_real'] = float(self.params['init_real_last'])
#                        print('changing init_real to {} with mu={}'.format(self.params['init_real'], mu_calc))
    
                        # reset xtolerance (possibly changed by workaround)
                        self.params['root_options']['xtol'] = self.params['solver_xtol_orig']
                        self.params['solver_abstol'] = self.params['solver_abstol_orig']
    
                    else:
                        if self.params['verboselevel']>=1:
                            print('NOT CONVERGED')
        
                            # compute refinement
                            print('REFINEMENT NECESSARY')
                            print('refinement no {k}'.format(k=k_refinement))
                            print('refinement necessary due to non convergence of:')
                            print('mu={mu}, sigma={sigma}'.format(mu=mu_calc, sigma=sigma_calc))
                            
                        # increase tolerances by a factor of 10 per 5 refinements
                        if workaround_usage>=1 or i==0: # for i=0 we have to go this branch as no last successful mu known
                            if k_refinement % self.params['solver_workaround_tolreduction_refinements'] == 0:
                                self.params['root_options']['xtol'] *= 10.0
                                self.params['solver_abstol'] *= 10.0
                            
                        if i>0: # we can only go to the 'next' mu if the 'last' is already defined   (for i>0)
                            mu_calc = lastmu + (mu_calc-lastmu)/2.0
                            sigma_calc = lastsigma + (sigma_calc-lastsigma)/2.0
                            if self.params['verboselevel']>=1:                        
                                print('new pair: mu={mu}, sigma={sigma}'.format(mu=mu_calc, sigma=sigma_calc))
                                
                        k_refinement+=1
                        
                     
                if not converged_inner:
                    
                    workaround_usage += 1
                    lambda_calc = lambda_arr[i-1]
                    lastlambda = lambda_calc
                    lastmu = mu
                    lastsigma = sigma
                    self.params['root_options']['xtol'] = self.params['solver_xtol_orig']
                    self.params['solver_abstol'] = self.params['solver_abstol_orig']
                    warn(('CONVERGENCE FAILED for mu={mu}, sigma={sigma}, '+
                          'eigen_ind={eigen_ind}. ').format(**locals())+
                         'WORKAROUND: accepting the last successfully computed eigenvalue {}'.format(lastlambda)+
                         ' (workaround #{}), since we are most likely close to the double eigenvalue at the '.format(workaround_usage)+
                         'transition from real to complex.')
                         
                    if workaround_usage >= self.params['solver_workaround_max']:
                        if self.params['verboselevel']>=1:
                            plt.figure()
                            plt.suptitle('eigenvalue which did not converge, eigen_ind={eigen_ind}, sigma={sigma}'
                                .format(sigma=sigma_calc, eigen_ind=eigen_ind))
                            plt.subplot(211)
                            plt.title('real part')
                            plt.plot(mu_sigma[:, 0], lambda_arr.real)
                            plt.subplot(212)
                            plt.title('imag part')
                            plt.plot(mu_sigma[:, 0], lambda_arr.imag)
                            plt.xlabel('mu')
                            plt.show()
                        warn(('DAMNED!!!! convergence failed for mu={mu_calc}, sigma={sigma_calc}, '+
                                         'eigen_ind={eigen_ind}').format(**locals()))
                        lambda_arr[i:] = -2./3.*self.params['lambda_real_abs_thresh'] # set too very low value for not disturbing
                        return lambda_arr  
            
                if mu_calc == mu or sigma_calc == sigma:
                    converged_outer = True
                    lambda_arr[i] = lambda_calc
                    
                    if converged_inner:
                        workaround_usage = 0 # reset workaround counter
                    
                    # to account for the exponential term in the reinjection condition if a refractory 
                    # period AND a lower bound is considered we have this cut off possibility that 
                    # just stops computing an eigenvalue curve (keeping the last converged one for the skipped mu,sigma pairs)
                    if 'lambda_real_abs_thresh' in self.params and abs(lambda_arr[i].real) > self.params['lambda_real_abs_thresh']:
                        print('we reached the lowest possible lambda value, further calculation is unnecessary for eigenvalue index {} at mu={}, sigma={}'.format(eigen_ind, mu_calc, sigma_calc))
                        lambda_arr[i:] = lambda_calc
                        return lambda_arr
                else:
                    warn('throwing away in-between mu sigma values and those eigenvalues as well')
                
                
                
        return lambda_arr
          

    # refines given mu_sigma (n x 2)-array such that the refined (m x 2)-array with m>=n 
    # has delta_min as maximum neighbor distance (2-norm)
    # to allow for coarse user specified mu sigma grids => refinement is required since 
    # algorithm depends on small steps in param space (eigenvalues continuously depend on params)
    def refine_mu_sigma_curve(self, mu_sigma):
        
        delta_min = self.params['diff_mu_sigma_min']
        
        
        if self.params['verboselevel'] > 1:
            print('refining mu sigma curve enforcing a minimal distance of {diff}'.
                  format(diff=delta_min))
        
        
        assert(mu_sigma.shape[1]==2) # insist of correct array shape
        
        N = mu_sigma.shape[0]
        
        diff = np.diff(mu_sigma, axis=0)
        norm_diff = np.linalg.norm(diff, axis=1)
        L = np.ceil(norm_diff / delta_min)  # no of necessary intervals between k and k+1 (min. width delta_min)
        diff_refined = diff / np.tile(L.reshape(-1, 1), (1, 2)) #np.tile(norm_diff/L, (2, 1)).T * diff
        
        if np.max(L) == 1:
            return mu_sigma
        
        # only continue if we really have to refine something
        mu_sigma_refined = []
        for k in range(N-1):
            mu_sigma_refined += [mu_sigma[k, :] + l*diff_refined[k, :] for l in range(int(L[k]))]
        mu_sigma_refined += [mu_sigma[N-1, :]]
        
        mu_sigma = np.array(mu_sigma_refined)
        
        return mu_sigma 
    
    
    
    # this function returns at least min_ev (real) eigenvalues by integrating  
    # the operator's eigenfunction equation and returning all zeros of the boundary 
    # condition at the other side of the voltage interval, i.e. the zeros correspond 
    # to eigenvalues of the fokker-planck-operator   
    # the search starts within the interval lambda_grid_init which is successively extended 
    # with intervals of the same resolution until all required eigenvalues are found (or a 
    # number of maxextensions) interval extensions have been done.
    # if addzero is True then the trivial 0 eigenvalue will be appended to the results
    # if adjoint is True then the adjoint operator will be used instead of the original one
    # if plotting is True then the result and the procedure is visualized 
    #
    # assumption: mu, sigma are chosen s.t. the regime is purely noise-dominated, i.e. the 
    # first min_ev eigenvalues are real. complex eigenvalues are ignored by this function
    def real_eigenvalues(self, lambda_grid, plotting=False, min_ev=1, 
                         adjoint=False, maxextensions=10000):
        
        params = self.params
        
        k_extension = 0
        lambda_found_all = []
        lambda_grid_all = []
        bc_grid_all = []
        init_real_all = []
        
        while len(lambda_found_all)<min_ev and k_extension<maxextensions:
        
            print('searching for real eigenvalues within lambda-interval [{left}, {right}] for mu={mu}, sigma={sig}'.format(left=lambda_grid[0], 
                                                                                                  right=lambda_grid[-1], mu=self.params['mu'], sig=self.params['sigma']))
        
            grid_lambda_points = len(lambda_grid)
            
            bc_grid = -np.ones_like(lambda_grid) 
            init_grid = np.ones_like(bc_grid)
            for i in reversed(range(grid_lambda_points)):
                if params['verboselevel']>=2:
                    print('processing lambda={l}'.format(l=lambda_grid[i]))
                if not adjoint:
                    boundary_cond_i = self.eigenflux_lb([lambda_grid[i], 0.0]) # real and imag. part of q(V_lb) for this lambda
                else:
                    boundary_cond_i = self.adjoint_gamma([lambda_grid[i], 0.0])
                bc_grid[i] = boundary_cond_i[0] # ignore the imag. part (only real eigenvalues are asserted)
                init_grid[i] = self.params['init_real_last']
        
            # (manually) interpolate the values of qlb at the discrete lambda grid
            # for finding the roots (which are the requested eigenvalues)
            # note: the zero of the linear function connecting sign-changing (x_0, g_0) and (x_1, g_1)
            # is given by ( x_0*g_1 - x_1*g_0 ) / ( g_1 - g_0 )
            ind_left_root_boolean = (bc_grid[:-1] * bc_grid[1:]) < 0
            ind_left_root = ind_left_root_boolean.nonzero()[0]
            
            lambda_found = (((lambda_grid[ind_left_root] * bc_grid[ind_left_root+1])
                             - (lambda_grid[ind_left_root+1] * bc_grid[ind_left_root])) 
                            /
                            (bc_grid[ind_left_root+1] - bc_grid[ind_left_root])) 
            
            init_real = init_grid[ind_left_root] # left from eigval is hopefully close enough for the flux
            
        
        # the following is not necessary anymore since we decided to assume 0 ev always as existent (theory)
        #     add zero eigenvalue in case it is not found through the above sign-change procedure
        #     eps = 1e-6         
        #     if abs(lambda_grid[-1])<eps: # arbitrary tolerance in case we do not choose exactly zero border
        #         if not (grid_lambda_points-2 in ind_left_root): 
        #             lambda_found = np.concatenate([lambda_found, [0]])
        
            # reverse order of eigenvalues and append them
            lambda_found_all = np.concatenate([lambda_found_all, lambda_found[::-1]])

            
            lambda_grid_all = np.concatenate([lambda_grid, lambda_grid_all])
            bc_grid_all = np.concatenate([bc_grid, bc_grid_all])
            init_real_all = np.concatenate([init_real_all, init_real[::-1]])
            
            # TODO
            # adapt eigenvalue grid depending on smallest distance of the last l_refinement eigenvalues (if they exist)
            # but maybe this is not useful since the eigenvalues in general do not appear in the pairs
            # due to the purely real diffusive ones for the case of lower bound < reset...
            
            # compute next interval with same no of points and spacing as last one
            left_lambda = lambda_grid[0]-(lambda_grid[-1]-lambda_grid[0])
            right_lambda = lambda_grid[0]-np.spacing(lambda_grid[0])
            gridpoints_lambda = len(lambda_grid)
            lambda_grid = np.linspace(left_lambda, 
                                          right_lambda, 
                                          num=gridpoints_lambda,
                                          endpoint=True)
            k_extension += 1
    
    
        print('found {n} (real) eigenvalues for mu={mu}, sigma={sigma}:'.format(n=len(lambda_found_all),
                                                                mu=params['mu'], sigma=params['sigma']))
        # add to verbos
        print(lambda_found_all)
        print('with real init fluxes:')
        print(init_real_all)
    
        if k_extension == maxextensions:
            print('GIVING UP. could not find {m} eigenvalues after {k} extensions.'.format(m=min_ev, k=k_extension))
            exit()
    
    
        if plotting:
    
            plt.figure()
            plt.title('mu={mu}, sigma={sigma} (model: {model}; adjoin: {adj})'
                     .format(mu=params['mu'], sigma=params['sigma'], model=params['model'], adj=adjoint))
            plt.plot(lambda_grid_all, bc_grid_all)
            plt.xlabel('$\lambda$')
            if not adjoint:
                plt.ylabel('$q(V_{lb})$')
            else:
                plt.ylabel('$\gamma(V_s,V_r)$')
    #         plt.ylim([-1,1])
            plt.plot(lambda_found_all, np.zeros_like(lambda_found_all), 'ro')
            
            plt.show()
        
        # add initial fluxes
        
        
        return lambda_found_all, init_real_all, bc_grid_all
    
    
    
    def eigenvalue(self, lambda_init, adjoint=False, real=False, root_extra_options={}):
        
        lambda_init_arr = np.array([lambda_init.real, lambda_init.imag])
    
        if not adjoint: 
            func_2d = self.eigenflux_lb            
        else:
            func_2d = self.adjoint_gamma
        
        root_method = self.params['root_method']
        root_options = self.params['root_options'].copy()
        root_options.update(root_extra_options)
        
        if real:
            func_1d = lambda x: func_2d(np.array([x, 0.0]))[0]
            lambda_init_arr = lambda_init_arr[0]
            func = func_1d
        else:
            func = func_2d        
        
            result = root(fun=func, 
                            x0=lambda_init_arr,
                            method=root_method,
                            options=root_options)                
       
        # checking abs tol criterion only if the abs real part of lambda is small and the real part of  phi_lb is large 
#         if abs(result.x[0]) < self.params['solver_abstol_threshold_lambdareal'] and abs(self.phi_q_lb[0]) > self.params['solver_abstol_threshold_philb']:
#             print('skipping absolute tolerance criterion checking since phi,q={pq}'.format(pq=self.phi_q_lb))
#         elif 
        if np.linalg.norm(result.fun) > self.params['solver_abstol']: 
            raise NotConverged(result, abstol_failed=True)
    
        elif not result.success:
            raise NotConverged(result)
                
        
        if self.params['verboselevel']>=1:
            print('successfully converged with solver details:')
            print(result)
        # successfull convergence if we arrive here
        lambda_2d = result.x
        
        return complex(*lambda_2d)
    
    # computes the eigenfunction for given (assumably correct) lambda_
    # lambda_ can be either a complex number or a 2d vector for real and imag. part
    # note: usually this function is called after the true eigenvalue has been computed 
    # by the function eigenvalue
    # NOTE THAT in the case of a refractory period the steady state density does NOT include 
    # only the non-refractory part (similar for the rate)
    def eigenfunction(self, lambda_, mu, sigma, adjoint=False):
        if isinstance(lambda_, complex): lambda_2d = [lambda_.real, lambda_.imag]
        elif isinstance(lambda_, (float,int)): lambda_2d = [lambda_, 0.] 
        else: lambda_2d = lambda_

        self.params['mu'] = mu
        self.params['sigma'] = sigma
        
        if not adjoint:
            condition, Vgrid, phi, q = self.eigenflux_lb(lambda_2d, eigFunc=True)
            eigfunc = phi
            q_or_dpsi = q
        else:
            condition, Vgrid, psi, dpsi = self.adjoint_gamma(lambda_2d, eigFunc=True)
            eigfunc = psi
            q_or_dpsi = dpsi
        
        # the following assertion would ensure that only valid eigenfunctions are returned
        # but it required the respective init fluxes which are so-far not available and thus 
        # very large values are assumed
#        assert np.linalg.norm(np.array(condition)) < self.params['solver_abstol']
        if self.params['verboselevel'] > 0:
            print('eigenfunction: adjoint={}, mu={}, sigma={}, norm(condition) = {} (abstol={}) for lambda={}'.format(adjoint, mu, sigma, np.linalg.norm(np.array(condition)), self.params['solver_abstol'], lambda_))
            
        return Vgrid, eigfunc, q_or_dpsi
    
    
    # helper method to allow grid creation and extraction of model params for numba time integration
    def params_tuple(self):
        params = self.params

        # input         
        mu = float(params['mu'])
        sigma = float(params['sigma'])
        
        # grid (non-uniform spacing would also be allowed)
        Vgrid = np.linspace(float(params['V_lb']), float(params['V_s']), 
                            params['grid_V_points'])            
        
        # neuronal params      
        k_r = np.argmin(np.abs(Vgrid - params['V_r']))
        tau_ref = params['tau_ref']
        model = params['model']
        if model == 'PIF':
            hasLeak = False
            hasExp = False
        elif model == 'LIF':
            hasLeak = True
            hasExp = False
        elif model == 'EIF':
            hasLeak = True
            hasExp = True
        else: # unknown model
            raise Exception('Model {0} not supported'.format(model))
        
        if hasLeak:
            C = params['C']
            g_L = params['g_L']
            E_L = params['E_L']
        else:
            C = g_L = E_L = 0.0
            
        if hasExp:
            delta_T = params['delta_T']
            V_T = params['V_T']
        else:
            delta_T = V_T = 0.0
            
        return mu, sigma, Vgrid, k_r, hasLeak, C, g_L, E_L, hasExp, delta_T, V_T, tau_ref
    

    # NOTE: this midpoint rule leads to order 2 of the magnus expansion; orders 4 and 6 are also easy        
    # former method eigenflux_lb_magnus
    def eigenflux_lb(self, lambda_2d, eigFunc=False):
        
        lambda_complex = complex(*lambda_2d)

        mu, sigma, Vgrid, k_r, hasLeak, C, g_L, E_L, hasExp, delta_T, V_T, tau_ref = self.params_tuple() 

        phi = np.zeros(len(Vgrid), dtype=np.complex128)
        q = np.zeros_like(phi)
        
        # initialization as in Richardson 2007 and Ostojic 2011 
        # (complex version inspired by Srdjan Ostojic as in Schaffer 13)
        # as the eigenequation is linear, we can (together with the absorbing condition) scale
        # the flux at the spike boundary with an arbitrary complex constant
        # e.g. we can choose q(V_s)=init_q it to be (any) real nonzero number
        if 'init_real' in self.params:
            # use last successfull and normalized flux init to allow fast ODE integration (huge eigenfuncs if q_init_r=1)
            init_q_r = float(self.params['init_real'])
        else:
            init_q_r = 1.0
            
        
        # for PIF neurons the exponential is applied to a matrix which does not depend on V
        # for an uniform grid (second condition) we can then move the matrix exp outside the V loop
        # of course this is implemented only for speedup reasons / result corresponds to the general case
        uniform_grid = np.abs(np.diff(Vgrid, 2)).max() < 1e-15        
        if self.params['model'] != 'PIF' or not uniform_grid:
            eigeneq_backwards_magnus_general(init_q_r, phi, q, Vgrid, k_r, lambda_complex, 
                        mu, sigma, hasLeak, C, g_L, E_L, hasExp, delta_T, V_T, tau_ref)
            
        else:  
            # construction of the matrix A for the PIF
            sig2h = sigma**2/2.0
            A = np.zeros((2, 2), dtype=np.complex128)            
            
            A[0, 1] = lambda_complex
            A[1, 0] = 1/sig2h  
            fV = 0.0
            A[1, 1] = -(fV + mu)/sig2h
            dV = Vgrid[1]-Vgrid[0]            
         
            exp_A_dV = np.zeros((2, 2), dtype=np.complex128)
            exp_mat2_full_scalar(A, dV, exp_A_dV) # exp_A_dV contains now exp(A*dV)
            # (numba-njit-compatible) equivalent to exp_A_dV = scipy.linalg.expm(A*dV)
            
            eigeneq_backwards_magnus_pif(init_q_r, phi, q, Vgrid, k_r, lambda_complex, tau_ref, exp_A_dV)


            
        # return normalized eigenfunc and calculate init flux
        phi_norm = normalize(phi, Vgrid)
        q /= phi_norm
        self.params['init_real_last'] = init_q_r/phi_norm
        
        q_lb_2d = [q[0].real, q[0].imag]
        
        if not eigFunc:
            # return the real and imag. part of the flux q evaluated at V_lb
            return q_lb_2d
        else:
            return (q_lb_2d, Vgrid, phi, q)
    
    
    # former method adjoint_gamma_magnus
    def adjoint_gamma(self, lambda_2d, eigFunc=False):
        
        lambda_complex = complex(*lambda_2d)

        mu, sigma, Vgrid, k_r, hasLeak, C, g_L, E_L, hasExp, delta_T, V_T, tau_ref = self.params_tuple() 

        Nv = len(Vgrid)
        psi = np.zeros(Nv, dtype=np.complex128)
        dpsi = np.zeros_like(psi)

        # initialization (currently not working...)
        if 'init_real' in self.params:
            # use last successfull and normalized flux init to allow fast ODE integration (huge eigenfuncs if q_init_r=1)
            init_psi_r = float(self.params['init_real'])
        else:
            init_psi_r = 1.0

         # for PIF neurons the exponential is applied to a matrix which does not depend on V
        # for an uniform grid (second condition) we can then move the matrix exp outside the V loop
        # of course this is implemented only for speedup reasons / result corresponds to the general case
        uniform_grid = np.abs(np.diff(Vgrid, 2)).max() < 1e-15 
    
        if self.params['model'] != 'PIF' or not uniform_grid:
            adjoint_eigeneq_forwards_magnus_general(init_psi_r, psi, dpsi, Vgrid, lambda_complex, 
                                         mu, sigma, hasLeak, C, g_L, E_L, hasExp, delta_T, V_T)

        else:
            # construction of the matrix A for the PIF
            sig2h = sigma**2/2.0
            A = np.zeros((2, 2), dtype=np.complex128)
            
            A[0,1] = 1.0
            A[1,0] = lambda_complex/sig2h  
            fV = 0.0
            A[1, 1] = -(fV + mu)/sig2h
            
            dV = Vgrid[1]-Vgrid[0]
            exp_A_dV = np.zeros_like(A)
            
            exp_mat2_full_scalar(A, dV, exp_A_dV) # exp_A_dV contains now exp(A*dV)
            # (numba-njit-compatible) equivalent to exp_A_dV = scipy.linalg.expm(A*dV)
            adjoint_eigeneq_forwards_magnus_pif(init_psi_r, psi, dpsi, Vgrid, lambda_complex, 
                                 mu, sigma, hasLeak, C, g_L, E_L, hasExp, delta_T, V_T, exp_A_dV)
            

        # TODO: INCLUDE NORMALIZATION INTO ADJOINT.
        # problems: no convergence to eigenvalues in eigenvalue_curve with normalization
        #psi_norm = normalize(psi, Vgrid) # only psi not dpsi is used in the following
        psi_norm = 1.0
        self.params['init_real_last'] = init_psi_r/psi_norm
        
        k_r = np.argmin(np.abs(Vgrid - self.params['V_r']))        
        gamma_complex = psi[Nv-1] - psi[k_r] * cmath.exp( -lambda_complex*tau_ref)
        
        gamma_2d = [gamma_complex.real, gamma_complex.imag]
        
        if not eigFunc:
            # return the real and imag. part of the flux q evaluated at V_lb
            return gamma_2d
        else:
            return (gamma_2d, Vgrid, psi, dpsi)

# normalizeation based on L1 norm of f 
# i.e. integrate abs of f over the grid 
# (corresponds to integration for nonnegative functions such as the stationary distribution)
def normalize(f, grid):
    f_norm = np.sum(np.abs(f[:-1]) * np.diff(grid))
    f /= f_norm
    return f_norm


def inner_prod(f1, f2, grid):
    prod = f1 * f2
    return np.sum( 0.5*(prod[:-1] + prod[1:]) * np.diff(grid) )     


# redundant code with method in specsolv class but need it for numba
@njit
def intfire_rhs(V, hasLeak, C, g_L, E_L, hasExp, delta_T, V_T): 
    rhs = 0.0 

    if hasLeak:
        rhs += g_L/C * (E_L-V)
    
    if hasExp:
        rhs += g_L/C * delta_T * math.exp((V-V_T)/delta_T)
    
    return rhs # = (I_L + I_exp)/C => rhs of dV/dt excluding mu and sigma

@njit
def eigeneq_backwards_magnus_pif(init_q_r, phi, q, Vgrid, k_r, lambda_complex, tau_ref, exp_A_dV):        
    # initialization as in Richardson 2007 and Ostojic 2011 
    # (complex version inspired by Srdjan Ostojic as in Schaffer 13)
    # as the eigenequation is linear, we can (together with the absorbing condition) scale
    # the flux at the spike boundary with an arbitrary complex constant
    # e.g. we can choose q(V_s)=init_q it to be (any) complex nonzero number
    
    Nv = len(Vgrid)        
    phi[Nv-1] = 0.0 + 0.0j # absorbing boundary at the spike voltage
#    q[Nv-1] = 1.0 + 0.0j # linear eq = arbitrary complex scaling
    q[Nv-1] = init_q_r + 0.0j # linear eq = arbitrary complex scaling
    
    for k in range(Nv-2, -1, -1): # k= Nv-2 , ... , 0
    
        q[k] = exp_A_dV[0,0]*q[k+1] + exp_A_dV[0,1]*phi[k+1]
        phi[k] = exp_A_dV[1,0]*q[k+1] + exp_A_dV[1,1]*phi[k+1]
        
        # reinjection condition
        if k == k_r:
            q[k] = q[k] - q[Nv-1]*cmath.exp(-lambda_complex*tau_ref)

# fully exponential integrator based on truncated magnus series
@njit
def eigeneq_backwards_magnus_general(init_q_r, phi, q, Vgrid, k_r, lambda_complex, mu, sigma, 
                                  hasLeak, C, g_L, E_L, hasExp, delta_T, V_T, tau_ref):        
    # initialization as in Richardson 2007 and Ostojic 2011 
    # (complex version inspired by Srdjan Ostojic as in Schaffer 13)
    # as the eigenequation is linear, we can (together with the absorbing condition) scale
    # the flux at the spike boundary with an arbitrary complex constant
    # e.g. we can choose q(V_s)=init_q it to be (any) complex nonzero number   
    Nv = len(Vgrid)        
    phi[Nv-1] = 0.0 + 0.0j # absorbing boundary at the spike voltage
#    q[Nv-1] = 1.0 + 0.0j # linear eq = arbitrary complex scaling
    q[Nv-1] = init_q_r + 0.0j # linear eq = arbitrary complex scaling

    # abbreviation
    sig2h = sigma**2/2.0   
    
    A = np.zeros((2, 2), dtype=np.complex128)
    A[0, 1] = lambda_complex
    A[1, 0] = 1/sig2h  
    
    exp_A_dV = np.zeros_like(A)
    
    for k in range(Nv-2, -1, -1): # k= Nv-2 , ... , 0
    
        # grid spacing, allowing non-uniform grids
        dV = Vgrid[k+1]-Vgrid[k]
        
        fV = intfire_rhs((Vgrid[k+1]+Vgrid[k])/2.0, hasLeak, C, g_L, E_L, hasExp, delta_T, V_T)
        A[1, 1] = -(fV + mu)/sig2h
        
        exp_mat2_full_scalar(A, dV, exp_A_dV) # exp_A_dV contains now exp(A*dV)
        # (numba-njit-compatible) equivalent to exp_A_dV = scipy.linalg.expm(A*dV)
        
        # the following computes the matrix vector product exp(A*dV) * [q, phi]^T
        q[k] = exp_A_dV[0,0]*q[k+1] + exp_A_dV[0,1]*phi[k+1]
        phi[k] = exp_A_dV[1,0]*q[k+1] + exp_A_dV[1,1]*phi[k+1]
        
        # reinjection condition
        if k == k_r:
            q[k] = q[k] - q[Nv-1]*cmath.exp(-lambda_complex*tau_ref) 
   
@njit
def adjoint_eigeneq_forwards_magnus_general(init_psi_r, psi, dpsi, Vgrid, lambda_complex, 
                                         mu, sigma, hasLeak, C, g_L, E_L, 
                                         hasExp, delta_T, V_T):
    # initialization
    psi[0] = init_psi_r + 0.0j # linear eq = arbitrary complex scaling
    dpsi[0] = 0.0 + 0.0j # left boundary condition of adjoint operator
    
    # abbreviation
    sig2h = sigma**2/2.0       

    
    A = np.zeros((2, 2), dtype=np.complex128)     
    A[0,1] = 1.0
    A[1,0] = lambda_complex/sig2h    
    
    exp_A_dV = np.zeros_like(A)
    
    # exponential magnus integration
    for k in range(1, len(Vgrid)):
        
        # grid spacing, allowing non-uniform grids
        dV = Vgrid[k]-Vgrid[k-1]
        
        fV = intfire_rhs((Vgrid[k]+Vgrid[k-1])/2.0, hasLeak, C, g_L, E_L, hasExp, delta_T, V_T)
        A[1, 1] = -(fV + mu)/sig2h
        
        exp_mat2_full_scalar(A, dV, exp_A_dV) # exp_A_dV contains now exp(A*dV)
        # (numba-njit-compatible) equivalent to exp_A_dV = scipy.linalg.expm(A*dV)

        # the following computes the matrix vector product exp(A*dV) * [psi, dpsi]^T
        psi[k] = exp_A_dV[0,0]*psi[k-1] + exp_A_dV[0,1]*dpsi[k-1]
        dpsi[k] = exp_A_dV[1,0]*psi[k-1] + exp_A_dV[1,1]*dpsi[k-1]
        
        # that's it. no discontinuous reinjection => instead in characteristic eq.

@njit
def adjoint_eigeneq_forwards_magnus_pif(init_psi_r, psi, dpsi, Vgrid, lambda_complex, 
                                         mu, sigma, hasLeak, C, g_L, E_L, 
                                         hasExp, delta_T, V_T, exp_A_dV):
    # initialization
    psi[0] = init_psi_r + 0.0j # linear eq = arbitrary complex scaling
    dpsi[0] = 0.0 + 0.0j # left boundary condition of adjoint operator
    
    # exponential magnus integration
    for k in range(1, len(Vgrid)):

        # the following computes the matrix vector product exp(A*dV) * [psi, dpsi]^T
        psi[k] = exp_A_dV[0,0]*psi[k-1] + exp_A_dV[0,1]*dpsi[k-1]
        dpsi[k] = exp_A_dV[1,0]*psi[k-1] + exp_A_dV[1,1]*dpsi[k-1]
        
        # that's it. no discontinuous reinjection => instead in characteristic eq.



# this numba function computes the complex matrix exponential exp(X*t) where t is a real scalar
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



class NotConverged(Exception):
    def __init__(self, solver_result, abstol_failed=False):
        self.solver_result = solver_result
        self.abstol_failed = abstol_failed
        message = ''        
        if self.abstol_failed:
            message +=' >>> absolute tolerance criterion not satisfied'
        if not self.solver_result.success:
            message += ' >>> solver root() did not converge'
        
        if 'nfev' in self.solver_result:
            nfev = self.solver_result.nfev
        else:
            nfev = '...unknown for this solver...'
        self.message = ('<NotConverged: solver failed after {nfev} '+
                        'function evaluations with msg: \n{msg}>').format(
                                nfev=nfev, 
                                msg=message)
    
    def __str__(self):
        return self.message
        
    
# some helper functions
def warn(*s):
    print('warning: {}'.format(s))
    
def error(*s):
    print('error: {}'.format(s))
    exit()
    
    
    
# SPECTRALSOLVER CODE ENDS HERE

# FOLLOWING LINES: POSTPROCESSING AND PLOTTING  
    

# due to (mu,sigma) points that are close to double zeros of lambda_{1/2} there are numerical 
# ambiguities that lead to artefacts in terms of spikes in the quantities which are removed here:
# interpolate all quantities in quant_names based on 
# 1.) unexpected (i.e. wrong) large imaginary part of f.c_mu = f_1 * c_mu_1 + f_2 * c_mu_2, and
# 2.) unexpected peaks (i.e. points with discontinuous derivative)
# (just take the left neighbored 
# value (mu[i-1]) which is assumed to be correct)
# note that the min/max sigma values should be specified in order not to manipulate the 
# (correct) discontinuities that occur due to switching between diffusive and regular modes 
# for larger mu and sigma
# usage:
# ALL quantities should be given but hardly asserted is only the existence of f_* and c_mu_* 
# that are used to detect numerical artifacts
def quantities_postprocess(quantities, quant_names, 
                           minsigma_interp, maxsigma_interp, maxmu_interp, 
                           tolerance_conjugation=1e-5):
    
    mu = quantities['mu']
    sigma = quantities['sigma']    
    N_mu = mu.shape[0]
    N_sigma = sigma.shape[0]   
    
    assert all((q in quant_names for q in ['f_1', 'f_2', 'c_mu_1', 'c_mu_2']))
    
    f_1 = quantities['f_1']
    c_mu_1 = quantities['c_mu_1']
    f_2 = quantities['f_2']
    c_mu_2 = quantities['c_mu_2']
    f_cmu  = f_1*c_mu_1 + f_2*c_mu_2    
    
    for j in range(N_sigma):
    
        # compute the mu index i where lambda_1 is complex but for all indicies < i is real
        i_complex = np.where(np.abs(f_cmu[:, j].imag) > tolerance_conjugation)[0]
        for i in i_complex:
            print('(potential double eigenvalue): fixing quantities (wrong imag part for sigma[{}]={} at mu[{}]={}'
                .format(j, sigma[j], i, mu[i])) 
                
            for q in quant_names:
                    quant = quantities[q]
                    q_left = quant[i-1, j] 
#                    q_right = quant[i+1, j]                                                      
                    quant[i, j] = q_left
#                    quant[i, j] = (q_left+q_right)/2.
                    
        # interpolate spikes for sigma up to maxsigma_interp
        if minsigma_interp <= sigma[j] and sigma[j] <= maxsigma_interp:
            re = f_cmu[:, j].real
            for i in range(1, N_mu-1):
                if (re[i]-re[i-1])*(re[i+1]-re[i]) < 0 and mu[i] < maxmu_interp:
                    print('(potential double eigenvalue): fixing quantities (spike in real part for sigma[{}]={} at mu[{}]={}'
                        .format(j, sigma[j], i, mu[i])) 
                        
                    for q in quant_names:
                        quant = quantities[q]
                        q_left = quant[i-1, j]
                        quant[i, j] = q_left

    
    
# OPTIONAL/TODO:
# put biorthonormality check also in specsolv?
#if check_biorthonormality:
#    norm_psi = params['norm_psi']
#    skip_sig = 1
#    skip_mu = 100
#    
#    specsolv = SpectralSolver(params.copy())    
#    
#    print('checking biorthonormality based on {} norming, skipping {} sigmas'.
#        format('psi' if norm_psi else 'phi', skip_sig))
#    time.sleep(3)
#            
#    for i in range(0, N_mu, skip_mu):    
#        for j in range(0, N_sigma, skip_sig):
#            
#            V_vec, phi_1 = specsolv.eigenfunction(lambda_1[i, j], mu[i], sigma[j])
#            V_vec, phi_2 = specsolv.eigenfunction(lambda_2[i, j], mu[i], sigma[j])
#            
#            V_vec, psi_1 = specsolv.eigenfunction(lambda_1[i, j], mu[i], sigma[j], adjoint=True)
#            V_vec, psi_2 = specsolv.eigenfunction(lambda_2[i, j], mu[i], sigma[j], adjoint=True)
#            
#            # normalize psi_j such that: <psi_j | phi_j> = integral of psi_j.H * phi_j = 1 
#            # this leads to biorthonormalization, i.e. <psi_j | phi_i> = delta_ij 
#            norm_prod1 = inner_prod(psi_1, phi_1, V_vec)            
#            norm_prod2 = inner_prod(psi_2, phi_2, V_vec)
#            if norm_psi:
#                psi_1 = psi_1 / norm_prod1 #.conjugate() # conjugation due to fun1.conjugate()
#                psi_2 = psi_2 / norm_prod2  #.conjugate()
#            else:
#                phi_1 = phi_1 / norm_prod1 # no conjugation here
#                phi_2 = phi_2 / norm_prod2                
#                
#            print('normalization for mu={}, sigma={}, lambda={}, n1={}, n2={}'
#                .format(mu[i], sigma[j], lambda_1[i, j], norm_prod1, norm_prod2))
#            print('<psi_1|phi_1> = {}'.format(inner_prod(psi_1, phi_1, V_vec)))
#            print('<psi_2|phi_2> = {}'.format(inner_prod(psi_2, phi_2, V_vec)))
#            print('<psi_1|phi_2> = {}'.format(inner_prod(psi_1, phi_2, V_vec)))
#            print('<psi_2|phi_1> = {}'.format(inner_prod(psi_2, phi_1, V_vec)))
#    


# plotting stuff (was in own file plotting.py before)

def get_sigma_inds_validation(inds_sigma, sigma, sigma_validation):
    inds_sigma_validation = []
    for j in inds_sigma:
        sigma_j_validation = np.argmin(np.abs(sigma[j] - sigma_validation))
        inds_sigma_validation.append(sigma_j_validation)
    return inds_sigma_validation

def plot_quantities_eigvals(quantities, inds_sigma_plot, colormap_sigma='winter',
                            plot_validation=False, quantities_validation={}, 
                            marker_validation='o', ms_validation=2, 
                            color_validation='black', linestyle_validation='None'):

    mu = quantities['mu']
    sigma = quantities['sigma']
    lambda_1 = quantities['lambda_1']
    lambda_2 = quantities['lambda_2']

    N_sigma = sigma.shape[0]

   # final eigenvalue quantities
    plt.figure()
    plt.suptitle('final eigenvalues $\lambda_1, \lambda_2$')
    
    lambda_1_2 = np.concatenate([lambda_1, lambda_2])
    ylim_real = [np.amin(lambda_1_2.real), 0]
    
    if plot_validation:
        inds_sigma_validation = get_sigma_inds_validation(inds_sigma_plot, 
                                                          sigma, 
                                                          quantities_validation['sigma'])
                                    
    
    ax_real = plt.subplot(3, 2, 1)
    ax_imag = plt.subplot(3, 2, 2)
    for k_j, j in enumerate(inds_sigma_plot):
            
        # color
        cm = plt.get_cmap(colormap_sigma) 
        cNorm  = colors.Normalize(vmin=sigma[0], vmax=sigma[-1])
        scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cm)
        linecolor = scalarMap.to_rgba(sigma[j])
        # for poster cns prague 2015
#            rgb = [0, float(k_j)/(len(inds_sigma_plot)-1), 0]
#            linecolor = rgb
        
        if plot_validation:
            j_validation = inds_sigma_validation[k_j]
        
        # lambda_1 (real part)
        plt.subplot(3, 2, 1)
        # labels
        if j in [0, N_sigma//2, N_sigma-1]:
            siglabel = '$\sigma={0:.3}$ [mV/$\sqrt{{\mathrm{{ms}}}}$]'.format(sigma[j])
            if plot_validation:
                siglabel += ', $\sigma_\mathrm{{val}}=$ {0:.3}'.format(quantities_validation['sigma'][j_validation])
        else:
            siglabel = None

        if plot_validation and 'lambda_1' in quantities_validation:
            plt.plot(quantities_validation['mu'], 
                     quantities_validation['lambda_1'][:, j_validation].real,
                     color=color_validation, marker=marker_validation, markersize=ms_validation, 
                     linestyle=linestyle_validation)


        plt.plot(mu, lambda_1[:, j].real, label=siglabel, color=linecolor)

        if j==0:
            plt.title('real part')
            plt.ylim(ylim_real)
            plt.ylabel('$\Re(\lambda_1)$ [kHz]')
        if j==N_sigma-1:
            plt.legend(loc='best')
        
        
        # lambda_1 (imag part)
        plt.subplot(3, 2, 2, sharex=ax_real)

        if plot_validation and 'lambda_1' in quantities_validation:
            plt.plot(quantities_validation['mu'], 
                     quantities_validation['lambda_1'][:, j_validation].imag,
                     color=color_validation, marker=marker_validation, markersize=ms_validation,
                     linestyle=linestyle_validation)

        plt.plot(mu, lambda_1[:, j].imag, color=linecolor)
        
        if j==0:
            plt.title('imag. part')
            plt.ylim(np.amin(lambda_1.imag), np.amax(lambda_1.imag))
            plt.ylabel('$\Im(\lambda_1)$ [kHz]')
    
        
        # lambda_2 (real part)
        plt.subplot(3, 2, 3, sharex=ax_real, sharey=ax_real)
        
        if plot_validation and 'lambda_2' in quantities_validation:
            plt.plot(quantities_validation['mu'], 
                     quantities_validation['lambda_2'][:, j_validation].real,
                     color=color_validation, marker=marker_validation, markersize=ms_validation, 
                     linestyle=linestyle_validation)            
        
        plt.plot(mu, lambda_2[:, j].real, color=linecolor)
        
        if j==0:
            plt.ylim(ylim_real)
            plt.ylabel('$\Re(\lambda_2)$ [kHz]')
        
        plt.xlabel('$\mu$ [mV/ms]')
        
        
        # lambda_2 (imag part)
        plt.subplot(3, 2, 4, sharex=ax_real, sharey=ax_imag)
        
        if plot_validation and 'lambda_2' in quantities_validation:
            plt.plot(quantities_validation['mu'], 
                     quantities_validation['lambda_2'][:, j_validation].imag,
                     color=color_validation, marker=marker_validation, markersize=ms_validation, 
                     linestyle=linestyle_validation)            
        
        plt.plot(mu, lambda_2[:, j].imag, color=linecolor)
        
        if j==0:
            plt.ylim(np.amin(lambda_2.imag), np.amax(lambda_2.imag))
            plt.ylabel('$\Im(\lambda_2)$ [kHz]')
            
        plt.xlabel('$\mu$ [mV/ms]')


def plot_quantities_real(quantities, inds_sigma_plot, colormap_sigma='winter',
                         plot_validation=False, quantities_validation={}, 
                         marker_validation='o', ms_validation=2, 
                         color_validation='black', linestyle_validation='None'):
    
    mu = quantities['mu']
    sigma = quantities['sigma']
    r_inf = quantities['r_inf']
    dr_inf_dmu = quantities['dr_inf_dmu']
    dr_inf_dsigma = quantities['dr_inf_dsigma']
    V_mean_inf = quantities['V_mean_inf']
    dV_mean_inf_dmu = quantities['dV_mean_inf_dmu']
    dV_mean_inf_dsigma = quantities['dV_mean_inf_dsigma']
    

    N_sigma = sigma.shape[0]    
    
    plt.figure() 
    
    plt.suptitle('real quantities')
    
    if plot_validation:
        inds_sigma_validation = get_sigma_inds_validation(inds_sigma_plot, 
                                                      sigma, 
                                                      quantities_validation['sigma'])

    for k_j, j in enumerate(inds_sigma_plot):
        # color
        cm = plt.get_cmap(colormap_sigma) 
        cNorm  = colors.Normalize(vmin=sigma[0], vmax=sigma[-1])
        scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cm)
        linecolor = scalarMap.to_rgba(sigma[j])
        
        if plot_validation:
            j_validation = inds_sigma_validation[k_j]        
        
        # r_inf
        ax1 = plt.subplot(2, 3, 1)
    
        # labels
        if j in [0, N_sigma//2, N_sigma-1]:
            siglabel = '$\sigma={0:.3}$ [mV/$\sqrt{{\mathrm{{ms}}}}$]'.format(sigma[j])
            if plot_validation:
                siglabel += ', $\sigma_\mathrm{{val}}=$ {0:.3}'.format(quantities_validation['sigma'][j_validation])
            
        else:
            siglabel = None

        if plot_validation and 'r_inf' in quantities_validation:

            plt.plot(quantities_validation['mu'], 
                     quantities_validation['r_inf'][:, j_validation],
                     color=color_validation, marker=marker_validation, markersize=ms_validation,
                     linestyle=linestyle_validation)

        plt.plot(mu, r_inf[:, j], label=siglabel, color=linecolor)

        if j==0:
#                plt.ylim(0, 30)
            plt.ylabel('$r_\infty$ [kHz]')
        if j==N_sigma-1:
            plt.legend(loc='best')
        
        # dr_inf_dmu
        plt.subplot(2, 3, 2, sharex=ax1)

        if plot_validation and 'dr_inf_dmu' in quantities_validation:
            plt.plot(quantities_validation['mu'], 
                     quantities_validation['dr_inf_dmu'][:, j_validation],
                     color=color_validation, marker=marker_validation, markersize=ms_validation, 
                     linestyle=linestyle_validation)
        
        
        plt.plot(mu, dr_inf_dmu[:, j], color=linecolor)
        if j==0:
#                 plt.ylim(0, 200)
            plt.ylabel('$\partial_\mu r_\infty$ [1/mV)]')
        
        # dr_inf_dsigma
        plt.subplot(2, 3, 3, sharex=ax1)

        if plot_validation and 'dr_inf_dsigma' in quantities_validation:
            plt.plot(quantities_validation['mu'], 
                     quantities_validation['dr_inf_dsigma'][:, j_validation],
                     color=color_validation, marker=marker_validation, markersize=ms_validation, 
                     linestyle=linestyle_validation)

        plt.plot(mu, dr_inf_dsigma[:, j], color=linecolor)
        if j==0:
#                 plt.ylim(-90, -50)
            plt.ylabel('$\partial_\sigma r_\infty$ [$1/(\mathrm{mV} \sqrt{{\mathrm{{ms}}}})$]')
        
        # V_mean_inf
        plt.subplot(2, 3, 4, sharex=ax1)
        
        if plot_validation and 'V_mean_inf' in quantities_validation:
            plt.plot(quantities_validation['mu'], 
                     quantities_validation['V_mean_inf'][:, j_validation],
                     color=color_validation, marker=marker_validation, markersize=ms_validation, 
                     linestyle=linestyle_validation)
        
        plt.plot(mu, V_mean_inf[:, j], color=linecolor)
        if j==0:
#                plt.ylim(0, 0.5)
            plt.ylabel('$\langle V \\rangle_\infty$ [mV]')
            plt.xlabel('$\mu$ [mV/s]')
        
        # dV_mean_inf_dmu
        plt.subplot(2, 3, 5, sharex=ax1)
        
        if plot_validation and 'dV_mean_inf_dmu' in quantities_validation:
            plt.plot(quantities_validation['mu'], 
                     quantities_validation['dV_mean_inf_dmu'][:, j_validation],
                     color=color_validation, marker=marker_validation, markersize=ms_validation, 
                     linestyle=linestyle_validation)
        
        plt.plot(mu, dV_mean_inf_dmu[:, j], color=linecolor)
        if j==0:
            plt.ylabel('$\partial_\mu \langle V \\rangle_\infty$ [ms]')
            plt.xlabel('$\mu$ [mV/ms]')
        
        # dV_mean_inf_dmu
        plt.subplot(2, 3, 6, sharex=ax1)        

        if plot_validation and 'dV_mean_inf_dsigma' in quantities_validation:
            plt.plot(quantities_validation['mu'], 
                     quantities_validation['dV_mean_inf_dsigma'][:, j_validation],
                     color=color_validation, marker=marker_validation, markersize=ms_validation, 
                     linestyle=linestyle_validation)
                
        plt.plot(mu, dV_mean_inf_dsigma[:, j], color=linecolor)
        if j==0:
            plt.ylabel(r'$\partial_\sigma \langle V \rangle_\infty [\sqrt{\mathrm{ms}}]$')
            plt.xlabel('$\mu$ [mV/s]')
        
def plot_quantities_complex(complex_quants_plot, quantities, inds_sigma_plot, 
                            colormap_sigma='winter',
                            plot_validation=False, quantities_validation={}, 
                            marker_validation='o', ms_validation=2, 
                            color_validation='black', linestyle_validation='None'):
    
    # generate summed and/or multiplied quantities    
    for q in complex_quants_plot:        
        q_mult_index = q.find('*')            
        if q_mult_index >= 0:
            q_left = q[:q_mult_index]
            q_right = q[q_mult_index+1:]
            quantities[q] = quantities[q_left] * quantities[q_right]
            # validation (currently manually computed in script)
            if plot_validation and q_left in quantities_validation and q_right in quantities_validation:
                quantities_validation[q] = quantities_validation[q_left] * quantities_validation[q_right]
        
    

    sigma = quantities['sigma']
    N_sigma = sigma.shape[0]
    
    
    plt.figure()
    
    plt.suptitle('complex quantities')    
    
    spr = misc.utils.SubplotRect(2, len(complex_quants_plot))    
    
    if plot_validation:
        inds_sigma_validation = get_sigma_inds_validation(inds_sigma_plot, 
                                                  sigma, 
                                                  quantities_validation['sigma'])

    for k_j, j in enumerate(inds_sigma_plot):
        
        
        # color
        cm = plt.get_cmap(colormap_sigma) 
        cNorm  = colors.Normalize(vmin=sigma[0], vmax=sigma[-1])
        scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cm)
        linecolor = scalarMap.to_rgba(sigma[j])
        
        if plot_validation:
            j_validation = inds_sigma_validation[k_j]   
        
        spr.first()
        ax1 = spr.current_axes()
        firstquant = True
        
        for q in complex_quants_plot:        

            if firstquant and j == 0:
                plt.ylabel('real part')
            
            if firstquant and j in  [0, N_sigma//2, N_sigma-1]:
                siglabel = '$\sigma={0:.3}$ [mV/$\sqrt{{\mathrm{{ms}}}}$]'.format(sigma[j])
                if plot_validation:
                    siglabel += ', $\sigma_\mathrm{{val}}=$ {0:.3}'.format(quantities_validation['sigma'][j_validation])
            else:
                siglabel = None

            # we are already in real part subplot row in the col of the current quantity
            #spr.current_axes()

            plt.title(q)

            
            if plot_validation and q in quantities_validation:
                plt.plot(quantities_validation['mu'], 
                         quantities_validation[q][:, j_validation].real,
                         color=color_validation, marker=marker_validation, 
                         markersize=ms_validation, linestyle=linestyle_validation)
                    
            plt.plot(quantities['mu'], quantities[q][:, j].real, 
                     color=linecolor, label=siglabel)            

            if firstquant and j == N_sigma-1:                
                plt.legend(loc='best')
            
            
            # move to imag. part subplot row again            
            spr.nextrow(sharex=ax1)   
            
            if firstquant and j == 0:
                plt.ylabel('imag. part')
                
            if j == 0:
                plt.xlabel('$\mu$')
                
            if plot_validation and q in quantities_validation:
                plt.plot(quantities_validation['mu'], 
                         quantities_validation[q][:, j_validation].imag,
                         color=color_validation, marker=marker_validation, 
                         markersize=ms_validation, linestyle=linestyle_validation)
                    
            
            plt.plot(quantities['mu'], quantities[q][:, j].imag, color=linecolor)
            
            
            # move to real part subplot row again
            spr.nextrow(sharex=ax1)
                  
            # move to next quantity
            spr.nextcol(sharex=ax1)    
        
            firstquant = False    


def plot_quantities_composed(composed_quantities, quantities, inds_sigma_plot, 
                            colormap_sigma='winter',
                            plot_validation=False, quantities_validation={}, 
                            comp_quants_validat={},
                            marker_validation='o', ms_validation=2, 
                            color_validation='black', linestyle_validation='None'):

    sigma = quantities['sigma']
    N_sigma = sigma.shape[0]
    
    
    plt.figure()
    
    plt.suptitle('composed quantities (e.g., dot products)')    
    
    spr = misc.utils.SubplotRect(2, len(composed_quantities))    
    
    if plot_validation:
        inds_sigma_validation = get_sigma_inds_validation(inds_sigma_plot, 
                                                  sigma, 
                                                  quantities_validation['sigma'])

    for k_j, j in enumerate(inds_sigma_plot):
        
        
        # color
        cm = plt.get_cmap(colormap_sigma) 
        cNorm  = colors.Normalize(vmin=sigma[0], vmax=sigma[-1])
        scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cm)
        linecolor = scalarMap.to_rgba(sigma[j])
        
        if plot_validation:
            j_validation = inds_sigma_validation[k_j]   
        
        spr.first()
        ax1 = spr.current_axes()
        firstquant = True
        
        for q in composed_quantities.keys():        

            if firstquant and j == 0:
                plt.ylabel('real part')
            
            if firstquant and j in  [0, N_sigma//2, N_sigma-1]:
                siglabel = '$\sigma={0:.3}$ [mV/$\sqrt{{\mathrm{{ms}}}}$]'.format(sigma[j])
                if plot_validation:
                    siglabel += ', $\sigma_\mathrm{{val}}=$ {0:.3}'.format(quantities_validation['sigma'][j_validation])
            else:
                siglabel = None

            # we are already in real part subplot row in the col of the current quantity
            #spr.current_axes()

            plt.title(q)

            
            if plot_validation and q in comp_quants_validat:
                plt.plot(quantities_validation['mu'], 
                         comp_quants_validat[q][:, j_validation].real,
                         color=color_validation, marker=marker_validation, 
                         markersize=ms_validation, linestyle=linestyle_validation)
                    
            plt.plot(quantities['mu'], composed_quantities[q][:, j].real, 
                     color=linecolor, label=siglabel)            

            if firstquant and j == N_sigma-1:                
                plt.legend(loc='best')
            
            
            # move to imag. part subplot row again            
            spr.nextrow(sharex=ax1)   
            
            if firstquant and j == 0:
                plt.ylabel('imag. part')
                
            if j == 0:
                plt.xlabel('$\mu$')
                
            if plot_validation and q in comp_quants_validat:
                plt.plot(quantities_validation['mu'], 
                         comp_quants_validat[q][:, j_validation].imag,
                         color=color_validation, marker=marker_validation, 
                         markersize=ms_validation, linestyle=linestyle_validation)
                    
            
            plt.plot(quantities['mu'], composed_quantities[q][:, j].imag, color=linecolor)
            
            
            # move to real part subplot row again
            spr.nextrow(sharex=ax1)
                  
            # move to next quantity
            spr.nextcol(sharex=ax1)    
        
            firstquant = False    


def plot_raw_spectrum_sigma(lambda_all, mu, sigma, sigma_inds, max_per_fig=6, colormap='jet'):
    
    N_eigvals = lambda_all.shape[0]
    N_sigma = sigma.shape[0]
    
    for sii in range(0, len(sigma_inds), max_per_fig):
        sigma_inds_fig = sigma_inds[sii:min(sii+max_per_fig, N_sigma)]
        # subplot variable: sigma 
        plt.figure()
        plt.suptitle('full spectrum in $\mu,\sigma$ space by $\sigma$')
        subplotid = 1 # left/right real/imag, rows: eigvals
        N_plotcols = len(sigma_inds_fig) # N_sigma//sigma_skip_inds+(1 if N_sigma % sigma_skip_inds > 0 else 0)
        # axis sharing
        ax_real = plt.subplot(2, N_plotcols, subplotid, sharex=None, sharey=None)
        ax_imag = plt.subplot(2, N_plotcols, subplotid+N_plotcols, sharex=ax_real, sharey=None)
        for j in sigma_inds_fig: #  range(0, N_sigma, sigma_skip_inds):
            
            
            for k in range(N_eigvals):
                # labels
                if k in range(N_eigvals) and j==0:
                    eiglabel = '$\lambda_{0}^\mathrm{{raw}}$'.format(k+1)
                else:
                    eiglabel = None
                    
                # color
                cm = plt.get_cmap(colormap) 
                cNorm  = colors.Normalize(vmin=0, vmax=N_eigvals-1)
                scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cm)
                linecolor = scalarMap.to_rgba(k)
                
                # eigenval: real part
                plt.subplot(2, N_plotcols, subplotid, 
                            sharex=ax_real if subplotid > 1 else None, 
                            sharey=ax_real if subplotid > 1 else None)
                plt.plot(mu, lambda_all[k, :, j].real, label=eiglabel, color=linecolor)
                if j==0:
                    if k==0:
                        plt.ylabel('$\Re(\lambda)$ [kHz]')
                if k==0:
                    plt.title('$\sigma={0:.3}$ [mV/$\sqrt{{\mathrm{{ms}}}}$]'.format(sigma[j]))
                    plt.ylim(np.amin(lambda_all.real), 0)
                
                # eigenval: imag part
                plt.subplot(2, N_plotcols, subplotid+N_plotcols, 
                            sharex=ax_real, 
                            sharey=ax_imag if subplotid > 1 else None)
                plt.plot(mu, lambda_all[k, :, j].imag, color=linecolor)
                if j==0:
                    if k==0:
                        plt.ylabel('$\Im(\lambda)$ [kHz]')
                if k==0: 
                    plt.xlabel('$\mu$ [mV/ms]')
                    plt.ylim(np.amin(lambda_all.imag), np.amax(lambda_all.imag))
            
            if j==0:
                plt.subplot(2, N_plotcols, subplotid, 
                            sharex=ax_real if subplotid > 1 else None, 
                            sharey=ax_real if subplotid > 1 else None)
                plt.legend(loc='best')
            
            subplotid += 1
        

def plot_raw_spectrum_eigvals(lambda_all, mu, sigma, colormap='winter'):
    
    N_eigvals = lambda_all.shape[0]
    N_sigma = lambda_all.shape[2]
    
    # subplot variable: eigenvalue
    plt.figure()
    plt.suptitle('full spectrum in $\mu,\sigma$ space by $\lambda$ index')
    subplotid = 1 # left/right real/imag, rows: eigvals
    ax_real = plt.subplot(N_eigvals, 2, subplotid, sharex=None, sharey=None)
    ax_imag = plt.subplot(N_eigvals, 2, subplotid+1, sharex=ax_real, sharey=None)
    for k in range(N_eigvals):
        
        for j in range(N_sigma):
            # labels
            if j in [0, N_sigma-1] and k==0:
                siglabel = '$\sigma={0:.3}$ [mV/$\sqrt{{\mathrm{{ms}}}}$]'.format(sigma[j])
            else:
                siglabel = None
                
            # color
            cm = plt.get_cmap(colormap) 
            cNorm  = colors.Normalize(vmin=sigma[0], vmax=sigma[-1])
            scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cm)
            linecolor = scalarMap.to_rgba(sigma[j])
            
            # eigenval: real part
            plt.subplot(N_eigvals, 2, subplotid, 
                        sharex=ax_real if subplotid > 1 else None, 
                        sharey=ax_real if subplotid > 1 else None)
            plt.plot(mu, lambda_all[k, :, j].real, label=siglabel, color=linecolor)
            if j==0:
                if k==0:
                    plt.title('real part')
                plt.ylim(np.amin(lambda_all.real), 0)
                plt.ylabel('$\Re(\lambda_{0}^\mathrm{{raw}})$ [kHz]'.format(k+1))
                if k==N_eigvals-1:
                    plt.xlabel('$\mu$ [mV/ms]')
            
            # eigenval: imag part
            plt.subplot(N_eigvals, 2, subplotid+1, 
                        sharex=ax_real, 
                        sharey=ax_imag if subplotid > 1 else None)
            plt.plot(mu, lambda_all[k, :, j].imag, color=linecolor)
            if j==0:
                if k==0:
                    plt.title('imag. part')
                plt.ylim(np.amin(lambda_all.imag), np.amax(lambda_all.imag))
                plt.ylabel('$\Im(\lambda_{0}^\mathrm{{raw}})$ [kHz]'.format(k+1))
                if k==N_eigvals-1:
                    plt.xlabel('$\mu$ [mV/s]')
    
        if k==0:
            plt.subplot(N_eigvals, 2, subplotid, 
                        sharex=ax_real if subplotid > 1 else None, 
                        sharey=ax_real if subplotid > 1 else None)
            plt.legend(loc='best')
        subplotid += 2
        



