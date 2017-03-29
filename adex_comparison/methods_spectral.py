# -*- coding: utf-8 -*-

'''
Solver framework for the eigenvalue problem associated with the Fokker-Planck operator of 
(linear or non-linear) integrate-and-fire model neurons subject to white noise input.

Work in progress since January 2014; current state: January 7, 2016

@author: Moritz Augustin <augustin@ni.tu-berlin.de>
'''

# TODO: 
# - remove richardson & odeint code + parameters [keep backup and visualize the effect]


from __future__ import division # omit unwanted integer division
import math
import cmath
from warnings import warn, simplefilter
import numpy as np
from scipy.optimize import root
from scipy.integrate import odeint # only used when 'lsoda' is on
import scipy.linalg
import matplotlib.pyplot as plt
import multiprocessing
import itertools
import time
import tables
# for plotting
import matplotlib.colors as colors
import matplotlib.cm as cmx
import misc.utils

# if numba is not installed define dummy jit decorators
try:
    from numba import jit, njit
except:
    warn('numba missing, skipping just in time compilation')
    jit = lambda func: func
    njit = jit

# 1a) lambda_all, mu, sigma = specsolv.spectrum_load(filename, quantities=['r_inf', 'lambda_1',...])
# 1b) lambda_all = specsolv.spectrum_calc_rectangle(musigma)
#                    <> calls specsolve.real_eigenvalues & specsolve.eigenvalue_curve
# 2a) quantity_dict = specsolv.quantites_load(filename, quantities=None) # implies load all available
# 2b) quantity_dict = specsolv.quantities_calc_rectangle(musigma, quantities=['f_1', '...'])


# specsolv:
# compute_eigenvalue_rect(params, other parameters...) -> mu, sigma, lambda_raw?
# calls inner function compute_eigenvalue_curve_given_sigma(j) -> lambda_j
# load_eigvals(...) -> lambda_raw
#   postproc (compl conj. etc) NOT in spectralsolv
# compute_quantities_rect() for mu,sigma pairs -> quantities['r_inf'] = list of values for mu, sigma pairs
# calls inner function compute_quantities_given_mu_sigma in parallel


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


def compute_quantities_given_sigma(arg_tuple):
    params, quant_names, lambda_1, lambda_2, mu_arr, sigma_arr, j = arg_tuple
    
    
    N_mu = mu_arr.shape[0]
    sigma_j = sigma_arr[j]
    dmu = params['dmu_couplingterms']
    dsigma = params['dsigma_couplingterms']
    
    quant_j = {}
    
    comp_single_start = time.time()

    # creating quantity_j arrays initialized with zeros
    for q in quant_names:
        
        # real quantities
        if q in ['r_inf', 'dr_inf_dmu', 'dr_inf_dsigma', 
                 'V_mean_inf', 'dV_mean_inf_dmu', 'dV_mean_inf_dsigma']:
            quant_j[q] = np.zeros(N_mu)
            
        # complex quants
        elif q in ['f_1', 'f_2', 'psi_r_1', 'psi_r_2', 
                   'c_mu_1', 'c_mu_2', 'c_sigma_1', 'c_sigma_2']:
            quant_j[q] = np.zeros(N_mu) + 0j # complex dtype
            
        # unknown quants
        else:
            error('unknown/unsupported quantity {}'.format(q))
    
    params_quants = params.copy()
    
    # use more grid points only before threshold (new)    
    if 'quantities_grid_V_addpoints' in params:
        params_quants['grid_V_addpoints'] = params['quantities_grid_V_addpoints']
    
    # use more grid points for whole domain (old)
#    if 'quantities_grid_V_points' in params:
#        params_quants['grid_V_points'] = params['quantities_grid_V_points'] # finer grids especially for threshold fluxes    
    
    specsolv = SpectralSolver(params_quants)
    

    def thresholdflux(V_arr, phi, sigma, order):
    
        dV = V_arr[-1]-V_arr[-2] # assume uniform grid locally before threshold (at least 4 points)
    
        if params['verboselevel'] > 0:
            print('thresholdflux with order={}; phi[-4:]={}'.format(order, phi[-4:]))    
    
        if order == 1: # linear is too inaccurate for drift dominated regime
            phi_deriv_thr = -phi[-2]/dV
            
        elif order == 2: 
            phi_deriv_thr = (-2.*phi[-2] + phi[-3]/2. )/dV
            
        elif order == 3:
            phi_deriv_thr = (-3.*phi[-2] + 3./2.*phi[-3] - 1./3.*phi[-4])/dV
            
        else:
            error('order {} is not supported -- only 1, 2 or 3'.format(order))
            
        flux = -sigma**2/2. * phi_deriv_thr
        return flux

#    def phi0_rinf_noref(mu, sigma):
#        print('DEBUG: use again caching')        
#        V_arr, phi_0_noref, q = specsolv.eigenfunction(0., mu, sigma) # 0 is always an eigenvalue
#        phi_0_noref = phi_0_noref.real
#        normalize(phi_0_noref, V_arr) # ensure that phi_0(V) integrates to unity
#        
#        r_inf_noref = thresholdflux(V_arr, phi_0_noref, 
#                                    sigma, order=3)
#        
#        return V_arr, phi_0_noref, r_inf_noref

# from Josef: p0 integrates to unity (richardson), r0 respective r_inf
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
#            r_inf_noref_old = thresholdflux(V_arr_cache, phi_0_noref, 
#                                            sigma, order=params['threshold_interp_order'])
#            print('r_inf_new - r_inf_old = {} ; mu={}, sigma={}'.format(r_inf_noref-r_inf_noref_old, mu, sigma))
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
        
#        if mu>0:
#            plt.figure()
#            plt.subplot(221)
#            plt.title('real phi')
#            plt.plot(V_arr, phi_1.real, 'o', color='red')
#            plt.subplot(222)
#            plt.title('real psi')
#            plt.plot(V_arr, psi_1.real, 'o', color='black')
#            plt.subplot(223)
#            plt.title('imag phi')
#            plt.plot(V_arr, phi_1.imag, 'o', color='red')
#            plt.subplot(224)
#            plt.title('imag psi')
#            plt.plot(V_arr, psi_1.imag, 'o', color='black')
#            plt.show()
            
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
    
    
    specsolve = SpectralSolver(params.copy())
                                
    def eigenvalue_robust(mu_sigma_init, mu_sigma_target, lambda_init):
        mu_init, sigma_init = mu_sigma_init
        mu_target, sigma_target = mu_sigma_target
        mu_sigma_curve = np.array([[mu_init, sigma_init],
                                   [mu_target, sigma_target]])
        # alternative to the following: try/except -- interpolate coarse grid of lambda_all if error
        lambda_curve = specsolve.eigenvalue_curve(lambda_init, mu_sigma_curve)
        lambda_target = lambda_curve[-1]
        return lambda_target


    for i in range(N_mu):
        # abbreviations
        mu_i = mu_arr[i]
        lambda_1_ij = lambda_1[i, j]
        lambda_2_ij = lambda_2[i, j]
        
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
        
        
#        q_in = lambda q_list: any([q in quant_names for q in q_list])
#        
#        if q_in(['r_inf', 'V_mean_inf', 'c_mu_1', 'c_mu_2', 'c_sigma_1', 'c_sigma_2']):
#            V_arr, phi_0_noref, r_inf_noref = phi0_rinf_noref(mu_i, sigma_j)
#            #TODO: change the function above to use these values. maybe we have to add the 
#                # variables to the above scope first
#            # assume that the following eigenfunctions have the same V_arr grid
#        
#        if q_in(['f_1', 'psi_r_1']): # when not using finite diff also: c_*
#                
#            pass
#            #psi_1
#            #phi_1
#        if q_in(['f_2', 'psi_r_2']): # when not using finite diff also: c_*
#            pass            
#            #psi_2
#            #phi_2
    
    
        for q in quant_names:
            
            
            if q == 'r_inf':
                r_inf = rinf_ref(mu_i, sigma_j)
            
            if q == 'dr_inf_dmu':
                r_inf_plus_mu = rinf_ref(mu_i+dmu, sigma_j)
                r_inf_minus_mu = rinf_ref(mu_i-dmu, sigma_j)
                # central difference quotient
                dr_inf_dmu = (r_inf_plus_mu - r_inf_minus_mu) / (2*dmu)
            
            if q == 'dr_inf_dsigma':
                r_inf_plus_sigma = rinf_ref(mu_i, sigma_j+dsigma)
                r_inf_minus_sigma = rinf_ref(mu_i, sigma_j-dsigma)
                # central difference quotient
                dr_inf_dsigma = (r_inf_plus_sigma - r_inf_minus_sigma) / (2*dsigma)
            
            if q == 'V_mean_inf':
                V_mean_inf = Vmeaninf_noref(mu_i, sigma_j)
            
            if q == 'dV_mean_inf_dmu':
                V_mean_inf_plus_mu = Vmeaninf_noref(mu_i+dmu, sigma_j)
                V_mean_inf_minus_mu = Vmeaninf_noref(mu_i-dmu, sigma_j)
                # central difference quotient
                dV_mean_inf_dmu = (V_mean_inf_plus_mu - V_mean_inf_minus_mu) / (2*dmu)
            
            if q == 'dV_mean_inf_dsigma':
                V_mean_inf_plus_sigma = Vmeaninf_noref(mu_i, sigma_j+dsigma)
                V_mean_inf_minus_sigma = Vmeaninf_noref(mu_i, sigma_j-dsigma)
                # central difference quotient
                dV_mean_inf_dsigma = (V_mean_inf_plus_sigma - V_mean_inf_minus_sigma) / (2*dsigma)
            
            # f_k is the flux of the k-th eigenfunction phi_k evaluated at the threshold V_s 
            # (V_s is the right most point in the V_arr grid cell where we use backwards finite 
            # differences respecting the absorbing boudary condition)
            # note that in contrast to the manuscript we normalize phi that yields q_Nv != 1 
            # in general and thus f_1, f_2 != 1
            if q == 'f_1':
                V_arr, phi_1, q_1 = phi1(mu_i, sigma_j, lambda_1_ij)
#                f_1 = sigma_j**2 / 2.0 * phi_1[-2] / (V_arr[-1]-V_arr[-2])
                f_1 = q_1[-1] # threshold flux
#                f_1_old = thresholdflux(V_arr, phi_1, sigma_j, order=params['threshold_interp_order'])
                # old version: phi_1 is assumed to fulfill phi_1[-1] = 0 (due to initialization/boundary cond.) 
                # uses that phi_1 is normalized by using psi_1 which remains as computed from method eigenfunction
#                print('|f_1_new - f_1_old| = {} ; mu={}, sigma={}'.format(abs(f_1-f_1_old), mu_i, sigma_j))
                
            if q == 'f_2':
                V_arr, phi_2, q_2 = phi2(mu_i, sigma_j, lambda_2_ij)
#                f_2 = sigma_j**2 / 2.0 * phi_2[-2] / (V_arr[-1]-V_arr[-2])
                f_2 = q_2[-1] # threshold flux
                # old version: phi_2 is assumed to fulfill phi_2[-1] = 0 (due to initialization/boundary cond.) 
#                f_2_old = thresholdflux(V_arr, phi_2, sigma_j, order=params['threshold_interp_order'])
                # uses that phi_2 is normalized by using psi_2 which remains as computed from method eigenfunction   
#                print('|f_2_new - f_2_old| = {} ; mu={}, sigma={}'.format(abs(f_2-f_2_old), mu_i, sigma_j))
            
            # psi_r_k is just the eigenfunction psi_k evaluated at the reset
            if q == 'psi_r_1':
                V_arr, psi_1 = psi1(mu_i, sigma_j, lambda_1_ij)
                k_r = np.argmin(np.abs(V_arr-params['V_r']))
                psi_r_1 = psi_1[k_r]
            
            if q == 'psi_r_2':
                V_arr, psi_2 = psi2(mu_i, sigma_j, lambda_2_ij)
                k_r = np.argmin(np.abs(V_arr-params['V_r']))
                psi_r_2 = psi_2[k_r]
            
            
                
            # inner product between (discretized) partial derivative of psi w.r.t mu and 
            # the stationary distribution of active (i.e. non refractory) neurons
            if q == 'c_mu_1':
                
                # we need to evaluate psi_1(mu+-dmu) and thus lambda_1(mu+-dmu)
                lambda_1_plus_mu = eigenvalue_robust((mu_i, sigma_j), (mu_i+dmu, sigma_j), lambda_1_ij)
                V_arr, psi_1_plus_mu = psi1(mu_i+dmu, sigma_j, lambda_1_plus_mu, cache=False)
                
                lambda_1_minus_mu = eigenvalue_robust((mu_i, sigma_j), (mu_i-dmu, sigma_j), lambda_1_ij)
                V_arr, psi_1_minus_mu = psi1(mu_i-dmu, sigma_j, lambda_1_minus_mu, cache=False)
                
                if params['verboselevel'] > 0:
                    print('c_mu_1: lambda_1_plus_mu={}, lambda_1_minus_mu={}, lambda_1_ij={}'.format(lambda_1_plus_mu, lambda_1_minus_mu, lambda_1_ij))                
                
                # discretization of the partial derivative of psi w.r.t mu
                dpsi_1_dmu = (psi_1_plus_mu - psi_1_minus_mu)/(2*dmu)
                V_arr, phi_0_noref, r_inf_noref = phi0_rinf_noref(mu_i, sigma_j)

                if params['verboselevel'] > 0:                
                    print('c_mu_1: dpsi_1_dmu(absmin,absmax)=({}, {}) phi0(absmin,absmax)=({}, {})'.format(np.abs(dpsi_1_dmu).min(), np.abs(dpsi_1_dmu).max(), phi_0_noref.min(), phi_0_noref.max()))
                
                c_mu_1 = inner_prod(dpsi_1_dmu, phi_0_noref, V_arr)

                if params['verboselevel'] > 0:                
                    print('c_mu_1={}'.format(c_mu_1))
              
              
            if q == 'c_mu_2':
                
                # we need to evaluate psi_2(mu+-dmu) and thus lambda_2(mu+-dmu)
                lambda_2_plus_mu = eigenvalue_robust((mu_i, sigma_j), (mu_i+dmu, sigma_j), lambda_2_ij)
                V_arr, psi_2_plus_mu = psi2(mu_i+dmu, sigma_j, lambda_2_plus_mu, cache=False)
                
                lambda_2_minus_mu = eigenvalue_robust((mu_i, sigma_j), (mu_i-dmu, sigma_j), lambda_2_ij)
                V_arr, psi_2_minus_mu = psi2(mu_i-dmu, sigma_j, lambda_2_minus_mu, cache=False)
                
                # discretization of the partial derivative of psi w.r.t mu
                dpsi_2_dmu = (psi_2_plus_mu - psi_2_minus_mu)/(2*dmu)
                V_arr, phi_0_noref, r_inf_noref = phi0_rinf_noref(mu_i, sigma_j)
                
                # inner product between 
                #   (discretized) partial derivative of psi w.r.t mu and 
                # the stationary distribution of active (i.e. non refractory) neurons
                c_mu_2 = inner_prod(dpsi_2_dmu, phi_0_noref, V_arr)
            
            
            if q == 'c_sigma_1':
                
                # we need to evaluate psi_1(sigma+-dsigma) and thus lambda_1(sigma+-dsigma)
                lambda_1_plus_sigma = eigenvalue_robust((mu_i, sigma_j), (mu_i, sigma_j+dsigma), lambda_1_ij)
                V_arr, psi_1_plus_sigma = psi1(mu_i, sigma_j+dsigma, lambda_1_plus_sigma, cache=False)
                
                lambda_1_minus_sigma = eigenvalue_robust((mu_i, sigma_j), (mu_i, sigma_j-dsigma), lambda_1_ij)
                V_arr, psi_1_minus_sigma = psi1(mu_i, sigma_j-dsigma, lambda_1_minus_sigma, cache=False)
                
                # discretization of the partial derivative of psi w.r.t sigma
                dpsi_1_dsigma = (psi_1_plus_sigma - psi_1_minus_sigma)/(2*dsigma)
                V_arr, phi_0_noref, r_inf_noref = phi0_rinf_noref(mu_i, sigma_j)
                
                c_sigma_1 = inner_prod(dpsi_1_dsigma, phi_0_noref, V_arr)
            
            
            if q == 'c_sigma_2':
                
                # we need to evaluate psi_1(sigma+-dsigma) and thus lambda_1(sigma+-dsigma)
                lambda_2_plus_sigma = eigenvalue_robust((mu_i, sigma_j), (mu_i, sigma_j+dsigma), lambda_2_ij)
                V_arr, psi_2_plus_sigma = psi2(mu_i, sigma_j+dsigma, lambda_2_plus_sigma, cache=False)
                
                lambda_2_minus_sigma = eigenvalue_robust((mu_i, sigma_j), (mu_i, sigma_j-dsigma), lambda_2_ij)
                V_arr, psi_2_minus_sigma = psi2(mu_i, sigma_j-dsigma, lambda_2_minus_sigma, cache=False)
                
                # discretization of the partial derivative of psi w.r.t sigma
                dpsi_2_dsigma = (psi_2_plus_sigma - psi_2_minus_sigma)/(2*dsigma)
                V_arr, phi_0_noref, r_inf_noref = phi0_rinf_noref(mu_i, sigma_j)
                
                c_sigma_2 = inner_prod(dpsi_2_dsigma, phi_0_noref, V_arr)
        

            # remove the if condition only keep the body after all quantities are there
            if q in locals().keys():
                quant_j[q][i] = locals()[q] # put local vars in dict


    comp_single_duration = time.time() - comp_single_start        
    
    return j, quant_j, comp_single_duration


#def compute_quantities_given_sigma_old(arg_tuple):
#    params, quant_names, lambda_1, lambda_2, mu_arr, sigma_arr, j = arg_tuple
#    
#    
#    N_mu = mu_arr.shape[0]
#    sigma_j = sigma_arr[j]
#    dmu = params['dmu_couplingterms']
#    dsigma = params['dsigma_couplingterms']
#    
#    quant_j = {}
#    
#    comp_single_start = time.time()
#
#    # creating quantity_j arrays initialized with zeros
#    for q in quant_names:
#        
#        # real quantities
#        if q in ['r_inf', 'dr_inf_dmu', 'dr_inf_dsigma', 
#                 'V_mean_inf', 'dV_mean_inf_dmu', 'dV_mean_inf_dsigma']:
#            quant_j[q] = np.zeros(N_mu)
#            
#        # complex quants
#        elif q in ['f_1', 'f_2', 'psi_r_1', 'psi_r_2', 
#                   'c_mu_1', 'c_mu_2', 'c_sigma_1', 'c_sigma_2']:
#            quant_j[q] = np.zeros(N_mu) + 0j # complex dtype
#            
#        # unknown quants
#        else:
#            error('unknown/unsupported quantity {}'.format(q))
#    
#    params_quants = params.copy()
#    
#    # use more grid points only before threshold (new)    
#    if 'quantities_grid_V_addpoints' in params:
#        params_quants['grid_V_addpoints'] = params['quantities_grid_V_addpoints']
#    
#    # use more grid points for whole domain (old)
##    if 'quantities_grid_V_points' in params:
##        params_quants['grid_V_points'] = params['quantities_grid_V_points'] # finer grids especially for threshold fluxes    
#    
#    specsolv = SpectralSolver(params_quants)
#    
#
#    def thresholdflux(V_arr, phi, sigma, order):
#    
#        dV = V_arr[-1]-V_arr[-2] # assume uniform grid locally before threshold (at least 4 points)
#    
#        if params['verboselevel'] > 0:
#            print('thresholdflux with order={}; phi[-4:]={}'.format(order, phi[-4:]))    
#    
#        if order == 1: # linear is too inaccurate for drift dominated regime
#            phi_deriv_thr = -phi[-2]/dV
#            
#        elif order == 2: 
#            phi_deriv_thr = (-2.*phi[-2] + phi[-3]/2. )/dV
#            
#        elif order == 3:
#            phi_deriv_thr = (-3.*phi[-2] + 3./2.*phi[-3] - 1./3.*phi[-4])/dV
#            
#        else:
#            error('order {} is not supported -- only 1, 2 or 3'.format(order))
#            
#        flux = -sigma**2/2. * phi_deriv_thr
#        return flux
#
##    def phi0_rinf_noref(mu, sigma):
##        print('DEBUG: use again caching')        
##        V_arr, phi_0_noref = specsolv.eigenfunction(0., mu, sigma) # 0 is always an eigenvalue
##        phi_0_noref = phi_0_noref.real
##        normalize(phi_0_noref, V_arr) # ensure that phi_0(V) integrates to unity
##        
##        r_inf_noref = thresholdflux(V_arr, phi_0_noref, 
##                                    sigma, order=3)
##        
##        return V_arr, phi_0_noref, r_inf_noref
#
## from Josef: p0 integrates to unity (richardson), r0 respective r_inf
##r0ref = r0/(1+r0*Tref);
##p0 = r0ref * p0/r0;
##%prob. density only reflecting nonrefr. proportion which integrates to r0ref/r0
##Vmean_sps = dV*sum(V.*p0) + (1-r0ref/r0)*(Vs+Vr)/2;  %remark: (1-r0ref/r0)==r0ref*Tref
#
#
#    def phi0_rinf_noref(mu, sigma):
#        # use global caching variables to prevent redundant computation of stationary mode
#        global phi_0_noref_cache, V_arr_cache 
#        if phi_0_noref_cache[(mu, sigma)] is None: # only compute stationary mode if not already stored temporarily            
#            V_arr_cache, phi_0_noref = specsolv.eigenfunction(0., mu, sigma) # 0 is always an eigenvalue
#            phi_0_noref = phi_0_noref.real
#            normalize(phi_0_noref, V_arr_cache) # ensure that phi_0(V) integrates to unity
#            phi_0_noref_cache[(mu, sigma)] = phi_0_noref
#        else:
#            phi_0_noref = phi_0_noref_cache[(mu, sigma)]
#        
#        
#        r_inf_noref = thresholdflux(V_arr_cache, phi_0_noref, 
#                                    sigma, order=params['threshold_interp_order'])
#        
#        return V_arr_cache, phi_0_noref, r_inf_noref
#        
#    # assuming existence of: V_arr, phi_0_noref, r_inf_noref
#    def rinf_ref(mu, sigma):
#        # get stationary distribution neclecting refractory period
#        V_arr, phi_0_noref, r_inf_noref = phi0_rinf_noref(mu, sigma)
#        
#        # respect the refractory period which is assumed to be nonnegative
#        tau_ref = params['tau_ref']
#        assert tau_ref >= 0
#        r_inf = 1.0 / (1.0/r_inf_noref + tau_ref)
#        
#        return r_inf
#        
#    def Vmeaninf_noref(mu, sigma):
#        # get stationary distribution neclecting refractory period
#        V_arr, phi_0_noref, r_inf_noref = phi0_rinf_noref(mu, sigma)
#        
#        # use midpoint integration for integral(V*phi0(V)*dV)
#        V_center = 0.5 * (V_arr[:-1] + V_arr[1:])
#        phi0_center = 0.5 * (phi_0_noref[:-1] + phi_0_noref[1:])
#        V_mean_inf_noref =  np.sum(V_center * phi0_center * np.diff(V_arr))
#
#        return V_mean_inf_noref
#        
#    
#    # computing eigenfunction psi_1 of adjoint operator, lambda_1 should be an eigenvalue
#    def psi1(mu, sigma, lambda_1, cache=True):
#        # use global caching variable to prevent redundant computation
#        global psi_1_cache, V_arr_cache
#        if not cache or psi_1_cache is None: # only compute stationary mode if not already stored temporarily            
#            V_arr_cache, psi_1_cache = specsolv.eigenfunction(lambda_1, mu, sigma, 
#                                                              adjoint=True) 
#        return V_arr_cache, psi_1_cache
#        
#    
#    # computing eigenfunction psi_2 of adjoint operator, lambda_2 should be an eigenvalue
#    def psi2(mu, sigma, lambda_2, cache=True):
#        # use global caching variable to prevent redundant computation
#        global psi_2_cache, V_arr_cache
#        if not cache or psi_2_cache is None: # only compute stationary mode if not already stored temporarily            
#            V_arr_cache, psi_2_cache = specsolv.eigenfunction(lambda_2, mu, sigma, 
#                                                              adjoint=True) 
#        return V_arr_cache, psi_2_cache
#
#    
#    def phi1(mu, sigma, lambda_1):
#        V_arr, psi_1 = psi1(mu, sigma, lambda_1)
#        V_arr, phi_1 = specsolv.eigenfunction(lambda_1, mu, sigma)
#        # binormalize phi_1 s.t. the inner product (psi_1, phi_1) = 1
#        inprod1 = inner_prod(psi_1, phi_1, V_arr)
#        if params['verboselevel'] > 0:
#            print('inner product (phi1)={}'.format(inprod1))
#        phi_1 /= inprod1
#        
##        if mu>0:
##            plt.figure()
##            plt.subplot(221)
##            plt.title('real phi')
##            plt.plot(V_arr, phi_1.real, 'o', color='red')
##            plt.subplot(222)
##            plt.title('real psi')
##            plt.plot(V_arr, psi_1.real, 'o', color='black')
##            plt.subplot(223)
##            plt.title('imag phi')
##            plt.plot(V_arr, phi_1.imag, 'o', color='red')
##            plt.subplot(224)
##            plt.title('imag psi')
##            plt.plot(V_arr, psi_1.imag, 'o', color='black')
##            plt.show()
#            
#        return V_arr, phi_1
#        
#    def phi2(mu, sigma, lambda_2):
#        V_arr, psi_2 = psi2(mu, sigma, lambda_2)
#        V_arr, phi_2 = specsolv.eigenfunction(lambda_2, mu, sigma)
#        # binormalize phi_2 s.t. the inner product (psi_2, phi_2) = 1
#        inprod2 = inner_prod(psi_2, phi_2, V_arr)
#        if params['verboselevel'] > 0:
#            print('inner product (phi2)={}'.format(inprod2))
#        phi_2 /= inprod2
#        return V_arr, phi_2
#    
#    
#    specsolve = SpectralSolver(params.copy())
#                                
#    def eigenvalue_robust(mu_sigma_init, mu_sigma_target, lambda_init):
#        mu_init, sigma_init = mu_sigma_init
#        mu_target, sigma_target = mu_sigma_target
#        mu_sigma_curve = np.array([[mu_init, sigma_init],
#                                   [mu_target, sigma_target]])
#        # alternative to the following: try/except -- interpolate coarse grid of lambda_all if error
#        lambda_curve = specsolve.eigenvalue_curve(lambda_init, mu_sigma_curve)
#        lambda_target = lambda_curve[-1]
#        return lambda_target
#
#
#    for i in range(N_mu):
#        # abbreviations
#        mu_i = mu_arr[i]
#        lambda_1_ij = lambda_1[i, j]
#        lambda_2_ij = lambda_2[i, j]
#        
#        print('======================= mu={}, sigma={} == computing quantities'.format(mu_i, sigma_j))
#
#        # create variable for temporarily stored functions such as phi0 
#        # which can be accessed from the following helper functions
#        # all mu/sigma combinations required as for each pair r_inf and V_mean_inf depend on them
#        global phi_0_noref_cache
#        phi_0_noref_cache = {(mu_i, sigma_j):       None,
#                             (mu_i+dmu, sigma_j):   None,
#                             (mu_i-dmu, sigma_j):   None,
#                             (mu_i, sigma_j+dsigma): None,
#                             (mu_i, sigma_j-dsigma): None,}
#    
#        # assuming same V grid for all mu sigma and all eigenfunctions
#        global V_arr_cache
#        V_arr_cache = None 
#        
#        # psi1,2 are required for (i) f_1,2 due to phi normalization => biorth, and (ii) psi_r_* itself
#        # (however, we disable this caching for computation of c_k_x quantities 
#        # where psi(mu+-dmu,sigma+-dsigma) are required)
#        global psi_1_cache
#        psi_1_cache = None
#        global psi_2_cache
#        psi_2_cache = None
#        
#        
##        q_in = lambda q_list: any([q in quant_names for q in q_list])
##        
##        if q_in(['r_inf', 'V_mean_inf', 'c_mu_1', 'c_mu_2', 'c_sigma_1', 'c_sigma_2']):
##            V_arr, phi_0_noref, r_inf_noref = phi0_rinf_noref(mu_i, sigma_j)
##            #TODO: change the function above to use these values. maybe we have to add the 
##                # variables to the above scope first
##            # assume that the following eigenfunctions have the same V_arr grid
##        
##        if q_in(['f_1', 'psi_r_1']): # when not using finite diff also: c_*
##                
##            pass
##            #psi_1
##            #phi_1
##        if q_in(['f_2', 'psi_r_2']): # when not using finite diff also: c_*
##            pass            
##            #psi_2
##            #phi_2
#    
#    
#        for q in quant_names:
#            
#            
#            if q == 'r_inf':
#                r_inf = rinf_ref(mu_i, sigma_j)
#            
#            if q == 'dr_inf_dmu':
#                r_inf_plus_mu = rinf_ref(mu_i+dmu, sigma_j)
#                r_inf_minus_mu = rinf_ref(mu_i-dmu, sigma_j)
#                # central difference quotient
#                dr_inf_dmu = (r_inf_plus_mu - r_inf_minus_mu) / (2*dmu)
#            
#            if q == 'dr_inf_dsigma':
#                r_inf_plus_sigma = rinf_ref(mu_i, sigma_j+dsigma)
#                r_inf_minus_sigma = rinf_ref(mu_i, sigma_j-dsigma)
#                # central difference quotient
#                dr_inf_dsigma = (r_inf_plus_sigma - r_inf_minus_sigma) / (2*dsigma)
#            
#            if q == 'V_mean_inf':
#                V_mean_inf = Vmeaninf_noref(mu_i, sigma_j)
#            
#            if q == 'dV_mean_inf_dmu':
#                V_mean_inf_plus_mu = Vmeaninf_noref(mu_i+dmu, sigma_j)
#                V_mean_inf_minus_mu = Vmeaninf_noref(mu_i-dmu, sigma_j)
#                # central difference quotient
#                dV_mean_inf_dmu = (V_mean_inf_plus_mu - V_mean_inf_minus_mu) / (2*dmu)
#            
#            if q == 'dV_mean_inf_dsigma':
#                V_mean_inf_plus_sigma = Vmeaninf_noref(mu_i, sigma_j+dsigma)
#                V_mean_inf_minus_sigma = Vmeaninf_noref(mu_i, sigma_j-dsigma)
#                # central difference quotient
#                dV_mean_inf_dsigma = (V_mean_inf_plus_sigma - V_mean_inf_minus_sigma) / (2*dsigma)
#            
#            # f_k is the flux of the k-th eigenfunction phi_k evaluated at the threshold V_s 
#            # (V_s is the right most point in the V_arr grid cell where we use backwards finite 
#            # differences respecting the absorbing boudary condition)
#            # note that in contrast to the manuscript we normalize phi that yields q_Nv != 1 
#            # in general and thus f_1, f_2 != 1
#            if q == 'f_1':
#                V_arr, phi_1 = phi1(mu_i, sigma_j, lambda_1_ij)
##                f_1 = sigma_j**2 / 2.0 * phi_1[-2] / (V_arr[-1]-V_arr[-2])
#                f_1 = thresholdflux(V_arr, phi_1, sigma_j, order=params['threshold_interp_order'])
#                # phi_1 is # assumed to fulfill phi_1[-1] = 0 (due to initialization/boundary cond.) 
#                # phi_1 is normalized by using psi_1 which remains as computed from method eigenfunction
#                if params['verboselevel'] > 0:
#                    print('f_1={}'.format(f_1))
#                
#            if q == 'f_2':
#                V_arr, phi_2 = phi2(mu_i, sigma_j, lambda_2_ij)
##                f_2 = sigma_j**2 / 2.0 * phi_2[-2] / (V_arr[-1]-V_arr[-2])
#                f_2 = thresholdflux(V_arr, phi_2, sigma_j, order=params['threshold_interp_order'])                
#                # phi_2 is # assumed to fulfill phi_2[-1] = 0 (due to initialization/boundary cond.) 
#                # phi_2 is normalized by using psi_2 which remains as computed from method eigenfunction
#            
#            # psi_r_k is just the eigenfunction psi_k evaluated at the reset
#            if q == 'psi_r_1':
#                V_arr, psi_1 = psi1(mu_i, sigma_j, lambda_1_ij)
#                k_r = np.argmin(np.abs(V_arr-params['V_r']))
#                psi_r_1 = psi_1[k_r]
#                if params['verboselevel'] > 0:
#                    print('psi_r_1={}'.format(psi_r_1))
#            
#            if q == 'psi_r_2':
#                V_arr, psi_2 = psi2(mu_i, sigma_j, lambda_2_ij)
#                k_r = np.argmin(np.abs(V_arr-params['V_r']))
#                psi_r_2 = psi_2[k_r]
#            
#            
#                
#            # inner product between (discretized) partial derivative of psi w.r.t mu and 
#            # the stationary distribution of active (i.e. non refractory) neurons
#            if q == 'c_mu_1':
#                
#                # we need to evaluate psi_1(mu+-dmu) and thus lambda_1(mu+-dmu)
#                lambda_1_plus_mu = eigenvalue_robust((mu_i, sigma_j), (mu_i+dmu, sigma_j), lambda_1_ij)
#                V_arr, psi_1_plus_mu = psi1(mu_i+dmu, sigma_j, lambda_1_plus_mu, cache=False)
#                
#                lambda_1_minus_mu = eigenvalue_robust((mu_i, sigma_j), (mu_i-dmu, sigma_j), lambda_1_ij)
#                V_arr, psi_1_minus_mu = psi1(mu_i-dmu, sigma_j, lambda_1_minus_mu, cache=False)
#                
#                if params['verboselevel'] > 0:
#                    print('c_mu_1: lambda_1_plus_mu={}, lambda_1_minus_mu={}, lambda_1_ij={}'.format(lambda_1_plus_mu, lambda_1_minus_mu, lambda_1_ij))                
#                
#                # discretization of the partial derivative of psi w.r.t mu
#                dpsi_1_dmu = (psi_1_plus_mu - psi_1_minus_mu)/(2*dmu)
#                V_arr, phi_0_noref, r_inf_noref = phi0_rinf_noref(mu_i, sigma_j)
#
#                if params['verboselevel'] > 0:                
#                    print('c_mu_1: dpsi_1_dmu(absmin,absmax)=({}, {}) phi0(absmin,absmax)=({}, {})'.format(np.abs(dpsi_1_dmu).min(), np.abs(dpsi_1_dmu).max(), phi_0_noref.min(), phi_0_noref.max()))
#                
#                c_mu_1 = inner_prod(dpsi_1_dmu, phi_0_noref, V_arr)
#
#                if params['verboselevel'] > 0:                
#                    print('c_mu_1={}'.format(c_mu_1))
#              
#              
#            if q == 'c_mu_2':
#                
#                # we need to evaluate psi_2(mu+-dmu) and thus lambda_2(mu+-dmu)
#                lambda_2_plus_mu = eigenvalue_robust((mu_i, sigma_j), (mu_i+dmu, sigma_j), lambda_2_ij)
#                V_arr, psi_2_plus_mu = psi2(mu_i+dmu, sigma_j, lambda_2_plus_mu, cache=False)
#                
#                lambda_2_minus_mu = eigenvalue_robust((mu_i, sigma_j), (mu_i-dmu, sigma_j), lambda_2_ij)
#                V_arr, psi_2_minus_mu = psi2(mu_i-dmu, sigma_j, lambda_2_minus_mu, cache=False)
#                
#                # discretization of the partial derivative of psi w.r.t mu
#                dpsi_2_dmu = (psi_2_plus_mu - psi_2_minus_mu)/(2*dmu)
#                V_arr, phi_0_noref, r_inf_noref = phi0_rinf_noref(mu_i, sigma_j)
#                
#                # inner product between 
#                #   (discretized) partial derivative of psi w.r.t mu and 
#                # the stationary distribution of active (i.e. non refractory) neurons
#                c_mu_2 = inner_prod(dpsi_2_dmu, phi_0_noref, V_arr)
#            
#            
#            if q == 'c_sigma_1':
#                
#                # we need to evaluate psi_1(sigma+-dsigma) and thus lambda_1(sigma+-dsigma)
#                lambda_1_plus_sigma = eigenvalue_robust((mu_i, sigma_j), (mu_i, sigma_j+dsigma), lambda_1_ij)
#                V_arr, psi_1_plus_sigma = psi1(mu_i, sigma_j+dsigma, lambda_1_plus_sigma, cache=False)
#                
#                lambda_1_minus_sigma = eigenvalue_robust((mu_i, sigma_j), (mu_i, sigma_j-dsigma), lambda_1_ij)
#                V_arr, psi_1_minus_sigma = psi1(mu_i, sigma_j-dsigma, lambda_1_minus_sigma, cache=False)
#                
#                # discretization of the partial derivative of psi w.r.t sigma
#                dpsi_1_dsigma = (psi_1_plus_sigma - psi_1_minus_sigma)/(2*dsigma)
#                V_arr, phi_0_noref, r_inf_noref = phi0_rinf_noref(mu_i, sigma_j)
#                
#                c_sigma_1 = inner_prod(dpsi_1_dsigma, phi_0_noref, V_arr)
#            
#            
#            if q == 'c_sigma_2':
#                
#                # we need to evaluate psi_1(sigma+-dsigma) and thus lambda_1(sigma+-dsigma)
#                lambda_2_plus_sigma = eigenvalue_robust((mu_i, sigma_j), (mu_i, sigma_j+dsigma), lambda_2_ij)
#                V_arr, psi_2_plus_sigma = psi2(mu_i, sigma_j+dsigma, lambda_2_plus_sigma, cache=False)
#                
#                lambda_2_minus_sigma = eigenvalue_robust((mu_i, sigma_j), (mu_i, sigma_j-dsigma), lambda_2_ij)
#                V_arr, psi_2_minus_sigma = psi2(mu_i, sigma_j-dsigma, lambda_2_minus_sigma, cache=False)
#                
#                # discretization of the partial derivative of psi w.r.t sigma
#                dpsi_2_dsigma = (psi_2_plus_sigma - psi_2_minus_sigma)/(2*dsigma)
#                V_arr, phi_0_noref, r_inf_noref = phi0_rinf_noref(mu_i, sigma_j)
#                
#                c_sigma_2 = inner_prod(dpsi_2_dsigma, phi_0_noref, V_arr)
#        
#
#            # remove the if condition only keep the body after all quantities are there
#            if q in locals().keys():
#                quant_j[q][i] = locals()[q] # put local vars in dict
#
#
#    comp_single_duration = time.time() - comp_single_start    
#
#    
#    return j, quant_j, comp_single_duration



# very old quantity computation:

    #    # eigenvalue arrays
    #    lambda_1 = np.zeros((N_mu, N_sigma), dtype=np.complex128)
    #    lambda_2 = np.zeros_like(lambda_1)
    #    
    #    # quantity arrays
    #    r_inf = np.zeros((N_mu, N_sigma))
    #    V_mean_inf = np.zeros_like(r_inf)
    #    dr_inf_mu = np.zeros_like(r_inf)
    #    dr_inf_sigma = np.zeros_like(r_inf)
    #    f_1 = np.zeros_like(r_inf, dtype=np.complex128)
    #    f_2 = np.zeros_like(f_1)
    #    psiVr_1 = np.zeros_like(f_1)
    #    psiVr_2 = np.zeros_like(f_1)
    ##    d_1 = np.zeros_like(f_1) # helper quantity for the c_*_1 difference quotients
    ##    d_2 = np.zeros_like(f_1) # helper quantity for the c_*_2 difference quotients
    #    c_mu_1 = np.zeros_like(f_1)
    #    c_mu_2 = np.zeros_like(f_1)
    #    c_sigma_1 = np.zeros_like(f_1)
    #    c_sigma_2 = np.zeros_like(f_1)
    ##    g_12 = np.zeros_like(f_1) # helper quantity for the C_*_1_2 difference quotients
    ##    g_21 = np.zeros_like(f_1) # helper quantity for the C_*_2_1 difference quotients
    #    C_mu_1_2 = np.zeros_like(f_1)
    #    C_mu_2_1 = np.zeros_like(f_1)
    #    C_sigma_1_2 = np.zeros_like(f_1)
    #    C_sigma_2_1 = np.zeros_like(f_1)
    #    
    #    # note: our lambda_1 = maurizio's lambda_{-1}     and    our lambda_2 = maurizio's lambda 1
    #    lambda_1[:, :] = lambda_all[0, :, :]
    #    lambda_2[:, :] = lambda_all[1, :, :]
    #
    #    # QUANTITIES: compute required coefficients on the mu, sigma grid
    #    # (i) r_inf and V_mean_inf (based on phi_0=p_inf)
    #    # (ii) dr_inf_mu, dr_inf_sigma (based on r_inf)
    #    # (iii) f_1 and f_2 (based on phi_1 and phi_2 [PAIRWISE ORTHONORMALIZATION!], resp.)
    #    # (iv) c_mu_1, c_mu_2, c_sigma_1, c_sigma_2 (based on d_1,2 based on psi_1,2 [PAIRWISE ORTHONORM.!] and phi_0)
    #        
    #    
    # specsolv = SpectralSolver(params.copy())
    #    plt.figure()
        # it could make sense to parallelize this loop. BUT: all quantities need to be 
        # separately specified
#    inner_prod = lambda fun1, fun2, V: np.sum( 0.5*((fun1*fun2)[:-1] + (fun1*fun2)[1:]) * np.diff(V) ) 
    #    for j in range(N_sigma): #[N_sigma-3, N_sigma-2, N_sigma-1]: #range(N_sigma): #[0, N_sigma-1]: #range(N_sigma): 
    #        for i in range(N_mu): #[0, N_mu-5, N_mu-4, N_mu-3, N_mu-2, N_mu-1]: #range(N_mu): #[0, N_mu-1]: #range(N_mu):
    #            
    #            print('computing quantities for mu={}, sigma={}:'.format(mu[i], sigma[j]))         
    #            
    #            
    #            # V_mean and r_inf
    #            V_vec, phi_0 = specsolv.eigenfunction(0., mu[i], sigma[j]) # 0 is always an eigenvalue
    #             
    #            meanV = lambda pV, V: np.sum((V*pV.real)[:-1] * np.diff(V))
    #            threshFlux = lambda fun, V: -sigma[j]**2/2.0 * (fun[-1]-fun[-2]) / (V[-1]-V[-2])  # only if [1/ms]:  * 1000 # in Hz 
    #
    #            V_mean_inf[i, j] = meanV(phi_0, V_vec)
    #            r_inf[i, j] = threshFlux(phi_0, V_vec).real # imag. is 0 in this case
    #            
    #            V_vec_new, phi_1 = specsolv.eigenfunction(lambda_1[i, j], mu[i], sigma[j])
    #            V_vec_new, phi_2 = specsolv.eigenfunction(lambda_2[i, j], mu[i], sigma[j])
    #            
    #            V_vec_new, psi_1 = specsolv.eigenfunction(lambda_1[i, j], mu[i], sigma[j], adjoint=True)
    #            V_vec_new, psi_2 = specsolv.eigenfunction(lambda_2[i, j], mu[i], sigma[j], adjoint=True)
    #
    #            # normalize psi_j such that: <psi_j | phi_j> = integral of psi_j.H * phi_j = 1 
    #            # this leads to biorthonormalization, i.e. <psi_j | phi_i> = delta_ij 
    #            # (no complex conjugation required, cf. Mattia and Del Giudice 2002 PRE)
    #            norm_prod1 = inner_prod(psi_1, phi_1, V_vec)            
    #            norm_prod2 = inner_prod(psi_2, phi_2, V_vec)
    #            phi_1 = phi_1 / norm_prod1
    #            phi_2 = phi_2 / norm_prod2
    #                
    #            # f_1, f_2
    #            f_1[i, j] = threshFlux(phi_1, V_vec)
    #            f_2[i, j] = threshFlux(phi_2, V_vec)
    #
    #            # psi1Vr
    #            k_r = np.argmin(np.abs(V_vec-params['V_r']))
    #            psiVr_1[i, j] = psi_1[k_r]
    #            psiVr_2[i, j] = psi_2[k_r]
    #
    #           V_mean  and r_inf CONSIDERING REFRACTORY PERIOD    
    #           dr_inf and V mean derivatives w.r.t mu/sigma
    #
    #            # corrected c_* quantities & AND ADD sigma derivatives
    #            if i < N_mu-1:
    #                dmu = float(params['dmu_couplingterms']) #float(mu[i+1] - mu[i])
    #                dlambda = float(params['dlambda_couplingterms'])
    ##                is_before_transition = 0 < abs(lambda_1[i+1, j].imag) and abs(lambda_1[i+1, j].imag) < 1e-3
    ##                if is_before_transition:
    ##                    abstol_old = specsolv.params['solver_abstol']
    ##                    specsolv.params['solver_abstol'] = 1e-2
    #                specsolv.params['mu'] = mu[i]+dmu
    #                specsolv.params['sigma'] = sigma[j]
    #                try:
    #                    lambda_muplus = specsolv.eigenvalue(lambda_init=lambda_1[i, j])
    #                except:
    #                    print('lambda for mu+dmu did not converge at mu={}, dmu={}, <lambda={}'.format(mu[i], dmu, lambda_1[i, j]))
    #                    lambda_muplus = lambda_1[i, j]
    ##                if is_before_transition:
    ##                    specsolv.params['solver_abstol'] = abstol_old
    #                # the latter did not converge close to real->imag transition
    #                #mu_sigma_dmu = np.array([[mu[i], sigma[j]],
    #                #                         [mu[i]+dmu, sigma[j]]])
    #                #ev_curve = specsolv.eigenvalue_curve(lambda_1[i, j], mu_sigma_dmu)
    #                #lambda_muplus = ev_curve[-1]
    #
    #                lambda_diff = lambda_muplus-lambda_1[i, j] #lambda_1[i+1, j] - lambda_1[i, j] # lambda(mu+dmu) - lambda(mu)
    #                V_vec_null, psi_muplus = specsolv.eigenfunction(lambda_1[i, j], mu[i]+dmu, sigma[j], 
    #                                                                adjoint=True)
    #                V_vec_null, psi_lambdaplus = specsolv.eigenfunction(lambda_1[i, j]+dlambda, mu[i], sigma[j], 
    #                                                                adjoint=True)
    #                                                                
    #                if norm_psi:
    #                    psi_muplus /= norm_prod1
    #                    psi_lambdaplus /= norm_prod1
    #                diff_psi_mu = 1/dmu * psi_muplus + 1/dlambda * psi_lambdaplus * 1/dmu * lambda_diff
    #                c_mu_1[i, j] = inner_prod(diff_psi_mu, phi_0, V_vec)
    #            else:
    #                c_mu_1[i, j] = c_mu_1[i-1, j] # we just constant extrapolate instead of backwards fd.
    










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
    
    
    #    # remove very small imaginary parts (numerical zeros) for easier quantity computations
    #    for m in range(N_eigvals):
    #        for j in range(N_sigma):
    #            boolinds_zero_imag_mj = np.abs(lambda_all.imag[m, :, j]) < tolerance_conjugation
    #            lambda_all[m, boolinds_zero_imag_mj, j] = lambda_all[m, boolinds_zero_imag_mj, j].real # i.e. neglect imag. part
    #
    # no return required. inplace change
    print('DONE: enforcing complex conjugated spectrum done')
    print('')



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




class SpectralSolver(object):
    
    def __init__(self, params):
        if params['verboselevel'] >= 1:
            print('constructing spectral solver object with params:')
            print(params)
        self.params = params
        
        if params['ode_method'] == 'magnus': # exponential integration -- recommended
            self.eigenflux_lb = self.eigenflux_lb_magnus
            self.adjoint_gamma = self.adjoint_gamma_magnus
            
        elif params['ode_method'] == 'richardson': # semi-exponential integration -- not recommended
            self.eigenflux_lb = self.eigenflux_lb_richardson
            self.adjoint_gamma = self.adjoint_gamma_richardson
            
        elif params['ode_method'] == 'lsoda': # general purpose (nonlinear) integration -- not recommended at all
            self.eigenflux_lb = self.eigenflux_lb_lsoda
            self.adjoint_gamma = self.adjoint_gamma_lsoda
        else:
            raise Exception('ODE method {0} not supported'.format(params['ode_method']))
        
        # only for odeint version:
        self.build_intfire_func()
        self.build_eigeneq_rhs() # wish for speedup: use nopython numba just in time compilation
        self.build_adjoint_eigeneq_rhs()
        
        self.ode_steps = []



# METHODS/FUNCTIONS:

# class SpectralSolver
# constructor(params_nondefault)

# eigenval_inits(params, lambdarange)

# eigenval_rect(params, init) -> invokes eigenval_curve IN PARALLEL => assembles CurveResult  objects to RectangleResult object
# class RectangleResult -- extend CurveResult???
# export to np+paramsdict

# eigenval_curve(params, init, id) -> invokes eigenval_point SEQUENTIALLY => returns CurveResult object
# class CurveResult
# constructor(params, init, id)
# export to np+paramsdict

# eigenval_point(params, init))
# criterion_eigenflux_lowerbound(params, eigenval_guess, ) -> diffeq_righthandside
# criterion_adjoint_reset_threshold(...) -> diffeq_adjoint_righthandside

# eigenfunc(params, eigenval, adjoint)

# NUMBA JIT FUNCTIONS
# diffeq_righthandside <= eigenvalue_guess=parameter dlambda/dt=0
# diffeq_adjoint_righthandside <= eigenvalue_guess=parameter  dlambda/dt=0


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
                             'f_1', 'f_2', 'psi_r_1', 'psi_r_2',
                             'c_mu_1', 'c_mu_2', 'c_sigma_1', 'c_sigma_2'
                            ], 
                N_procs=multiprocessing.cpu_count()):

        print('START: computing quantity rect: {}'.format(quant_names))

        lambda_1 = quantities_dict['lambda_1']        
        lambda_2 = quantities_dict['lambda_2']
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
                
            # complex quants
            elif q in ['f_1', 'f_2', 'psi_r_1', 'psi_r_2', 
                       'c_mu_1', 'c_mu_2', 'c_sigma_1', 'c_sigma_2']:
                quantities_dict[q] = np.zeros((N_mu, N_sigma)) + 0j # complex dtype
        
        arg_tuple_list = [(self.params, quant_names, lambda_1, lambda_2, mu_arr, sigma_arr, j)
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
                    
                    
                    # TODO: check whether we need this PASSAGE!  (commented out on 9th of March 16)
                    #if i>0 and lambda_init.imag != 0.0 and abs(lambda_init.imag) < (self.params['root_options']['eps'])**0.5:
#                    if i>0 and abs(lambda_init.imag) < (self.params['root_options']['eps'])**0.5:
#                        if self.params['verboselevel']>=1:
#                            print('hack (due to issue in minpack): adding imaginary perturbation to lambda_init'+
#                              ' to enforce reasonable finite diff step for jacobian (case 0 is already handled in minpack)')
#                        imag_amplitude = 0 if workaround_usage == 0 else k_refinement*self.params['solver_imag_pert_per_refinement_in_eps']
#                        signum = np.sign(lambda_init.imag) if lambda_init.imag != 0 else 1
#                        lambda_init = lambda_init.real + 1j*imag_amplitude*signum*np.sqrt(self.params['root_options']['eps']) # #1j instead of 5j or 2j #0j #1j*np.sign(lambda_init.imag)*np.sqrt(self.params['root_options']['eps'])
                    # commented out on 10th of March 16
                    
                    
                    # minpack's hybrid solver for nonlinear systems behaves badly when one 
                    # variable has values close to zero but not exactly zero. in our case this 
                    # is the imaginary part of the initial eigenvalue. therefore we ensure it is zero or 
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
                        
                        
                        #diag = np.array([1.0, k_refinement])
                        #print('changing diag to {0}'.format(diag))
                     
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
                    
                # old version:
#                if not converged_inner:                    
#                    print('REFINEMENT FAILED. GIVING UP. NO CONVERGENCE.')
#                    plt.figure()
#                    plt.suptitle('eigenvalue which did not converge, sigma={sigma}'.format(sigma=sigma_calc))
#                    plt.subplot(211)
#                    plt.title('real part')
#                    plt.plot(mu_sigma[:, 0], lambda_arr.real)
#                    plt.subplot(212)
#                    plt.title('imag part')
#                    plt.plot(mu_sigma[:, 0], lambda_arr.imag)
#                    plt.xlabel('mu')
#                    plt.show()
#                    warn(('convergence failed for mu={mu_calc}, sigma={sigma_calc}, '+
#                                     'eigen_ind={eigen_ind}').format(**locals()))
#                    return lambda_arr                                    
                    
            
                if mu_calc == mu or sigma_calc == sigma:
                    converged_outer = True
                    lambda_arr[i] = lambda_calc
                    
                    if converged_inner:
                        workaround_usage = 0 # reset workaround counter
                    
                    # to account for the exponential term in the reinjection condition if a refractory 
                    # period AND a lower bound is considered we have this cut off possibility that 
                    # just stops computing an eigenvalue curve (keeping the last converged one for the skipped mu,sigma pairs)
                    if 'lambda_real_abs_thresh' in self.params and abs(lambda_arr[i].real) > self.params['lambda_real_abs_thresh']:
                        warn('we reached the lowest possible lambda value')
                        lambda_arr[i:] = lambda_calc
                        return lambda_arr
                else:
                    warn('throwing away in-between mu sigma values and those eigenvalues as well')
                
                
                
        return lambda_arr
                
                
    #             
    #          
    #                 
    #     try:
    #                             
    #                         print('m={m}, mu={mu}, sigma={sigma}, lambda_init={lambda_init}'.format(**locals()))
    #                         lambda_m = eigenvalue(lambda_init, params)
    #                         
    #                         converged_inner = True
    #                         # check smoothness conditions and assume them to be fullfilled for i==0:
    #                         if i>0:
    #                             # check smoothness criterion: is the computed eigenvalue close enough to 
    #                             # the previous one, i.e. the initial approximation?
    #                             if abs((lambda_m - lastlambda)/lastlambda)>params['solver_smoothtol']:
    #                                 converged_inner = False
    #                                 print('SMOOTHNESS CRITERION (ABSOLUTE) FAILED')
    #                             
    #                             # detect unwished lambda curve hopping by computing a value close to mu_calc, sigma_calc 
    #                             # but backwards, i.e. in the direction of the previous mu and sigma values and check if 
    #                             # the linear lambda approximation deviates from lambda_init not too much
    #                             print('TODO: reenable curve hopping smoothness criterion')
    # #                             if converged_inner: # only proceed if smoothness criterion above did not fail
    # #                                 hopfraction = params['solver_hopfraction']
    # #                                 hoptol = params['solver_hoptol']
    # #                                 deltamu_backwards = (mu_calc - lastmu)*hopfraction
    # #                                 deltasigma_backwards = (sigma_calc - lastsigma)*hopfraction
    # #                                 params['mu'] -= deltamu_backwards  # assuming params['mu']==mu_calc
    # #                                 params['sigma'] -= deltasigma_backwards
    # #                                 lambda_backwards = complex(*eigenvalue(lambda_m, params))
    # #                                 lambda_backwards_initapprox = (lambda_m 
    # #                                                                - (lambda_m - lambda_backwards) / hopfraction)
    # #                                 params['mu'] = mu_calc
    # #                                 params['sigma'] = sigma_calc
    # #                                 if abs((lambda_backwards_initapprox - lambda_init)) > hoptol: #/lambda_init) > hoptol:
    # #                                     converged_inner = False
    # #                                     print('SMOOTHNESS CRITERION (CURVE HOPPING) FAILED')
    #                         
    #                         if converged_inner:
    #                             print('CONVERGED')
    #                         else:
    #                             print('NOT CONVERGED')
    #                             
    #                     except NotConverged as nc:
    #                         solver_result = nc.solver_result
    #                         print('NOT CONVERGED')
    #                         print(solver_result)
    #                         print(nc)
    #                     
    #                     if not converged_inner:
    #                         print('REFINEMENT NECESSARY')
    #                         k_refinement+=1
    #                         
    #                 if not converged_inner:
    #                     print('REFINEMENT FAILED. GIVING UP. NO CONVERGENCE.')
    #                     exit()
    #                 
    #                   
    #                 # update lastmu and lastsigma (last converged mu, sigma pair)
    #                 lastmu = mu_calc
    #                 lastsigma = sigma_calc
    #                 lastlambda = lambda_m
    #                 
    #                 
    #                 
    #                 
    #                 
    #                 if mu_calc == mu or sigma_calc == sigma:
    #                     converged_outer = True
    #                 else:
    #                     warn('throwing away in-between mu sigma values and those eigenvalues as well')
    #                 
    #         
    #     
    #     
    #     
    #     
    #     
    #     
    #     
    #         
    #         
    #         lambda_m_init = complex(lambda_real_init[m_ind]) # initialization based on the real zeros
    #         
    #         lastmu = None
    #         lastsigma = None
    #         lastlambda = None
    #         for i, (mu, sigma) in enumerate(zip(mu_vec, sigma_vec)):
    #             
    #             
    #             converged_outer = False # true if the m-th eigenvalue for mu, sigma has been found
    #             while not converged_outer:
    #             
    #                 converged_inner = False # true if the m-th eigenvalue for mu_calc, sigma_calc has been found
    #                 maxrefinements = 100
    #                 k_refinement = 0
    #                 while not converged_inner and k_refinement<maxrefinements:
    #                     if k_refinement==0:
    #                         mu_calc = mu
    #                         sigma_calc = sigma
    #                     else:
    #                         print('refinement no {k}'.format(k=k_refinement))
    #                         print('refinement necessary due to non convergence of:')
    #                         print('mu={mu}, sigma={sigma}'.format(mu=mu_calc, sigma=sigma_calc))
    #                         mu_calc = lastmu + (mu_calc-lastmu)/2.0
    #                         sigma_calc = lastsigma + (sigma_calc-lastsigma)/2.0
    #                         print('new pair: mu={mu}, sigma={sigma}'.format(mu=mu_calc, sigma=sigma_calc))
    #                         
    #                     params['mu'] = mu_calc
    #                     params['sigma'] = sigma_calc
    #                     
    #                     
    #                     # compute eigenvalue approximation as either linear or quadratic extrapolation based on 
    #                     # the last sucessfully computed eigenvalues
    #                     if i==0 or i==1: # also i=1 since we do not want to access mu, sigma values beyond the grid boundary
    #                         extrapolation = 'constant'
    #                     else:
    # #                         extrapolation = 'constant'
    #                         extrapolation = 'linear'
    #         #                     extrapolation = 'quadratic'
    #                         
    #                     if extrapolation == 'constant':
    #                         lambda_init = lambda_m
    #                     
    #                     elif extrapolation == 'linear':
    #                         eps_musigma = params['eps_solver_musigma_factor'] * params['root_options']['eps']
    #                         diffmu = mu_calc-lastmu
    #                         diffsigma = sigma_calc-lastsigma
    #                         norm_diff = np.sqrt(diffmu**2+diffsigma**2)
    #                         diffmu_normed = diffmu/norm_diff
    #                         diffsigma_normed = diffsigma/norm_diff
    #                         params['mu'] = lastmu - eps_musigma*diffmu_normed
    #                         params['sigma'] = lastsigma - eps_musigma*diffsigma_normed
    #                         print('initial approx before: mu={mu}, sigma={sigma}, lastmu={lm}, lastsigma={ls} diffmu_normed={dm}, diffsigma_normed={ds}'.format(
    #                                 mu=params['mu'], sigma=params['sigma'], dm=diffmu_normed, ds=diffsigma_normed, ls=lastsigma, lm=lastmu))
    #                         print('TODO: cache this number (has to be computed repeatedly if refinements are necessary (but attention to the scaling)...')
    #                         
    #                         # assume the following converges (sometimes not fullfilled...):
    #                         lambda_m_minus = eigenvalue(lastlambda, params)
    #                         lambda_m_minus_complex = complex(*lambda_m_minus)
    #                         params['mu'] = mu_calc
    #                         params['sigma'] = sigma_calc
    #                         lambda_init = lastlambda + (lastlambda-lambda_m_minus_complex) / eps_musigma * norm_diff
    #                         print('initial approx after: lambda_init={li} (lambda_minus={lm})'.
    #                               format(lm=lambda_m_minus_complex, li=lambda_init))
    #                              
    #                     elif extrapolation == 'quadratic':
    #                         raise(NotImplementedError('has to be derived from directional derivatives'))
    #                     
    #                     
    #                     
    #                     
    #                     
    #                     
    #     
    #     
    #     
    #     
    #     
    #     
    #     
    #     
    #     
    #     
    #     
    #     
    #     
    #     
    #     
    #     
    #     
    #     
    #     
    #     
    #     
    #     
    #     
    #     
    #     
    #     
    #     
    #     
    #     
    #     
    #     
    #     return lambda_arr
    
    
    # refines given mu_sigma (n x 2)-array such that the refined (m x 2)-array with m>=n 
    # has delta_min as maximum neighbor distance (2-norm)
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
        
            print('searching for real eigenvalues within lambda-interval [{left}, {right}]'.format(left=lambda_grid[0], 
                                                                                                  right=lambda_grid[-1]))
        
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
                            
        if 'grid_V_addpoints' in params:
            Vgrid_start = Vgrid[:-2]
            Vgrid_end_fine = np.linspace(Vgrid[-2], Vgrid[-1], params['grid_V_addpoints']+2)
            Vgrid = np.concatenate([Vgrid_start, Vgrid_end_fine])
            
        
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
    

    def eigenflux_lb_richardson(self, lambda_2d, eigFunc=False):
        
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

        eigeneq_backwards_richardson(init_q_r, phi, q, Vgrid, k_r, lambda_complex, mu, sigma, 
                                     hasLeak, C, g_L, E_L, hasExp, delta_T, V_T, tau_ref)

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
        

    # NOTE: this midpoint rule leads to order 2 of the magnus expansion; orders 4 and 6 are also easy        
    def eigenflux_lb_magnus(self, lambda_2d, eigFunc=False):
        
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
    
    
    # eigenflux_lb returns the two dimensional vector of (real and imag. part 
    # of) the complex flux q evaluated at V_lb as a function of complex 
    # lambda (which must also be represented as 2d array)
    # procedure: 
    # 1. backward integration from V_s to V_r
    # 2. flux reinjection
    # 3. if V_lb < V_r integration from V_r to V_lb
    # finally return the flux value at V_lb
    #
    # old_parameters:
    # lambda_2d:    the 2d array containing real and imag. part of lambda
    # params:            the old_parameters dictionary 
    def eigenflux_lb_lsoda(self, lambda_2d, eigFunc=False):
                
        params = self.params
        params['lambda'] = lambda_2d
        
        mu = params['mu']
        sigma = params['sigma']
        lambda_real = lambda_2d[0]
        lambda_imag = lambda_2d[1]
               
        
        # initialization as in Richardson 2007 and Ostojic 2011 
        # (complex version inspired by Srdjan Ostojic as in Schaffer 13)
        # as the eigenequation is linear, we can (together with the absorbing condition) scale
        # the flux at the spike boundary with an arbitrary complex constant
        # e.g. we can choose q(V_s)=init_q it to be (any) real nonzero number
        init_phi_r = 0
        init_phi_i = 0
        if 'init_real' in params:
            # use last successfull and normalized flux init to allow fast ODE integration (huge eigenfuncs if q_init_r=1)
            init_q_r = float(params['init_real'])
        else:
            init_q_r = 1.0
        init_q_i = 0
        
        atol=min(params['odeint_atol'], init_q_r*params['odeint_atol']) # set abs tolerance relative to init
         
        if params['verboselevel']>=2:
            print('mu={m}, sigma={s}, lambda={l}'.format(l=complex(*lambda_2d), 
                                                         m=params['mu'],
                                                         s=params['sigma']))
        
        # the following assertion enables to assume that the lower bound is never larger than the reset
        assert(params['V_lb'] <= params['V_r'])
        
        # earlier the actual computation of the eigenfunction depended on the parameter eigFunc
        # now we always compute the eigenfunc due to normalization but return the full function only 
        # on demand, i.e. if eigFunc is True
        computeEigFunc = True 
        # if we do not compute the eigenfunctions no fine integration grid is prescribed
        if not computeEigFunc:
            interval_right = [params['V_s'], params['V_r']]
            
            if params['V_lb'] < params['V_r']:
                interval_left = [params['V_r'], params['V_lb']]
        else: 
            grid_V_points_total = params['grid_V_points']
            phi_complex = np.zeros(grid_V_points_total, dtype=np.complex128)
    #         grid_full = np.zeros(grid_V_points_total)
            
            # ensure we equidistantly distribute the grid points onto the whole V grid
            # in the case of a lower bound smaller than the reset (i.e. two integration intervals)
            if params['V_lb'] == params['V_r']:
                grid_V_points_right = grid_V_points_total 
                interval_right = np.linspace(params['V_s'], params['V_r'], grid_V_points_total) # no left interval is needed
                grid_full = interval_right[::-1]
            else:
                fraction_grid_V_points_left = (params['V_r'] - params['V_lb']) / (params['V_s'] - params['V_lb'])
                grid_V_points_left = max(1, int(fraction_grid_V_points_left*grid_V_points_total))
                grid_V_points_right = grid_V_points_total - grid_V_points_left
                # in this (two interval) scenario the left interval gets +1 grid point due to double V_r
                interval_left = np.linspace(params['V_r'], params['V_lb'], grid_V_points_left+1)
                interval_right = np.linspace(params['V_s'], params['V_r'], grid_V_points_right)
                grid_full = np.concatenate([interval_right, interval_left[1:]])[::-1]
        
        X_right, ode_infodict_right = odeint(self.eigeneq_rhs,  #ode_rhs_wrapper, 
                       np.array([init_phi_r, init_phi_i, init_q_r, init_q_i, mu, sigma, lambda_real, lambda_imag]),
                       interval_right,
                       # Dfun=ode_jacobian,
                       #args=(params,),
                       mxstep=params['odeint_maxsteps'],
                       atol=atol,
                       rtol=params['odeint_rtol'], 
                       full_output=True) # maybe turn off for production version
        
        ode_steps = ode_infodict_right['nst'][0]        
        if params['verboselevel']>=3:
            print(' ODE(right) backward integration took {0} V steps'.format(ode_infodict_right['nst']))
        
        
        # reinjection of the spike threshold flux at the reset
        # refractory period handling taken from Srdjan Ostojic's code (based on Richardson 2007)
        # i.e. q(V_r) -= exp(lambda*tau_ref) (Ostojic had tau_m seperated out from the model eqs.) 
        q_update_complex = init_q_r*np.exp(-complex(*lambda_2d)*params['tau_ref'])
        if params['verboselevel'] >= 3:
            print('q_update_complex={0}'.format(q_update_complex))
        # q_r (real part of the flux q) reinjection at V_r
        X_right[-1, 2] -= q_update_complex.real
        # q_i (imag. part of q) reinjection at V_r
        X_right[-1, 3] -= q_update_complex.imag
         
        if computeEigFunc:
            phi_complex[-grid_V_points_right:].real = X_right[::-1, 0]
            phi_complex[-grid_V_points_right:].imag = X_right[::-1, 1]
            
        # integrate further if lower bound is smaller than reset (else integration 
        if params['V_lb'] < params['V_r']:
                
            X_left, ode_infodict_left = odeint(self.eigeneq_rhs, #ode_rhs_wrapper, 
                            X_right[-1, :],
                            interval_left,
                            # Dfun=ode_jacobian,
                            #args=(params,),
                            mxstep=params['odeint_maxsteps'],
                            atol=atol,
                            rtol=params['odeint_rtol'], 
                            full_output=True) # maybe turn off for production version)
            
            if params['verboselevel']>=3:
                print(' ODE(left) backward integration took {0} V steps'.format(ode_infodict_left['nst']))
                #print('[phi_r, phi_i, q_r, q_i] = {Xl}'.format(Xl=X_left))
            
            ode_steps += ode_infodict_right['nst'][0]
            
            
            if computeEigFunc:
                phi_complex[:grid_V_points_left].real = X_left[:0:-1, 0] # do not double the value at V_r 
                phi_complex[:grid_V_points_left].imag = X_left[:0:-1, 1]
            
        # if lower bound is equal the reset (or larger which should not happen) finish
        else:
            X_left = X_right
        
        self.ode_steps.append(ode_steps)
        
#         self.phi_q_lb = X_left[-1, :4] # for abs tol check toggle
        
        
        # return normalized eigenfunc and calculate init flux
        if computeEigFunc:
#        # REQUIRED FOR REASONABLY FAST COMPUTATION (abs tol of odeint... for sigma small)
#            phi_norm = np.sum(np.abs(phi_complex[:-1]) * np.diff(grid_full))
#            params['init_q_r_unchecked'] = init_q_r/phi_norm
#            phi_complex /= phi_norm
            # return normalized eigenfunc and calculate init flux
            phi_norm = normalize(phi_complex, grid_full)
            self.params['init_real_last'] = init_q_r/phi_norm
        
        q_lb_2d = X_left[-1, 2:4]/phi_norm
        
        
        if params['verboselevel']>=2:
            print('flux_lb = {f}'.format(f=q_lb_2d))
        
        if not eigFunc:
            # return the real and imag. part of the flux q evaluated at V_lb
            return q_lb_2d
        else:
            return (q_lb_2d, grid_full, phi_complex, q) 

    def adjoint_gamma_magnus(self, lambda_2d, eigFunc=False):
        
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

    def adjoint_gamma_richardson(self, lambda_2d, eigFunc=False):
        
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

        adjoint_eigeneq_forwards_richardson(init_psi_r, psi, dpsi, Vgrid, lambda_complex, 
                                            mu, sigma, hasLeak, C, g_L, E_L, hasExp, delta_T, V_T)

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


    # characteristic eq. for the adjoint operator (b.c. at reset & threshold)
    def adjoint_gamma_richardson_old(self, lambda_2d, eigFunc=False): 
        
        params = self.params
#        params['lambda'] = lambda_2d
        
        mu = float(params['mu'])
        sigma = float(params['sigma'])
        lambda_complex = complex(*lambda_2d)
        # non-uniform grids are allowed
        Vgrid = np.linspace(float(params['V_lb']), float(params['V_s']), 
                            params['grid_V_points'])

        Nv = len(Vgrid)
        psi = np.zeros((Nv), dtype=np.complex128)
        dpsi = np.zeros_like(psi)
        
        
                    
        print('richardson adjoint with init_psi_r = {}'.format(init_psi_r))
        
        # neuronal params:
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

        adjoint_eigeneq_forwards_richardson(init_psi_r, psi, dpsi, Vgrid, lambda_complex, 
                                             mu, sigma, hasLeak, C, g_L, E_L, 
                                             hasExp, delta_T, V_T)
                                             
          
#        if mu>-1.5 and ('plotted' not in self.params):               
#            plt.figure()
#            plt.subplot(211)
#            plt.plot(Vgrid, psi.real)
#            plt.subplot(212)
#            plt.plot(Vgrid, psi.imag)
#            plt.show()
#            self.params['plotted'] = True

        # TODO: INCLUDE NORMALIZATION INTO ADJOINT.
        # problems: no convergence to eigenvalues in eigenvalue_curve with normalization
        #psi_norm = normalize(psi, Vgrid) # only psi not dpsi is used in the following
        psi_norm = 1.0
        self.params['init_real_last'] = init_psi_r/psi_norm
        
        k_r = np.argmin(np.abs(Vgrid - params['V_r']))        
        gamma_complex = psi[Nv-1] - psi[k_r] * cmath.exp( -lambda_complex*tau_ref)
        
        gamma_2d = [gamma_complex.real, gamma_complex.imag]
        
        if not eigFunc:
            # return the real and imag. part of the flux q evaluated at V_lb
            return gamma_2d
        else:
            return (gamma_2d, Vgrid, psi, dpsi) 
    
    
    def adjoint_gamma_lsoda(self, lambda_2d, eigFunc=False): 
        
        params = self.params
        
        params['lambda'] = lambda_2d
        
        mu = params['mu']
        sigma = params['sigma']
        lambda_real = lambda_2d[0]
        lambda_imag = lambda_2d[1]
        
        # initialization (see notes)
#        init_psi_r = 1 # arbitrary due to linearity
        if 'init_real' in params:
            # use last successfull and normalized flux init to allow fast ODE integration (huge eigenfuncs if q_init_r=1)
            init_psi_r = float(params['init_real'])
        else:
            init_psi_r = 1.0
        init_psi_i = 0 # arbitrary due to linearity
        init_dpsi_r = 0 # due to b.c.
        init_dpsi_i = 0 # due to b.c.
        
        if params['verboselevel']>=2:
            print('mu={m}, sigma={s}, lambda={l}'.format(l=complex(*lambda_2d), 
                                                         m=params['mu'],
                                                         s=params['sigma']))
        
        # the following assertion enables to assume that the lower bound is never larger than the reset
        assert(params['V_lb'] <= params['V_r'])
          
        # although we could integrate the whole interval V_lb to V_s at once (no discontinuity) for 
        # the adjoint operator, we use the same splitting of the intervals to have the same grid points 
        # for the eigenfunction as for the non-adjoint operator 
        
        
        # earlier the actual computation of the eigenfunction depended on the parameter eigFunc
        # now we always compute the eigenfunc due to normalization but return the full function only 
        # on demand, i.e. if eigFunc is True
        computeEigFunc = True 
        
        if not computeEigFunc:
            interval_right = [params['V_r'], params['V_s']]
            
            if params['V_lb'] < params['V_r']:
                interval_left = [params['V_lb'], params['V_r']]
        else: 
            grid_V_points_total = params['grid_V_points']
            psi_complex = np.zeros(grid_V_points_total, dtype=np.complex128)
            dpsi_complex = np.zeros_like(psi_complex)
    #         grid_full = np.zeros(grid_V_points_total)
            
            # ensure we equidistantly distribute the grid points onto the whole V grid
            # in the case of a lower bound smaller than the reset (i.e. two integration intervals)
            if params['V_lb'] == params['V_r']:
                grid_V_points_right = grid_V_points_total 
                interval_right = np.linspace(params['V_r'], params['V_s'], grid_V_points_total) # no left interval is needed
                grid_full = interval_right
            else:
                fraction_grid_V_points_left = (params['V_r'] - params['V_lb']) / (params['V_s'] - params['V_lb'])
                grid_V_points_left = max(1, int(fraction_grid_V_points_left*grid_V_points_total))
                grid_V_points_right = grid_V_points_total - grid_V_points_left
                # in this (two interval) scenario the left interval gets +1 grid point due to double V_r
                interval_left = np.linspace(params['V_lb'], params['V_r'], grid_V_points_left+1)
                interval_right = np.linspace(params['V_r'], params['V_s'], grid_V_points_right)
                grid_full = np.concatenate([interval_left, interval_right[1:]])
        
        init_cond = np.array([init_psi_r, init_psi_i, init_dpsi_r, init_dpsi_i, mu, sigma, lambda_real, lambda_imag])
        
        if params['V_lb'] < params['V_r']:
        
            X_left = odeint(self.adjoint_eigeneq_rhs, 
                            init_cond,
                            interval_left, 
                            #args=(params,),
                            mxstep=params['odeint_maxsteps'],
                            atol=params['odeint_atol'],
                            rtol=params['odeint_rtol'])
            
            init_cond = X_left[-1, :] # update init cond for the right interval
            
            # adjoint: no reinjection of the spike threshold flux at the reset
            # refractory period handling is in rhs boundary condition
            if computeEigFunc:
                psi_complex[:grid_V_points_left].real = X_left[:-1, 0] # do not double the value at V_r
                psi_complex[:grid_V_points_left].imag = X_left[:-1, 1]
                dpsi_complex[:grid_V_points_left].real = X_left[:-1, 2]
                dpsi_complex[:grid_V_points_left].imag = X_left[:-1, 3]
                
        X_right = odeint(self.adjoint_eigeneq_rhs, 
                         init_cond,
                         interval_right, 
                         #args=(params,),
                         mxstep=params['odeint_maxsteps'],
                         atol=params['odeint_atol'],
                         rtol=params['odeint_rtol'])
        
        if computeEigFunc:
                psi_complex[-grid_V_points_right:].real = X_right[:, 0]  
                psi_complex[-grid_V_points_right:].imag = X_right[:, 1]
                dpsi_complex[-grid_V_points_right:].real = X_left[:, 2]
                dpsi_complex[-grid_V_points_right:].imag = X_left[:, 3]
                
        # for the non-adjoint operator: q_lb=0 determines whether we have foundan eigenvalue
        # here for the adjoint operator: gamma = psi(V_s) - psi(V_r)*exp(-lambda*tau_ref) 
                
        
        if computeEigFunc:
#        # REQUIRED FOR REASONABLY FAST COMPUTATION (abs tol of odeint... for sigma small)
#            phi_norm = np.sum(np.abs(phi_complex[:-1]) * np.diff(grid_full))
#            params['init_q_r_unchecked'] = init_q_r/phi_norm
#            phi_complex /= phi_norm
            # return normalized eigenfunc and calculate init flux
            # TODO: INCLUDE NORMALIZATION INTO ADJOINT.
            # problems: no convergence to eigenvalues in eigenvalue_curve with normalization
            #psi_norm = normalize(psi_complex, grid_full) # only psi not dpsi is used in the following
            psi_norm = 1.0
            self.params['init_real_last'] = init_psi_r/psi_norm
        
        
        psi_r_Vs = X_right[-1, 0]
        psi_i_Vs = X_right[-1, 1]
        psi_r_Vr = init_cond[0]
        psi_i_Vr = init_cond[1]
        gamma_complex = complex(psi_r_Vs, psi_i_Vs) - complex(psi_r_Vr, psi_i_Vr) * np.exp(-complex(*lambda_2d)*params['tau_ref'])

        # divide psi fully by psi_norm... not only output, also boundary cond...                      
        gamma_2d = np.array([gamma_complex.real, 
                             gamma_complex.imag]) / psi_norm
        
        if not eigFunc:
            # return the real and imag. part of gamma
            return gamma_2d
        else:
            return (gamma_2d, grid_full, psi_complex, dpsi_complex)

    # eigeneq_rhs returns a function for the the right hand side of 
    # the ODE system which is equivalent to the linear complex 
    # eigenfunction equation for the FP operator 
    # 1) 1d 2nd order -> 2d 1st order (as in Richardson 2007)
    # 2) complex -> 2d real
    # => in total 4 real variables which are collected in the vector X 
    # old_parameters:
    # X = (phi_r, phi_i, q_r, q_i)
    # V: (the integration variable; membrane voltage)
    # p: old_parameters dictionary
    def build_eigeneq_rhs(self):
        
        intfire_func = self.intfire_func
        
        #@jit(nopython=True)
        @njit        
        def eigeneq_rhs(X, V):
            
            rhs = np.zeros(8)
    
                
            # rename the input
            phi_r = X[0]
            phi_i = X[1]
            q_r = X[2]
            q_i = X[3]
            mu = X[4]
            sigma = X[5]
            lambda_r = X[6]
            lambda_i = X[7]
            
            # abbreviation
            fV = intfire_func(V)
            sig2h = sigma**2/2.0
            
            # right hand side of the (forward) ODE (the backwards direction is taken care of by the solver)
            d_phi_r = (fV+mu)/sig2h * phi_r   -   1/sig2h * q_r
            d_phi_i = (fV+mu)/sig2h * phi_i   -   1/sig2h * q_i
            d_q_r = -lambda_r * phi_r   +   lambda_i * phi_i
            d_q_i = -lambda_i * phi_r   -   lambda_r * phi_i
            
            # output
            rhs[0] = d_phi_r
            rhs[1] = d_phi_i
            rhs[2] = d_q_r
            rhs[3] = d_q_i
            rhs[4:8] = 0.0
               
            return rhs
        
        # numba currently does not support array creation in nopython mode
        # and does not speed up in object mode
        self.eigeneq_rhs = eigeneq_rhs
    
    # intfire_func returns a function for the right hand side (f) of 
    # the integrate and fire model ODE excluding the synaptic moments
    # e.g. for the LIF model f(V)=g_L/C*(E_L-V) 
    def build_intfire_func(self):
    
        params = self.params
        model = params['model']
        
        if model == 'PIF':
            
            #@jit(nopython=True)
            @njit            
            def intfire_func_optimized(V):
                return 0.0
        
        elif model == 'LIF':
            g_L = params['g_L']
            C = params['C']
            E_L = params['E_L']
            
            #@jit(nopython=True)
            @njit            
            def intfire_func_optimized(V):
                return g_L/C * (E_L-V)
        
        elif model == 'EIF':
            g_L = params['g_L']
            C = params['C']
            E_L = params['E_L']
            delta_T = params['delta_T']
            V_T = params['V_T']
            
            #@jit(nopython=True)
            @njit            
            def intfire_func_optimized(V):
                return g_L/C * ((E_L-V) + delta_T * math.exp((V-V_T)/delta_T))
        
        else: # unknown model
            raise Exception('Model {0} not supported'.format(model))
        
        self.intfire_func = intfire_func_optimized #njit(intfire_func_optimized) 


    def build_adjoint_eigeneq_rhs(self):
        
        intfire_func = self.intfire_func
        
        #@jit(nopython=True)
        @njit        
        def adjoint_eigeneq_rhs(X, V):
            
            rhs = np.zeros(8)
    
            # rename the input
            psi_r = X[0]
            psi_i = X[1]
            dpsi_r = X[2]
            dpsi_i = X[3]
            mu = X[4]
            sigma = X[5]
            lambda_r = X[6]
            lambda_i = X[7]
            
            # abbreviations
            fV = intfire_func(V)
            sig2h = sigma**2/2.0
            
            # right hand side of the ODE (adjoint operator -> 1st order 2d system -> real 4d system)
            d_psi_r = dpsi_r
            d_psi_i = dpsi_i
            d_dpsi_r = -(fV+mu)/sig2h * dpsi_r   +   (lambda_r*psi_r - lambda_i*psi_i)/sig2h
            d_dpsi_i = -(fV+mu)/sig2h * dpsi_i   +   (lambda_r*psi_i + lambda_i*psi_r)/sig2h
        
            # output
            rhs[0] = d_psi_r
            rhs[1] = d_psi_i
            rhs[2] = d_dpsi_r
            rhs[3] = d_dpsi_i
            rhs[4:8] = 0.0
               
            return rhs
        
        # numba currently does not support array creation in nopython mode
        # and does not speed up in object mode
        self.adjoint_eigeneq_rhs = adjoint_eigeneq_rhs 

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

# numba-jitted function for richardson's threshold integration method
@njit
def eigeneq_backwards_richardson(init_q_r, phi, q, Vgrid, k_r, lambda_complex, mu, sigma, 
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
    
    for k in range(Nv-2, -1, -1): # k= Nv-2 , ... , 0
    
        # grid spacing, allowing non-uniform grids
        dV = Vgrid[k+1]-Vgrid[k]
        
        # the flux variable q will be integrated by Euler's method         
        q[k] = q[k+1] + dV * lambda_complex * phi[k+1]
        
        # the eigenfunction variable phi will be integrated using an 
        # approximation of the full variation of parameters formula 
        # in the backwards direction (Richardson's threshold integration)
        fV = intfire_rhs(Vgrid[k+1], hasLeak, C, g_L, E_L, hasExp, delta_T, V_T)
        G = -(fV + mu) / sig2h
        H = q[k+1] / sig2h
        exp_dV_G = math.exp(dV*G)
        phi[k] = phi[k+1] * exp_dV_G  +  H/G*(exp_dV_G - 1)
        
        # reinjection condition
        if k == k_r:
            q[k] = q[k] - q[Nv-1]*cmath.exp(-lambda_complex*tau_ref)

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
def adjoint_eigeneq_forwards_richardson(init_psi_r, psi, dpsi, Vgrid, lambda_complex, 
                                         mu, sigma, hasLeak, C, g_L, E_L, 
                                         hasExp, delta_T, V_T):
                                             
                                             
    Nv = len(Vgrid)
    psi[0] = init_psi_r + 0.0j # linear eq = arbitrary complex scaling
    dpsi[0] = 0.0 + 0.0j # left boundary condition of adjoint operator
    
    # abbreviation
    sig2h = sigma**2/2.0
    for k in range(1, Nv): # k= 1 , ... , Nv-1
        # grid spacing, allowing non-uniform grids
        dV = Vgrid[k]-Vgrid[k-1]     
        
        # the variable psi will be integrated by Euler's method         
        psi[k] = psi[k-1] + dV * dpsi[k-1]

        # the variable dpsi will be integrated using an 
        # approximation of the full variation of parameters formula 
        # in the forwards direction (extension to Richardson's threshold int.)
        fV = intfire_rhs(Vgrid[k-1], hasLeak, C, g_L, E_L, hasExp, delta_T, V_T)  
        R = -(fV + mu) / sig2h # = G due to backwards->forwards _and_ adjoint
        S = lambda_complex / sig2h * psi[k-1]
        exp_dV_R = math.exp(dV*R)
        dpsi[k] = dpsi[k-1] * exp_dV_R  +  S/R * (exp_dV_R - 1)
        
        # that's it. no discontinuous reinjection => instead in characteristic eq.
        
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


        
    
# some helper functions
def warn(*s):
    print('warning: {}'.format(s))
    
def error(*s):
    print('error: {}'.format(s))
    exit()
    
    
    
    
    
    
    
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
        



