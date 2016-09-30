# -*- coding: utf-8 -*-

import numpy as np
import scipy.integrate
import scipy.optimize
import numba
import multiprocessing
#import itertools
import time
import tables
from warnings import warn
import matplotlib.pyplot as plt
import matplotlib.colors as colors

def EIF_steadystate_and_linresponse(mu_vals, sigma_vals, params, EIF_output_dict, output_names):
    
    print('Computing: {}'.format(output_names))
    
    N_mu_vals = len(mu_vals)    
    N_sigma_vals = len(sigma_vals)
    total = len(mu_vals)*len(sigma_vals)
    if total<=params['N_procs']:
        N_procs = 1
    else:
        N_procs = params['N_procs']
        
    # zero EIF_output_dict arrays
    for n in output_names:  
        # complex values dependent on mu, sigma, frequency
        if n in ['r1_mumod', 'r1_sigmamod']:
            EIF_output_dict[n] = np.zeros((N_mu_vals, N_sigma_vals, len(params['freq_vals']))) + 0j
            
        # complex values dependent on mu, sigma
        elif n in ['peak_real_r1_mumod', 'peak_imag_r1_mumod', 
                   'peak_real_r1_sigmamod', 'peak_imag_r1_sigmamod']:
            EIF_output_dict[n] = np.zeros((N_mu_vals, N_sigma_vals)) + 0j
            
        # real values dependent on mu, sigma
        else: 
            EIF_output_dict[n] = np.zeros((N_mu_vals, N_sigma_vals))
                
    arg_tuple_list = [(imu, mu_vals[imu], isig, sigma_vals[isig], params, output_names) 
                      for imu in range(N_mu_vals) for isig in range(N_sigma_vals)]    
                      
    comp_total_start = time.time()
                  
    if N_procs <= 1:
        # single processing version, i.e. loop
        pool = False
        result = (EIF_ss_and_linresp_given_musigma_wrapper(arg_tuple) 
                  for arg_tuple in arg_tuple_list) 
    else:
        # multiproc version
        pool = multiprocessing.Pool(params['N_procs'])
        result = pool.imap_unordered(EIF_ss_and_linresp_given_musigma_wrapper, arg_tuple_list)
        
    finished = 0 
    for imu, isig, res_given_musigma_dict in result:
        finished += 1
        print(('{count} of {tot} EIF steady-state / rate response calculations completed').
              format(count=finished, tot=total)) 
        for k in res_given_musigma_dict.keys():
            if k in ['r1_mumod', 'r1_sigmamod']:
                EIF_output_dict[k][imu,isig,:] = res_given_musigma_dict[k]
            else:
                EIF_output_dict[k][imu,isig] = res_given_musigma_dict[k]    
    
    # also include mu_vals, sigma_vals, and freq_vals in output dictionary
    EIF_output_dict['mu_vals'] = mu_vals
    EIF_output_dict['sigma_vals'] = sigma_vals
    EIF_output_dict['freq_vals'] = params['freq_vals'].copy()
    
    print('Computation of: {} done'.format(output_names))
    print('Total time for computation (N_mu_vals={Nmu}, N_sigma_vals={Nsig}): {rt}s'.
          format(rt=np.round(time.time()-comp_total_start,2), Nmu=N_mu_vals, Nsig=N_sigma_vals))
      
    return EIF_output_dict
    
    
def calc_EIF_output_and_cascade_quants(mu_vals, sigma_vals, params, 
                                       EIF_output_dict, output_names, save_rate_mod,
                                       LN_quantities_dict, quantity_names):
    
    print('Computing: {}'.format(output_names))
    
    N_mu_vals = len(mu_vals)    
    N_sigma_vals = len(sigma_vals)
    if N_sigma_vals<=params['N_procs']:
        N_procs = 1
    else:
        N_procs = params['N_procs']
        
    # zero EIF_output_dict arrays
    for n in output_names:  
        # complex values dependent on mu, sigma, frequency
        if n in ['r1_mumod', 'r1_sigmamod'] and save_rate_mod:
            EIF_output_dict[n] = np.zeros((N_mu_vals, N_sigma_vals, len(params['freq_vals']))) + 0j
            
        # complex values dependent on mu, sigma
        elif n in ['peak_real_r1_mumod', 'peak_imag_r1_mumod', 
                   'peak_real_r1_sigmamod', 'peak_imag_r1_sigmamod']:
            EIF_output_dict[n] = np.zeros((N_mu_vals, N_sigma_vals)) + 0j
            
        # real values dependent on mu, sigma
        else: 
            EIF_output_dict[n] = np.zeros((N_mu_vals, N_sigma_vals))
    
    # zero quantities_dict arrays
    for n in quantity_names:  
        # real values dependent on mu, sigma
        LN_quantities_dict[n] = np.zeros((N_mu_vals, N_sigma_vals))
                

                
    arg_tuple_list = [(isig, sigma_vals[isig], mu_vals, params, output_names,
                       quantity_names, save_rate_mod) for isig in range(N_sigma_vals)]    
                      
    comp_total_start = time.time()
                  
    if N_procs <= 1:
        # single processing version, i.e. loop
        pool = False
        result = (output_and_quantities_given_sigma_wrapper(arg_tuple) 
                  for arg_tuple in arg_tuple_list) 
    else:
        # multiproc version
        pool = multiprocessing.Pool(params['N_procs'])
        result = pool.imap_unordered(output_and_quantities_given_sigma_wrapper, arg_tuple_list)
        
    finished = 0 
    for isig, res_given_sigma_dict in result:
        finished += 1
        print(('{count} of {tot} EIF steady-state / rate response and LN quantity calculations completed').
              format(count=finished*N_mu_vals, tot=N_mu_vals*N_sigma_vals)) 
        for k in res_given_sigma_dict.keys():
            for imu, mu in enumerate(mu_vals):
                if k in ['r1_mumod', 'r1_sigmamod'] and save_rate_mod:
                    EIF_output_dict[k][imu,isig,:] = res_given_sigma_dict[k][imu,:]
                elif k in output_names and k not in ['r1_mumod', 'r1_sigmamod']:
                    EIF_output_dict[k][imu,isig] = res_given_sigma_dict[k][imu] 
                if k in quantity_names:
                    LN_quantities_dict[k][imu,isig] = res_given_sigma_dict[k][imu]    
    
    # also include mu_vals, sigma_vals, and freq_vals in output dictionaries
    EIF_output_dict['mu_vals'] = mu_vals
    EIF_output_dict['sigma_vals'] = sigma_vals
    EIF_output_dict['freq_vals'] = params['freq_vals'].copy()

    LN_quantities_dict['mu_vals'] = mu_vals
    LN_quantities_dict['sigma_vals'] = sigma_vals
    LN_quantities_dict['freq_vals'] = params['freq_vals'].copy()
    
    print('Computation of: {} done'.format(output_names))
    print('Total time for computation (N_mu_vals={Nmu}, N_sigma_vals={Nsig}): {rt}s'.
          format(rt=np.round(time.time()-comp_total_start,2), Nmu=N_mu_vals, Nsig=N_sigma_vals))
      
    return EIF_output_dict, LN_quantities_dict
    

def output_and_quantities_given_sigma_wrapper(arg_tuple):
    isig, sigma, mu_vals, params, output_names, quantity_names, save_rate_mod = arg_tuple
    V_vec = params['V_vals']
    Vr = params['Vr']
    kr = np.argmin(np.abs(V_vec-Vr))  # reset index value
    VT = params['VT']
    taum = params['taum']
    EL = params['EL']
    DeltaT = params['deltaT']
    
    dV = V_vec[1]-V_vec[0]
    Tref = params['t_ref']
    
    # for filter fitting (below)
    f = params['freq_vals']
    # for dosc fitting
    init_vals = [10.0, 0.01]  # tau (ms), f0 (kHz)
        
    N_mu_vals = len(mu_vals)    
    res_given_sigma_dict = dict()
    
    for n in output_names:  
        # complex values dependent on mu, sigma, frequency
        if n in ['r1_mumod', 'r1_sigmamod'] and save_rate_mod:
            res_given_sigma_dict[n] = np.zeros((N_mu_vals, len(params['freq_vals']))) + 0j
            
        # complex values dependent on mu, sigma
        elif n in ['peak_real_r1_mumod', 'peak_imag_r1_mumod', 
                   'peak_real_r1_sigmamod', 'peak_imag_r1_sigmamod']:
            res_given_sigma_dict[n] = np.zeros(N_mu_vals) + 0j
            
        # real values dependent on mu, sigma
        else: 
            res_given_sigma_dict[n] = np.zeros(N_mu_vals)
     
    for n in quantity_names:    
        if n not in ['r_ss', 'V_mean_ss']:  # omit doubling
            res_given_sigma_dict[n] = np.zeros(N_mu_vals)
            
    
    for imu, mu in enumerate(mu_vals):    
        # steady state output:
        p_ss, r_ss, q_ss = EIF_steady_state(V_vec, kr, taum, EL, Vr, VT, DeltaT, mu, sigma) 
        _, r_ss_dmu, _ = EIF_steady_state(V_vec, kr, taum, EL, Vr, VT, DeltaT, 
                                          mu+params['d_mu'], sigma)
        _, r_ss_dsig, _ = EIF_steady_state(V_vec, kr, taum, EL, Vr, VT, DeltaT, 
                                           mu, sigma+params['d_sigma'])
           
        if 'V_mean_ss' in output_names:
            # disregarding Tref -> use for Vmean when clamping both, V and w:
            V_mean = dV*np.sum(V_vec*p_ss)  
            res_given_sigma_dict['V_mean_ss'][imu] = V_mean
        
        r_ss_ref = r_ss/(1+r_ss*Tref)    
        p_ss = r_ss_ref * p_ss/r_ss
        q_ss = r_ss_ref * q_ss/r_ss  # prob. flux needed for sigma-mod calculation              
        r_ss_dmu_ref = r_ss_dmu/(1+r_ss_dmu*Tref) - r_ss_ref
        r_ss_dsig_ref = r_ss_dsig/(1+r_ss_dsig*Tref) - r_ss_ref   
        
        if 'V_mean_sps_ss' in output_names:
            # when considering spike shape (during refr. period):
            # density reflecting nonrefr. proportion, which integrates to r_ss_ref/r_ss 
            Vmean_sps = dV*np.sum(V_vec*p_ss) + (1-r_ss_ref/r_ss)*(params['Vcut']+Vr)/2  
            # note: (1-r_ss_ref/r_ss)==r_ss_ref*Tref 
            res_given_sigma_dict['V_mean_sps_ss'][imu] = Vmean_sps
            
        if 'r_ss' in output_names:
            res_given_sigma_dict['r_ss'][imu] = r_ss_ref
            
        if 'dr_ss_dmu' in output_names:      
            dr_ss_dmu = r_ss_dmu_ref/params['d_mu']
            res_given_sigma_dict['dr_ss_dmu'][imu] = dr_ss_dmu
            
        if 'dr_ss_dsigma' in output_names:        
            dr_ss_dsig = r_ss_dsig_ref/params['d_sigma']
            res_given_sigma_dict['dr_ss_dsigma'][imu] = dr_ss_dsig
        
            
        # next, mu-mod and sigma-mod over freq. range, and optionally, 
        # binary search to determine accurate peaks of |r1mu_vec|, 
        # and of |real(r1mu_vec)|, and |imag(r1mu_vec)| WITHIN range 
        # [0, params['freq_vals'][-1]], assuming a uniform freq_vals range 
        if 'r1_mumod' in output_names or 'peak_real_r1_mumod' in output_names \
        or 'peak_imag_r1_mumod' in output_names:
            w_vec = 2*np.pi*params['freq_vals']
            # mu1 = 1e-4
            # mu1 = params['d_mu'], consistently with use of r_ss_dmu_ref below
            inhom = params['d_mu']*p_ss
            r1mu_vec = EIF_lin_rate_response_frange(V_vec, kr, taum, EL, Vr, VT, DeltaT, 
                                                    Tref, mu, sigma, inhom, w_vec)
            #r1mu_vec /= mu1
            if save_rate_mod:                                        
                res_given_sigma_dict['r1_mumod'][imu,:] = r1mu_vec/params['d_mu']
            
            if 'peak_real_r1_mumod' in output_names:
                abs_re_im = 'real'    
                w_peak, peak_val = EIF_find_lin_response_peak(w_vec,r1mu_vec,r_ss_dmu_ref,
                                                          V_vec,kr,taum,EL,Vr,VT,DeltaT,
                                                          Tref, mu,sigma,inhom,abs_re_im)
                res_given_sigma_dict['peak_real_r1_mumod'][imu] = peak_val/params['d_mu']
                res_given_sigma_dict['f_peak_real_r1_mumod'][imu] = w_peak/(2*np.pi)
                                              
            if 'peak_imag_r1_mumod' in output_names:                                              
                abs_re_im = 'imag'    
                w_peak, peak_val = EIF_find_lin_response_peak(w_vec,r1mu_vec,r_ss_dmu_ref,
                                                          V_vec,kr,taum,EL,Vr,VT,DeltaT,
                                                          Tref, mu,sigma,inhom,abs_re_im)
                res_given_sigma_dict['peak_imag_r1_mumod'][imu] = peak_val/params['d_mu']
                res_given_sigma_dict['f_peak_imag_r1_mumod'][imu] = w_peak/(2*np.pi)                                          
       
        if 'r1_sigmamod' in output_names or 'peak_real_r1_sigmamod' in output_names \
        or 'peak_imag_r1_sigmamod' in output_names:
            w_vec = 2*np.pi*params['freq_vals']
            # sigma1 = 1e-4;  #inhom for sigma^2 modulation = -sigma21*dp_ssdV/2  
            # sigma1 = params['d_sigma'], consistently with use of r_ss_dsigma_ref (not implemented a.t.m.)
            driftterm = ( mu + ( EL-V_vec+DeltaT*np.exp((V_vec-VT)/DeltaT) )/taum ) * p_ss
            inhom = params['d_sigma'] * 2/sigma * (q_ss - driftterm)  # inhom = -sigma*sigma1*dp_ssdV
            r1sig_vec = EIF_lin_rate_response_frange(V_vec, kr, taum, EL, Vr, VT, DeltaT, 
                                                     Tref, mu, sigma, inhom, w_vec)
            #r1sig_vec /= sigma1
            if save_rate_mod:                                         
                res_given_sigma_dict['r1_sigmamod'][imu,:] = r1sig_vec/params['d_sigma']   
            
            if 'peak_real_r1_sigmamod' in output_names:
                abs_re_im = 'real'    
                w_peak, peak_val = EIF_find_lin_response_peak(w_vec,r1sig_vec,r_ss_dsig_ref,
                                                          V_vec,kr,taum,EL,Vr,VT,DeltaT,
                                                          Tref, mu,sigma,inhom,abs_re_im)
                res_given_sigma_dict['peak_real_r1_sigmamod'][imu] = peak_val/params['d_sigma']
                res_given_sigma_dict['f_peak_real_r1_sigmamod'][imu] = w_peak/(2*np.pi)
                                              
            if 'peak_imag_r1_sigmamod' in output_names:                                              
                abs_re_im = 'imag'    
                w_peak, peak_val = EIF_find_lin_response_peak(w_vec,r1sig_vec,r_ss_dsig_ref,
                                                          V_vec,kr,taum,EL,Vr,VT,DeltaT,
                                                          Tref, mu,sigma,inhom,abs_re_im)
                res_given_sigma_dict['peak_imag_r1_sigmamod'][imu] = peak_val/params['d_sigma']
                res_given_sigma_dict['f_peak_imag_r1_sigmamod'][imu] = w_peak/(2*np.pi)
        
        # now the quantities (fitting the filters)
        if 'tau_mu_exp' in quantity_names:
            # shortcut for normalized r1_mumod in for-loop
            r1_mumod_f0 = dr_ss_dmu 
            # real value equal to the time-integral of the filter from 0 to inf
            r1_mumod_normalized = r1mu_vec/params['d_mu'] /r1_mumod_f0
            init_val = 1.0 #ms
            tau = fit_exponential_freqdom(f, r1_mumod_normalized, init_val)
            res_given_sigma_dict['tau_mu_exp'][imu] = tau
            
        if 'tau_sigma_exp' in quantity_names:
            # shortcut for normalized r1_sigmamod in for-loop
            r1_sigmamod_f0 = dr_ss_dsig
            # real value equal to the time-integral of the filter from 0 to inf
            r1_sigmamod_normalized = r1sig_vec/params['d_sigma'] /r1_sigmamod_f0
            init_val = 0.1 #ms
            if r1_sigmamod_f0>0:  # explain
                tau = fit_exponential_freqdom(f, r1_sigmamod_normalized, init_val)
            else: 
                tau = 0.0
            res_given_sigma_dict['tau_sigma_exp'][imu] = tau
           
        if 'tau_mu_dosc' in quantity_names:
            sigmod = False
            r1_mumod_f0 = dr_ss_dmu
            # real value equal to the time-integral of the filter from 0 to inf
            # shortcuts for peak values and corresponding frequencies:
            fpeak_real_r1_mumod = res_given_sigma_dict['f_peak_real_r1_mumod'][imu]
            peak_real_r1_mumod = res_given_sigma_dict['peak_real_r1_mumod'][imu]/r1_mumod_f0
            fpeak_imag_r1_mumod = res_given_sigma_dict['f_peak_imag_r1_mumod'][imu]
            peak_imag_r1_mumod = res_given_sigma_dict['peak_imag_r1_mumod'][imu]/r1_mumod_f0

            # TODO: warning if first (smallest) mu > -0.5 and if dmu > 0.025
            firstfit = imu==0
            tau, f0 = fit_exp_damped_osc_freqdom(init_vals, fpeak_real_r1_mumod, 
                                                 peak_real_r1_mumod, 
                                                 fpeak_imag_r1_mumod, 
                                                 peak_imag_r1_mumod, firstfit, sigmod)
            res_given_sigma_dict['tau_mu_dosc'][imu] = tau
            res_given_sigma_dict['f0_mu_dosc'][imu] = f0
            init_vals =[tau, f0]
    
        # additional fitting optios in alternative version (below)    
    #print 'another mu curve done'
    return isig, res_given_sigma_dict

    
                                 
def EIF_ss_and_linresp_given_musigma_wrapper(arg_tuple):
    imu, mu, isig, sigma, params, output_names = arg_tuple
    V_vec = params['V_vals']
    Vr = params['Vr']
    kr = np.argmin(np.abs(V_vec-Vr))  # reset index value
    VT = params['VT']
    taum = params['taum']
    EL = params['EL']
    DeltaT = params['deltaT']
    
    res_given_musigma_dict = dict()
    
    # steady state output:
    p_ss, r_ss, q_ss = EIF_steady_state(V_vec, kr, taum, EL, Vr, VT, DeltaT, mu, sigma) 
    _, r_ss_dmu, _ = EIF_steady_state(V_vec, kr, taum, EL, Vr, VT, DeltaT, 
                                      mu+params['d_mu'], sigma)
    _, r_ss_dsig, _ = EIF_steady_state(V_vec, kr, taum, EL, Vr, VT, DeltaT, 
                                       mu, sigma+params['d_sigma'])
    
    dV = V_vec[1]-V_vec[0]
    Tref = params['t_ref']
    
    if 'V_mean_ss' in output_names:
        # disregarding Tref -> use for Vmean when clamping both, V and w:
        V_mean = dV*np.sum(V_vec*p_ss)  
        res_given_musigma_dict['V_mean_ss'] = V_mean
    
    r_ss_ref = r_ss/(1+r_ss*Tref)    
    p_ss = r_ss_ref * p_ss/r_ss
    q_ss = r_ss_ref * q_ss/r_ss  # prob. flux needed for sigma-mod calculation              
    r_ss_dmu_ref = r_ss_dmu/(1+r_ss_dmu*Tref) - r_ss_ref
    r_ss_dsig_ref = r_ss_dsig/(1+r_ss_dsig*Tref) - r_ss_ref   
    
    if 'V_mean_sps_ss' in output_names:
        # when considering spike shape (during refr. period):
        # density reflecting nonrefr. proportion, which integrates to r_ss_ref/r_ss 
        Vmean_sps = dV*np.sum(V_vec*p_ss) + (1-r_ss_ref/r_ss)*(params['Vcut']+Vr)/2  
        # note: (1-r_ss_ref/r_ss)==r_ss_ref*Tref 
        res_given_musigma_dict['V_mean_sps_ss'] = Vmean_sps
        
    if 'r_ss' in output_names:
        res_given_musigma_dict['r_ss'] = r_ss_ref
        
    if 'dr_ss_dmu' in output_names:      
        dr_ss_dmu = r_ss_dmu_ref/params['d_mu']
        res_given_musigma_dict['dr_ss_dmu'] = dr_ss_dmu
        
    if 'dr_ss_dsigma' in output_names:        
        dr_ss_dsig = r_ss_dsig_ref/params['d_sigma']
        res_given_musigma_dict['dr_ss_dsigma'] = dr_ss_dsig
    
        
    # next, mu-mod and sigma-mod over freq. range, and optionally, 
    # binary search to determine accurate peaks of |r1mu_vec|, 
    # and of |real(r1mu_vec)|, and |imag(r1mu_vec)| WITHIN range 
    # [0, params['freq_vals'][-1]], assuming a uniform freq_vals range 
    if 'r1_mumod' in output_names or 'peak_real_r1_mumod' in output_names \
    or 'peak_imag_r1_mumod' in output_names:
        w_vec = 2*np.pi*params['freq_vals']
        # mu1 = 1e-4
        # mu1 = params['d_mu'], consistently with use of r_ss_dmu_ref below
        inhom = params['d_mu']*p_ss
        r1mu_vec = EIF_lin_rate_response_frange(V_vec, kr, taum, EL, Vr, VT, DeltaT, 
                                                Tref, mu, sigma, inhom, w_vec)
        #r1mu_vec /= mu1
        res_given_musigma_dict['r1_mumod'] = r1mu_vec/params['d_mu']
        
        if 'peak_real_r1_mumod' in output_names:
            abs_re_im = 'real'    
            w_peak, peak_val = EIF_find_lin_response_peak(w_vec,r1mu_vec,r_ss_dmu_ref,
                                                      V_vec,kr,taum,EL,Vr,VT,DeltaT,
                                                      Tref, mu,sigma,inhom,abs_re_im)
            res_given_musigma_dict['peak_real_r1_mumod'] = peak_val/params['d_mu']
            res_given_musigma_dict['f_peak_real_r1_mumod'] = w_peak/(2*np.pi)
                                          
        if 'peak_imag_r1_mumod' in output_names:                                              
            abs_re_im = 'imag'    
            w_peak, peak_val = EIF_find_lin_response_peak(w_vec,r1mu_vec,r_ss_dmu_ref,
                                                      V_vec,kr,taum,EL,Vr,VT,DeltaT,
                                                      Tref, mu,sigma,inhom,abs_re_im)
            res_given_musigma_dict['peak_imag_r1_mumod'] = peak_val/params['d_mu']
            res_given_musigma_dict['f_peak_imag_r1_mumod'] = w_peak/(2*np.pi)                                          
   
    if 'r1_sigmamod' in output_names or 'peak_real_r1_sigmamod' in output_names \
    or 'peak_imag_r1_sigmamod' in output_names:
        w_vec = 2*np.pi*params['freq_vals']
        # sigma1 = 1e-4;  #inhom for sigma^2 modulation = -sigma21*dp_ssdV/2  
        # sigma1 = params['d_sigma'], consistently with use of r_ss_dsigma_ref (not implemented a.t.m.)
        driftterm = ( mu + ( EL-V_vec+DeltaT*np.exp((V_vec-VT)/DeltaT) )/taum ) * p_ss
        inhom = params['d_sigma'] * 2/sigma * (q_ss - driftterm)  # inhom = -sigma*sigma1*dp_ssdV
        r1sig_vec = EIF_lin_rate_response_frange(V_vec, kr, taum, EL, Vr, VT, DeltaT, 
                                                 Tref, mu, sigma, inhom, w_vec)
        #r1sig_vec /= sigma1
        res_given_musigma_dict['r1_sigmamod'] = r1sig_vec/params['d_sigma']   
        
        if 'peak_real_r1_sigmamod' in output_names:
            abs_re_im = 'real'    
            w_peak, peak_val = EIF_find_lin_response_peak(w_vec,r1sig_vec,r_ss_dsig_ref,
                                                      V_vec,kr,taum,EL,Vr,VT,DeltaT,
                                                      Tref, mu,sigma,inhom,abs_re_im)
            res_given_musigma_dict['peak_real_r1_sigmamod'] = peak_val/params['d_sigma']
            res_given_musigma_dict['f_peak_real_r1_sigmamod'] = w_peak/(2*np.pi)
                                          
        if 'peak_imag_r1_sigmamod' in output_names:                                              
            abs_re_im = 'imag'    
            w_peak, peak_val = EIF_find_lin_response_peak(w_vec,r1sig_vec,r_ss_dsig_ref,
                                                      V_vec,kr,taum,EL,Vr,VT,DeltaT,
                                                      Tref, mu,sigma,inhom,abs_re_im)
            res_given_musigma_dict['peak_imag_r1_sigmamod'] = peak_val/params['d_sigma']
            res_given_musigma_dict['f_peak_imag_r1_sigmamod'] = w_peak/(2*np.pi)
        
    return imu, isig, res_given_musigma_dict
    
    
@numba.njit
def EIF_steady_state(V_vec, kr, taum, EL, Vr, VT, DeltaT, mu, sigma):
    dV = V_vec[1]-V_vec[0]
    sig2term = 2.0/sigma**2
    n = len(V_vec)
    p_ss = np.zeros(n);  q_ss = np.ones(n);
    F = sig2term*( ( V_vec-EL-DeltaT*np.exp((V_vec-VT)/DeltaT) )/taum - mu ) 
    A = np.exp(dV*F)
    F_dummy = F.copy()
    F_dummy[F_dummy==0.0] = 1.0
    B = (A - 1.0)/F_dummy * sig2term
    sig2term_dV = dV * sig2term
    for k in xrange(n-1, kr, -1):
        if not F[k]==0.0:
            p_ss[k-1] = p_ss[k] * A[k] + B[k]
        else:
            p_ss[k-1] = p_ss[k] * A[k] + sig2term_dV
        q_ss[k-1] = 1.0    
    for k in xrange(kr, 0, -1):  
        p_ss[k-1] = p_ss[k] * A[k]
        q_ss[k-1] = 0.0
    p_ss_sum = np.sum(p_ss)   
    r_ss = 1.0/(dV*p_ss_sum)
    p_ss *= r_ss;  q_ss *= r_ss;
    return p_ss, r_ss, q_ss

    
@numba.njit
def EIF_lin_rate_response_frange(V_vec, kr, taum, EL, Vr, VT, DeltaT, Tref,
                                 mu, sigma, inhom, w_vec): 
    dV = V_vec[1]-V_vec[0]
    sig2term = 2.0/sigma**2
    n = len(V_vec)
    r1_vec = 1j*np.ones(len(w_vec)) 
    F = sig2term*( ( V_vec-EL-DeltaT*np.exp((V_vec-VT)/DeltaT) )/taum - mu ) 
    A = np.exp(dV*F)
    F_dummy = F.copy()
    F_dummy[F_dummy==0.0] = 1.0
    B = (A - 1.0)/F_dummy * sig2term
    sig2term_dV = dV * sig2term
    for iw in range(len(w_vec)):
        q1a = 1.0 + 0.0*1j;  p1a = 0.0*1j;  q1b = 0.0*1j;  p1b = 0.0*1j;
        fw = dV*1j*w_vec[iw]
        refterm = np.exp(-1j*w_vec[iw]*Tref)  
        for k in xrange(n-1, 0, -1):
            if not k==kr+1:
                q1a_new = q1a + fw*p1a
            else:
                q1a_new = q1a + fw*q1a - refterm
            if not F[k]==0.0:    
                p1a_new = p1a * A[k] + B[k] * q1a
                p1b_new = p1b * A[k] + B[k] * (q1b - inhom[k])
            else:
                p1a_new = p1a * A[k] + sig2term_dV * q1a
                p1b_new = p1b * A[k] + sig2term_dV * (q1b - inhom[k])
            q1b += fw*p1b
            q1a = q1a_new;  p1a = p1a_new;  p1b = p1b_new;   
        r1_vec[iw] = -q1b/q1a
    return r1_vec    
    
 
def EIF_find_lin_response_peak(w_vec, r1_vec, r1_f0, V_vec, kr, taum, EL, Vr, VT, 
                               DeltaT, Tref, mu, sigma, inhom, abs_re_im):                              
   if abs_re_im == 'abs':
       val = np.max(abs(r1_vec))
       ind = np.argmax(abs(r1_vec))
   elif abs_re_im == 'real':
       val = np.max(abs(np.real(r1_vec)))
       ind = np.argmax(abs(np.real(r1_vec)))
   elif abs_re_im == 'imag':
       val = np.max(abs(np.imag(r1_vec)))
       ind = np.argmax(abs(np.imag(r1_vec)))
   # now, binary search to get peak of r1_vec with higher accuracy 
   if ind>0: 
      w_peak = w_vec[ind];  dw = w_vec[1]-w_vec[0];
      w_min = 2*np.pi*1e-5
      bounds = [np.max([w_min,w_peak-dw]), w_peak+dw]
      cval = r1_vec[ind]
      while dw>2*np.pi*5e-6:
          r1l = EIF_lin_rate_response(V_vec, kr, taum, EL, Vr, VT, DeltaT, Tref,
                                      mu, sigma, inhom, bounds[0])
          if abs_re_im == 'abs':                             
              vall = abs(r1l)
          elif abs_re_im == 'real':
              vall = abs(np.real(r1l))
          elif abs_re_im == 'imag':
              vall = abs(np.imag(r1l))
          if vall>val:
              val = vall
              cval = r1l
              w_peak = bounds[0]
          else:
              r1r = EIF_lin_rate_response(V_vec, kr, taum, EL, Vr, VT, DeltaT, Tref,
                                          mu, sigma, inhom, bounds[1])
              if abs_re_im == 'abs':                             
                  valr = abs(r1r)
              elif abs_re_im == 'real':
                  valr = abs(np.real(r1r))
              elif abs_re_im == 'imag':
                  valr = abs(np.imag(r1r))               
              if valr>val:
                  val = valr
                  cval = r1r
                  w_peak = bounds[1]
          dw /= 2.0
          bounds = [np.max([w_min,w_peak-dw]), w_peak+dw]
      peak_val = cval
   else:
      w_peak = 0.0
      peak_val = r1_f0
   return w_peak, peak_val
  

@numba.njit
def EIF_lin_rate_response(V_vec, kr, taum, EL, Vr, VT, DeltaT, Tref,
                          mu, sigma, inhom, w): 
    dV = V_vec[1]-V_vec[0]
    sig2term = 2.0/sigma**2
    n = len(V_vec)   
    q1a = 1.0 + 0.0*1j;  p1a = 0.0*1j;  q1b = 0.0*1j;  p1b = 0.0*1j;
    fw = dV*1j*w
    refterm = np.exp(-1j*w*Tref)
    F = sig2term*( ( V_vec-EL-DeltaT*np.exp((V_vec-VT)/DeltaT) )/taum - mu ) 
    A = np.exp(dV*F)
    F_dummy = F.copy()
    F_dummy[F_dummy==0.0] = 1.0
    B = (A - 1.0)/F_dummy * sig2term
    sig2term_dV = dV * sig2term
    for k in xrange(n-1, 0, -1):
        if not k==kr+1:
            q1a_new = q1a + fw*p1a
        else:
            q1a_new = q1a + fw*q1a - refterm
        if not F[k]==0.0:    
            p1a_new = p1a * A[k] + B[k] * q1a
            p1b_new = p1b * A[k] + B[k] * (q1b - inhom[k])
        else:
            p1a_new = p1a * A[k] + sig2term_dV * q1a
            p1b_new = p1b * A[k] + sig2term_dV * (q1b - inhom[k])
        q1b += fw*p1b
        q1a = q1a_new;  p1a = p1a_new;  p1b = p1b_new;   
    r1 = -q1b/q1a
    return r1    


def calc_cascade_quantities(mu_vals, sigma_vals, params, EIF_output_dict, 
                            quantities_dict, quantity_names):
    
    print('Computing: {}'.format(quantity_names))   
    
    N_mu_vals = len(mu_vals)    
    N_sigma_vals = len(sigma_vals)
    total = len(sigma_vals)
    if total<=params['N_procs']:
        N_procs = 1
    else:
        N_procs = params['N_procs']
    
    N_procs = 1  # TODO: multiproc does not make sense here when precomputed EIF_output is very large
    
    # zero quantities_dict arrays
    for n in quantity_names:  
        # real values dependent on mu, sigma
        quantities_dict[n] = np.zeros((N_mu_vals, N_sigma_vals))
                
    arg_tuple_list = [(mu_vals, isig, sigma_vals[isig], params, EIF_output_dict, 
                       quantity_names) for isig in range(N_sigma_vals)]    
                      
    comp_total_start = time.time()
                  
    if N_procs <= 1:
        # single processing version, i.e. loop
        pool = False
        result = (quantities_forallmu_given_sigma_wrapper(arg_tuple) 
                  for arg_tuple in arg_tuple_list) 
    else:
        # multiproc version
        pool = multiprocessing.Pool(params['N_procs'])
        result = pool.imap_unordered(quantities_forallmu_given_sigma_wrapper, arg_tuple_list)
    
    finished = 0     
    for isig, res_given_sigma_dict in result:
        finished += 1
        print(('{count} of {tot} LN quantity calculations completed').
              format(count=finished, tot=total)) 
        for k in res_given_sigma_dict.keys():
            quantities_dict[k][:,isig] = res_given_sigma_dict[k]

    # furthermore:
    if 'r_ss' in quantity_names:
        # copy from EIF_output_dict
        quantities_dict['r_ss'] = EIF_output_dict['r_ss'].copy()

    if 'V_mean_ss' in quantity_names:
        # copy from EIF_output_dict
        quantities_dict['V_mean_ss'] = EIF_output_dict['V_mean_ss'].copy() 
        
    # also include mu_vals, sigma_vals, and freq_vals in output dictionary
    quantities_dict['mu_vals'] = mu_vals
    quantities_dict['sigma_vals'] = sigma_vals
    quantities_dict['freq_vals'] = EIF_output_dict['freq_vals'].copy()
        
    print('Computation of: {} done'.format(quantity_names))
    print('Total time for computation (N_mu_vals={Nmu}, N_sigma_vals={Nsig}): {rt}s'.
          format(rt=np.round(time.time()-comp_total_start,2), Nmu=N_mu_vals, Nsig=N_sigma_vals))
          
    return quantities_dict
    
 
def quantities_forallmu_given_sigma_wrapper(arg_tuple):
    mu_vals, isig, sigma, params, EIF_output_dict, quantity_names = arg_tuple   
     
    res_given_sigma_dict = dict()
        
    #  check equivalent of lsqcurvefit or minsearchbnd or lsqnonlin
        
    if 'tau_mu_exp' in quantity_names:
        # shortcuts for freq, and for normalized r1_mumod in for-loop
        f = EIF_output_dict['freq_vals']
        res_given_sigma_dict['tau_mu_exp'] = np.zeros(len(mu_vals))
        for imu, mu in enumerate(mu_vals): 
            r1_mumod_f0 = EIF_output_dict['dr_ss_dmu'][imu,isig] 
            # real value equal to the time-integral of the filter from 0 to inf
            r1_mumod_normalized = EIF_output_dict['r1_mumod'][imu,isig,:]/r1_mumod_f0
            init_val = 1.0 #ms
            tau = fit_exponential_freqdom(f, r1_mumod_normalized, init_val)
            res_given_sigma_dict['tau_mu_exp'][imu] = tau
            
    if 'tau_sigma_exp' in quantity_names:
        # shortcuts for freq, and for normalized r1_sigmamod in for-loop
        f = EIF_output_dict['freq_vals']
        res_given_sigma_dict['tau_sigma_exp'] = np.zeros(len(mu_vals))
        for imu, mu in enumerate(mu_vals): 
            r1_sigmamod_f0 = EIF_output_dict['dr_ss_dsigma'][imu,isig] 
            # real value equal to the time-integral of the filter from 0 to inf
            r1_sigmamod_normalized = EIF_output_dict['r1_sigmamod'][imu,isig,:]/r1_sigmamod_f0
            init_val = 0.1 #ms
            if r1_sigmamod_f0>0:  # explain
                tau = fit_exponential_freqdom(f, r1_sigmamod_normalized, init_val)
            else: 
                tau = 0.0
            res_given_sigma_dict['tau_sigma_exp'][imu] = tau
           
    if 'tau_mu_dosc' in quantity_names:
        res_given_sigma_dict['tau_mu_dosc'] = np.zeros(len(mu_vals))
        res_given_sigma_dict['f0_mu_dosc'] = np.zeros(len(mu_vals))
        init_vals = [10.0, 0.01]  # tau (ms), f0 (kHz)
        sigmod = False
        for imu, mu in enumerate(mu_vals): 
            r1_mumod_f0 = EIF_output_dict['dr_ss_dmu'][imu,isig]
            # real value equal to the time-integral of the filter from 0 to inf
            # shortcuts for peak values and corresponding frequencies:
            fpeak_real_r1_mumod = EIF_output_dict['f_peak_real_r1_mumod'][imu,isig]
            peak_real_r1_mumod = EIF_output_dict['peak_real_r1_mumod'][imu,isig]/r1_mumod_f0
            fpeak_imag_r1_mumod = EIF_output_dict['f_peak_imag_r1_mumod'][imu,isig]
            peak_imag_r1_mumod = EIF_output_dict['peak_imag_r1_mumod'][imu,isig]/r1_mumod_f0

            # TODO: warning if first (smallest) mu > -0.5 and if dmu > 0.025
            firstfit = imu==0
            tau, f0 = fit_exp_damped_osc_freqdom(init_vals, fpeak_real_r1_mumod, 
                                                 peak_real_r1_mumod, 
                                                 fpeak_imag_r1_mumod, 
                                                 peak_imag_r1_mumod, firstfit, sigmod)
            res_given_sigma_dict['tau_mu_dosc'][imu] = tau
            res_given_sigma_dict['f0_mu_dosc'][imu] = f0
            init_vals =[tau, f0]
           
    if 'tau_sigma_dosc' in quantity_names:
        res_given_sigma_dict['tau_sigma_dosc'] = np.zeros(len(mu_vals))
        res_given_sigma_dict['f0_sigma_dosc'] = np.zeros(len(mu_vals))
        init_vals = [10.0, 0.01]  # tau (ms), f0 (kHz)
        sigmod = True
        for imu, mu in enumerate(mu_vals): 
            r1_sigmamod_f0 = EIF_output_dict['dr_ss_dsigma'][imu,isig]
            # real value equal to the time-integral of the filter from 0 to inf
            # shortcuts for peak values and corresponding frequencies:
            fpeak_real_r1_sigmod = EIF_output_dict['f_peak_real_r1_sigmamod'][imu,isig]
            peak_real_r1_sigmod = EIF_output_dict['peak_real_r1_sigmamod'][imu,isig]/np.abs(r1_sigmamod_f0)
            fpeak_imag_r1_sigmod = EIF_output_dict['f_peak_imag_r1_sigmamod'][imu,isig]
            peak_imag_r1_sigmod = EIF_output_dict['peak_imag_r1_sigmamod'][imu,isig]/np.abs(r1_sigmamod_f0)

            # TODO: warning if first (smallest) mu > XX and if dmu > 0.025 CHECK FOR SIGMAMOD
            firstfit = imu==0
            if r1_sigmamod_f0>0:  # explain
                tau, f0 = fit_exp_damped_osc_freqdom(init_vals, fpeak_real_r1_sigmod, 
                                                     peak_real_r1_sigmod, 
                                                     fpeak_imag_r1_sigmod, 
                                                     peak_imag_r1_sigmod, firstfit, sigmod)
            else: 
                tau = 0.0
                f0 = res_given_sigma_dict['f0_sigma_dosc'][imu-1]                                             
#            if r1_sigmamod_f0>0:  # explain                                     
#                res_given_sigma_dict['tau_sigma_dosc'][imu] = tau
#            else: 
#                res_given_sigma_dict['tau_sigma_dosc'][imu] = 0.0
            res_given_sigma_dict['tau_sigma_dosc'][imu] = tau    
            res_given_sigma_dict['f0_sigma_dosc'][imu] = f0
            init_vals =[tau, f0]           
           
    if 'tau1_mu_bedosc' in quantity_names:
        res_given_sigma_dict['B_mu_bedosc'] = np.zeros(len(mu_vals))
        res_given_sigma_dict['tau1_mu_bedosc'] = np.zeros(len(mu_vals))
        res_given_sigma_dict['tau2_mu_bedosc'] = np.zeros(len(mu_vals))
        res_given_sigma_dict['f0_mu_bedosc'] = np.zeros(len(mu_vals))
        f = EIF_output_dict['freq_vals']     
        
        if sigma<=2.0:
            first_mu = 0.5
            init_vals = [1.0, 20.0, 0.01, 0.1] # tau1 (ms), tau2 (ms), f0 (kHz), B
        elif sigma<=3.0:
            first_mu = 3.0
            init_vals = [0.5, 10.0, 0.05, 0.1]
        else:
            first_mu = 5.0
            init_vals = [0.5, 10.0, 0.05, 0.1]            
        imu_start = np.argmin(np.abs(first_mu - mu_vals)) 
        firstfit = True
#        # first, forward from imu_start
#        for imu in range(imu_start, len(mu_vals)): 
#            r1_mumod_f0 = EIF_output_dict['dr_ss_dmu'][imu,isig] 
#            # real value equal to the time-integral of the filter from 0 to inf
#            r1_mumod_normalized = EIF_output_dict['r1_mumod'][imu,isig,:]/r1_mumod_f0
#            
#            B, tau1, tau2, f0 = fit_biexp_damped_osc_freqdom(f, r1_mumod_normalized, 
#                                                             init_vals, firstfit, 
#                                                             forward=True, lastrun=False)
#            res_given_sigma_dict['B_mu_bedosc'][imu] = B
#            res_given_sigma_dict['tau1_mu_bedosc'][imu] = tau1
#            res_given_sigma_dict['tau2_mu_bedosc'][imu] = tau2
#            res_given_sigma_dict['f0_mu_bedosc'][imu] = f0
##            print 'mu,sig = ', mu_vals[imu], sigma, ' done:1' 
#            firstfit = False
#            init_vals = [tau1, tau2, f0, B] 
##            print init_vals
#        
#        init_vals = [res_given_sigma_dict['tau1_mu_bedosc'][imu_start], 
#                     res_given_sigma_dict['tau2_mu_bedosc'][imu_start],
#                     res_given_sigma_dict['f0_mu_bedosc'][imu_start],
#                     res_given_sigma_dict['B_mu_bedosc'][imu_start]]
        
        # first, backward from imu_start
        find_imu = False  # find mu, which separates exp from bexpdosc fits 
        for imu in range(imu_start, -1, -1):
            r1_mumod_f0 = EIF_output_dict['dr_ss_dmu'][imu,isig] 
            # real value equal to the time-integral of the filter from 0 to inf
            r1_mumod_normalized = EIF_output_dict['r1_mumod'][imu,isig,:]/r1_mumod_f0
            
            B, tau1, tau2, f0 = fit_biexp_damped_osc_freqdom(f, r1_mumod_normalized, 
                                                             init_vals, firstfit, 
                                                             forward=False, lastrun=False)
            res_given_sigma_dict['B_mu_bedosc'][imu] = B
            res_given_sigma_dict['tau1_mu_bedosc'][imu] = tau1
            res_given_sigma_dict['tau2_mu_bedosc'][imu] = tau2
            res_given_sigma_dict['f0_mu_bedosc'][imu] = f0
            firstfit = False
            init_vals = [tau1, tau2, f0, B]
            print 'mu,sig,vals = ', mu_vals[imu], sigma, np.round(init_vals,2)
            if f0==0.0 and find_imu:
                imu_start = imu
                find_imu = False
           
#        # then, forward from newly determined mu_start which separates 
#        # exp from bexpdosc fits       
#        init_vals = [res_given_sigma_dict['tau1_mu_bedosc'][imu_start], 
#                     res_given_sigma_dict['tau2_mu_bedosc'][imu_start],
#                     0.05, 0.1]
#        firstfit = True  
#        for imu in range(imu_start+1, len(mu_vals)): 
#            r1_mumod_f0 = EIF_output_dict['dr_ss_dmu'][imu,isig] 
#            # real value equal to the time-integral of the filter from 0 to inf
#            r1_mumod_normalized = EIF_output_dict['r1_mumod'][imu,isig,:]/r1_mumod_f0
#             
#            B, tau1, tau2, f0 = fit_biexp_damped_osc_freqdom(f, r1_mumod_normalized, 
#                                                             init_vals, firstfit, 
#                                                             forward=True, lastrun=True)
#            res_given_sigma_dict['B_mu_bedosc'][imu] = B
#            res_given_sigma_dict['tau1_mu_bedosc'][imu] = tau1
#            res_given_sigma_dict['tau2_mu_bedosc'][imu] = tau2
#            res_given_sigma_dict['f0_mu_bedosc'][imu] = f0
##            print 'mu,sig = ', mu_vals[imu], sigma, ' done:3' 
#            firstfit = False
#            init_vals = [tau1, tau2, f0, B]
##            print init_vals
        # then, forward from mu_start  
        init_vals = [res_given_sigma_dict['tau1_mu_bedosc'][imu_start], 
                     res_given_sigma_dict['tau2_mu_bedosc'][imu_start],
                     res_given_sigma_dict['f0_mu_bedosc'][imu_start],
                     res_given_sigma_dict['B_mu_bedosc'][imu_start]]
        firstfit = False 
        for imu in range(imu_start+1, len(mu_vals)): 
            r1_mumod_f0 = EIF_output_dict['dr_ss_dmu'][imu,isig] 
            # real value equal to the time-integral of the filter from 0 to inf
            r1_mumod_normalized = EIF_output_dict['r1_mumod'][imu,isig,:]/r1_mumod_f0
             
            B, tau1, tau2, f0 = fit_biexp_damped_osc_freqdom(f, r1_mumod_normalized, 
                                                             init_vals, firstfit, 
                                                             forward=True, lastrun=False)
            res_given_sigma_dict['B_mu_bedosc'][imu] = B
            res_given_sigma_dict['tau1_mu_bedosc'][imu] = tau1
            res_given_sigma_dict['tau2_mu_bedosc'][imu] = tau2
            res_given_sigma_dict['f0_mu_bedosc'][imu] = f0
            firstfit = False
            init_vals = [tau1, tau2, f0, B]
            print 'mu,sig,vals = ', mu_vals[imu], sigma, np.round(init_vals,2)
            
#        plt.figure()
#        ax = plt.subplot(4,1,1)
#        plt.plot(mu_vals, res_given_sigma_dict['B_mu_bedosc'])
#        plt.subplot(4,1,2, sharex=ax)
#        plt.plot(mu_vals, res_given_sigma_dict['tau1_mu_bedosc'])
#        plt.subplot(4,1,3, sharex=ax)
#        plt.plot(mu_vals, res_given_sigma_dict['tau2_mu_bedosc'])
#        plt.subplot(4,1,4, sharex=ax)
#        plt.plot(mu_vals, res_given_sigma_dict['f0_mu_bedosc'])
        
    return isig, res_given_sigma_dict
    
           
def fit_exponential_freqdom(f, r1_mod_normalized, init_val):
    # Fitting exponential function A*exp(-t/tau) to normalized rate response in Fourier space:
    # A*tau / (1 + 1i*2*pi*f*tau)  f in kHz, tau in ms 
    # with A = 1/tau to guarantee equality at freq=0

    tau_lb = 0.001  #ms 
    tau_ub = 300.0  #ms
#    sol = scipy.optimize.minimize_scalar(exp_mean_sq_dist, args=(f, r1_mod_normalized), 
#                                         bounds=(tau_lb, tau_ub), method='bounded', 
#                                         options={'disp':True, 'maxiter':1000, 'xatol':1e-3})
#    tau = sol.x
    ydata = np.concatenate([np.real(r1_mod_normalized), np.imag(r1_mod_normalized)])
    tau, _ = scipy.optimize.curve_fit(exponential_fdom, f, ydata, p0=init_val, 
                                      bounds=(tau_lb, tau_ub)) 
    # curve_fit is a bit slower, but works better [for some reason] for sigma-mod 
    # when drdsigma is close to 0 but still pos.:
    # in this case the other method yields tau_max which may be a local optimum
    # To check this we could use a global optimization method using numba for this
    # simple 1-dim. optim. 
    # TODO: numba based global optim. (optional, probably faster)                                  
    return tau
    
def exp_mean_sq_dist(tau, *args):  # not used a.t.m.
    # Fitting exponential function A*exp(-t/tau) in Fourier space to data:
    # A*tau / (1 + 1i*2*pi*f*tau)  f in kHz, tau in ms 
    # with A = 1/tau to guarantee equality at freq=0 (assuming normalized data)
    f, r1_mod_normalized = args
    exp_fdom = 1.0 / (1.0 + 1j*2.0*np.pi*f*tau)  # exp function evaluated at freqs. f
    error = np.sum(np.abs(exp_fdom - r1_mod_normalized)**2)  
    # correct normalization for mean squared error not important
    return error
    
def exponential_fdom(f_vals, tau):
    out = 1.0 / (1.0 + 2*1j*np.pi*f_vals*tau)
    return np.concatenate([np.real(out), np.imag(out)])
 

def fit_exp_damped_osc_freqdom(init_vals, fpeak_real_r1_mumod, peak_real_r1_mumod, 
                               fpeak_imag_r1_mumod, peak_imag_r1_mumod, firstfit, sigmod):
    # Fitting damped oscillator function (with exponential decay) A*exp(-t/tau)*cos(2*pi*f0*t) 
    # to normalized rate response in Fourier space:
    # A*tau/2 * ( 1/(1 + 2*pi*1i*tau*(f-f0)) + 1/(1 + 2*pi*1i*tau*(f+f0)) )
    # with A = (1 + (2*pi*f0*tau)^2)/tau to guarantee equality at freq=0
         
    # first global method to avoid local optima and use tight boundaries in this case
    # start global (tight bounds)...
    if firstfit:  # first fit, seems appropriate for mu < -0.5
        if sigmod:
            tau_vals = np.arange(1.0, 25, 0.025)  #ms
        else:
            tau_vals = np.arange(10.0, 35, 0.025)  #ms
        f0_vals = np.arange(0.0, 0.005, 1e-4)  #kHz
    else:
        tau_vals = np.arange(np.max([0.0, init_vals[0]-5]), init_vals[0]+2, 0.005)  
        # dtau=0.005 good for global search, 0.025 maybe also ok and faster but should use optimization to refine
        f0_vals = np.arange(np.max([0.0, init_vals[1]-0.001]), init_vals[1]+0.02, 5e-5) 
        # df=1e-5 good for global search, df=5e-5 maybe also ok and faster but should use optimization to refine

#    errors = np.zeros((len(tau_vals),len(f0_vals)))
#    for i in range(len(tau_vals)):
#        for j in range(len(f0_vals)):
#            p = [tau_vals[i], f0_vals[j]]
#            dosc_at_fpreal = eval_dosc_fdom(p, fpeak_real_r1_mumod)
#            dosc_at_fpimag = eval_dosc_fdom(p, fpeak_imag_r1_mumod)
#            errors[i,j] = np.abs(dosc_at_fpreal - peak_real_r1_mumod)**2 + \
#                          np.abs(dosc_at_fpimag - peak_imag_r1_mumod)**2
    # --> numba function
    args = (fpeak_real_r1_mumod, peak_real_r1_mumod, fpeak_imag_r1_mumod, peak_imag_r1_mumod)
    errors = dosc_mean_sq_dist_2f_tauf0grid(tau_vals, f0_vals, args)     
    imin, jmin = np.unravel_index(errors.argmin(),errors.shape)
    tau = tau_vals[imin]
    f0 = f0_vals[jmin]
    #print sigmod, np.round(tau,2), np.round(f0,2)
    #...then refine
    init_vals = [tau, f0]
#    tau_lb = np.max([0.0, tau-10]);  tau_ub = tau+5;
#    f0_lb = np.max([0.0, f0-0.01]);  f0_ub = f0+0.05;
#    bnds = ( (tau_lb, tau_ub), (f0_lb, f0_ub) )
#    sol = scipy.optimize.minimize(dosc_mean_sq_dist_2f, init_vals, args=args, 
#                                  method='L-BFGS-B', bounds=bnds, 
#                                  options={'disp':False, 'ftol':1e-15, 'gtol':1e-10})
    # !!! gradient-based optim methods do not work reliably here -- more precisely, 
    # the discretization parameters to compute the derivatives (jacobian etc.) 
    # appropriately have not yet been determined properly !!!    
    if f0==0:                     
        sol = scipy.optimize.minimize(dosc_mean_sq_dist_2f, tau, args=args,
                                      method='Nelder-Mead', options={'xtol':1e-9, 'ftol':1e-9})                 
        #print sol  #TEMP  
        tau = sol.x
        f0 = 0.0
    else:
        sol = scipy.optimize.minimize(dosc_mean_sq_dist_2f, init_vals, args=args,
                                      method='Nelder-Mead', options={'xtol':1e-9, 'ftol':1e-9})                 
        #print sol  #TEMP  
        tau, f0 = sol.x  
    return tau, f0
                                 
                         
def dosc_mean_sq_dist_2f(p, *args):
    f_val1, r1_val1, f_val2, r1_val2 = args
    if np.size(p)<2:
        p = np.array([p[0], 0.0])
    dosc_at_f1 = eval_dosc_fdom(p, f_val1)
    dosc_at_f2 = eval_dosc_fdom(p, f_val2)
    error = np.abs(dosc_at_f1 - r1_val1)**2 + np.abs(dosc_at_f2 - r1_val2)**2  
    # correct normalization for mean squared error not important
    return error


@numba.njit
def dosc_mean_sq_dist_2f_tauf0grid(tau_vals, f0_vals, args):
    f_val1, r1_val1, f_val2, r1_val2 = args
    errors = np.zeros((len(tau_vals),len(f0_vals)))
    for i in range(len(tau_vals)):
        for j in range(len(f0_vals)):
            tau = tau_vals[i];  f0 = f0_vals[j];
            dosc_at_f1 = (1.0 + (2*np.pi*f0*tau)**2) / 2 * \
                         ( 1.0 / (1 + 2*np.pi*1j*tau*(f_val1-f0)) + \
                           1.0/ (1 + 2*np.pi*1j*tau*(f_val1+f0)) )
            dosc_at_f2 = (1.0 + (2*np.pi*f0*tau)**2) / 2 * \
                         ( 1.0 / (1 + 2*np.pi*1j*tau*(f_val2-f0)) + \
                           1.0/ (1 + 2*np.pi*1j*tau*(f_val2+f0)) )
            errors[i,j] = np.abs(dosc_at_f1 - r1_val1)**2 + \
                          np.abs(dosc_at_f2 - r1_val2)**2
    return errors  


@numba.njit
def eval_dosc_fdom(p, f):
    tau = p[0]  
    f0 = p[1]
    out = (1.0 + (2*np.pi*f0*tau)**2) / 2 * \
          ( 1.0 / (1 + 2*np.pi*1j*tau*(f-f0)) + 1.0/ (1 + 2*np.pi*1j*tau*(f+f0)) )
    return out


def fit_biexp_damped_osc_freqdom(f, r1_mumod_normalized, init_vals, firstfit, 
                                 forward=True, lastrun=True):
#    Fitting damped oscillator function 
#    A*exp(-t/tau1) + B*exp(-t/tau2)*cos(2*pi*f0)
#    tau1, tau2 in ms, f0 in kHz, all >= 0 (also A, B)
#    in Fourier space to data (mu-mod):
#    A*tau1 + B*tau2/2 * (1/(1+2*pi*1i*tau2*(f-f0)) + 1/(1+2*pi*1i*tau2*(f+f0)))
#    where A = drdmu/tau1 - B*tau2/((1 + (2*pi*f0*tau)^2)*tau1) to guarantee 
#    equal integrals of filters
#    Remark: 
#    when data (mu-mod) is normalized using drdmu, then drdmu in A should be 1
      
    ydata = np.concatenate([np.real(r1_mumod_normalized), np.imag(r1_mumod_normalized)])
    
    #TODO: let bounds depend on d_mu (same may apply for DOSC fitting)
	                      
    maxdtau1 = 25.0  #ms
    maxdtau2 = 50.0  #ms
    maxdf0 = 0.005  #kHz
    maxdB = 0.02
    
    f0_thresh = -10e-3  #this threshold value may be larger: 12e-3, 15e-3, 20e-3 and 25e-3 also look fine
    
    if lastrun:  #note: only lastrun is forward
        if firstfit: 
            # lower bound for tau1, tau2, f0, B
            lower = [0.75*init_vals[0], 0.75*init_vals[1], 1e-4, 1e-5]
            # upper bound for tau1, tau2, f0, B
            upper = [1.0*init_vals[0], 1.1*init_vals[1], 0.1, maxdB] 
            init_vals[2] = 0.01;  init_vals[3] = 0.02;  # not needed in Matlab
        else:
            lower = [0.5*init_vals[0], 0.75*init_vals[1], 1.0*init_vals[2], np.max([0.0,init_vals[3]-maxdB])] #0.5*init_vals[3]
            upper = [1.1*init_vals[0], 1.5*init_vals[1], init_vals[2]+maxdf0, init_vals[3]+maxdB] #5.0*init_vals[3]
        out, _ = scipy.optimize.curve_fit(biexp_damped_osc_fdom, f, ydata, p0=init_vals, 
                                          bounds=(lower, upper))
        tau1, tau2, f0, B = out
        A = 1.0/tau1 - B*tau2/( (1.0 + (2*np.pi*f0*tau2)**2)*tau1 )
    else:  
        if firstfit: 
            lower = [0.01, 0.05, 0.0, 0.0] 
            upper = [init_vals[0]+maxdtau1, 300.0, 0.2, 1.0] 
        else:
            lower = [0.5*init_vals[0], 0.8*init_vals[1], np.max([0.0,init_vals[2]-maxdf0]), np.max([0.0,init_vals[3]-maxdB])]   
            upper = [1.5*init_vals[0], 1.5*init_vals[1], init_vals[2]+maxdf0, init_vals[3]+maxdB] 
#            lower = [np.max([0.0,init_vals[0]-maxdtau1]), 
#                     np.max([0.0,init_vals[1]-maxdtau2]), 
#                     np.max([0.0,init_vals[2]-maxdf0]), 0.5*init_vals[3]]  
#            upper = [init_vals[0]+maxdtau1, init_vals[1]+maxdtau2, 
#                     init_vals[2]+maxdf0, 1.5*init_vals[3]]

        if init_vals[2]>f0_thresh:
            out, _ = scipy.optimize.curve_fit(biexp_damped_osc_fdom, f, ydata, p0=init_vals, 
                                              bounds=(lower, upper))
            tau1, tau2, f0, B = out
            A = 1.0/tau1 - B*tau2/( (1.0 + (2*np.pi*f0*tau2)**2)*tau1 )
        else:
            out, _ = scipy.optimize.curve_fit(exponential_fdom, f, ydata, p0=init_vals[0], 
                                              bounds=(lower[0], upper[0]))
            tau1 = out[0]
            tau2 = init_vals[1]
            f0 = 0.0
            B = 0.0
            A = 1.0/tau1 - B*tau2/( (1.0 + (2*np.pi*f0*tau2)**2)*tau1 )
    
#        if not firstfit and (f0>init_vals[2] or f0<=f0_thresh or B>=A):  #note: only lastrun is forward
#        # or B_out>1.0*init_vals[3]
#            out, _ = scipy.optimize.curve_fit(exponential_fdom, f, ydata, p0=init_vals[0], 
#                                              bounds=(lower[0], upper[0]))
#            tau1 = out[0]
#            tau2 = init_vals[1]
#            f0 = 0.0
#            B = 0.0
#            A = 1.0/tau1 - B*tau2/( (1.0 + (2*np.pi*f0*tau2)**2)*tau1 )

    return B, tau1, tau2, f0


def biexp_damped_osc_fdom(f_vals, *params):
    tau1, tau2, f0, B = params
    A = 1.0/tau1 - B*tau2/( (1.0 + (2*np.pi*f0*tau2)**2)*tau1 )
    out = A*tau1/(1.0 + 2*np.pi*1j*f_vals*tau1) + \
          B*tau2/2 * ( 1.0/(1.0 + 2*np.pi*1j*tau2*(f_vals-f0)) + \
                       1.0/(1.0 + 2*np.pi*1j*tau2*(f_vals+f0)) )
    return np.concatenate([np.real(out), np.imag(out)])


# LOAD / SAVE FUNCTIONS --------------------------------------------------------
def load(filepath, input_dict, quantities, param_dict):
    
    print('loading {} from file {}'.format(quantities, filepath))
    try:
        h5file = tables.open_file(filepath, mode='r')
        root = h5file.root
        
        for q in quantities:
            input_dict[q] = h5file.get_node(root, q).read()            
                   
        # loading params
        # only overwrite what is in the file. not start params from scratch. otherwise: uncomment following line
        #param_dict = {} 
        for child in root.params._f_walknodes('Array'):
            param_dict[child._v_name] = child.read()[0]
        for group in root.params._f_walk_groups():
            if group != root.params: # walk group first yields the group itself then its children
                param_dict[group._v_name] = {}
                for subchild in group._f_walknodes('Array'):
                    param_dict[group._v_name][subchild._v_name] = subchild.read()[0]            
        
        h5file.close()
    
    except IOError:
        warn('could not load quantities from file '+filepath)
    except:
        h5file.close()
    
    print('')
        

def save(filepath, output_dict, param_dict):
    
    print('saving {} into file {}'.format(output_dict.keys(), filepath))
    try:
        h5file = tables.open_file(filepath, mode='w')
        root = h5file.root
            
        for k in output_dict.keys():
            h5file.create_array(root, k, output_dict[k])
            print('created array {}'.format(k))
            
        h5file.create_group(root, 'params', 'Neuron model and numerics parameters')
        for name, value in param_dict.items():
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


# PLOTTING FUNCTIONS -----------------------------------------------------------
def plot_quantities_forpaper(quantities_dict, quantity_names, sigmas_quant_plot, mus_plot, sigmas_plot):
    
    mu_vals = quantities_dict['mu_vals']
    sigma_vals = quantities_dict['sigma_vals']   
    
    plt.figure()
    plt.suptitle('LN quantities')
    
    mu_lim = [-1, 4]
    inds_mu_plot = [i for i in range(len(mu_vals)) if mu_lim[0] <= mu_vals[i] <= mu_lim[1]]
    inds_sigma_plot = [np.argmin(np.abs(sigma_vals-sig)) for sig in sigmas_quant_plot]
    mu_plot_ind = np.argmin(np.abs(mu_vals-mus_plot[0]))
    N_sigma = len(inds_sigma_plot)    
    
    for k_j, j in enumerate(inds_sigma_plot):
        # color    
        rgb = [0, float(k_j)/(N_sigma-1), 0]
        linecolor = rgb
        
        if 'r_ss' in quantity_names:
            ax1 = plt.subplot(1, 4, 1)
            # labels
            if k_j in [0, N_sigma//2, N_sigma-1]:
                siglabel = '$\sigma={0:.3}$ [mV/$\sqrt{{\mathrm{{ms}}}}$]'.format(sigma_vals[j])
                           
            else:
                siglabel = None
    
            plt.plot(mu_vals[inds_mu_plot], quantities_dict['r_ss'][inds_mu_plot,j],
                     label=siglabel, color=linecolor)
            for l in range(len(sigmas_plot)): 
                if np.round(sigmas_plot[l],2)==np.round(sigma_vals[j],2):
                    plt.plot(mus_plot[0], quantities_dict['tau_mu_exp'][mu_plot_ind,j],
                    'o', color=linecolor) #, markeredgewidth=0.0)
            if k_j==0:
                plt.title(r'$r_{\infty}$', fontsize=14)
                plt.ylabel('[kHz]', fontsize=12)
                plt.xlabel('$\mu$ [mV/ms]', fontsize=12)
                #plt.ylim([-1, 46])
            
                               
        if 'V_mean_ss' in quantity_names:
            plt.subplot(1, 4, 2, sharex=ax1)
            plt.plot(mu_vals[inds_mu_plot], quantities_dict['V_mean_ss'][inds_mu_plot,j],
                     label=siglabel, color=linecolor)
            for l in range(len(sigmas_plot)): 
                if np.round(sigmas_plot[l],2)==np.round(sigma_vals[j],2):
                    plt.plot(mus_plot[0], quantities_dict['tau_sigma_exp'][mu_plot_ind,j],
                    'o', color=linecolor) #, markeredgewidth=0.0)
            if k_j==0:
                plt.title('$\langle V \\rangle_{\infty}$', fontsize=14)
                plt.ylabel('[mV]', fontsize=12)
                plt.xlabel('$\mu$ [mV/ms]', fontsize=12)
                #plt.ylim([-1, 46])
            if k_j==N_sigma-1:
                plt.legend(loc='best')
        
        if 'tau_mu_dosc' in quantity_names:
            plt.subplot(1, 4, 3, sharex=ax1)           
            plt.plot(mu_vals[inds_mu_plot], quantities_dict['tau_mu_dosc'][inds_mu_plot,j], 
                     label=siglabel, color=linecolor)
            for l in range(len(sigmas_plot)): 
                if np.round(sigmas_plot[l],2)==np.round(sigma_vals[j],2):
                    plt.plot(mus_plot[0], quantities_dict['tau_mu_dosc'][mu_plot_ind,j], 
                    'o', color=linecolor) #, markeredgewidth=0.0)
            if k_j==0:
                plt.title(r'$\tau$', fontsize=14)
                plt.ylabel('[ms]', fontsize=12)
                plt.xlabel('$\mu$ [mV/ms]', fontsize=12)
                plt.ylim([-1, 46])
        
        if 'f0_mu_dosc' in quantity_names:
            plt.subplot(1, 4, 4, sharex=ax1)                       
            plt.plot(mu_vals[inds_mu_plot], quantities_dict['f0_mu_dosc'][inds_mu_plot,j], 
                     label=siglabel, color=linecolor)
            for l in range(len(sigmas_plot)): 
                if np.round(sigmas_plot[l],2)==np.round(sigma_vals[j],2):
                    plt.plot(mus_plot[0], quantities_dict['f0_mu_dosc'][mu_plot_ind,j], 
                    'o', color=linecolor) #, markeredgewidth=0.0)  #markeredgecolor=...,
            if k_j==0:
                plt.title(r'$\omega/2\pi$', fontsize=14)
                plt.ylabel('[kHz]', fontsize=12)
                plt.xlabel('$\mu$ [mV/ms]', fontsize=12)
                plt.ylim([-0.005, 0.12])
    plt.show()            



#this was extra code for Fig5 manuscript, can be removed
def fig5_addon_Fabian(quantities_dict, quantity_names, sigmas_quant_plot, mus_plot, sigmas_plot):

    import matplotlib.pyplot as plt
    mu_vals = quantities_dict['mu_vals']
    sigma_vals = quantities_dict['sigma_vals']

    plt.figure()
    plt.suptitle('LN quantities')

    mu_lim = [-1, 4]
    inds_mu_plot = [i for i in range(len(mu_vals)) if mu_lim[0] <= mu_vals[i] <= mu_lim[1]]
    inds_sigma_plot = [np.argmin(np.abs(sigma_vals-sig)) for sig in sigmas_quant_plot]
    mu_plot_ind = np.argmin(np.abs(mu_vals-mus_plot[0]))
    N_sigma = len(inds_sigma_plot)

    for k_j, j in enumerate(inds_sigma_plot):
        # color
        rgb = [0, float(k_j)/(N_sigma-1), 0]
        linecolor = rgb

        if 'r_ss' in quantity_names:
            ax1 = plt.subplot(2, 2, 1)
            # ax1 = plt.subplot2grid((2,2), (0,0))
            # labels
            if k_j in [0, N_sigma//2, N_sigma-1]:
                siglabel = '$\sigma={0:.3}$ [mV/$\sqrt{{\mathrm{{ms}}}}$]'.format(sigma_vals[j])

            else:
                siglabel = None

            plt.plot(mu_vals[inds_mu_plot], quantities_dict['r_ss'][inds_mu_plot,j],
                     label=siglabel, color=linecolor)
            for l in range(len(sigmas_plot)):
                if np.round(sigmas_plot[l],2)==np.round(sigma_vals[j],2):
                    plt.plot(mus_plot[0], quantities_dict['r_ss'][mu_plot_ind,j],
                    'o', color=linecolor, markeredgewidth=0.0)
            if k_j==0:
                plt.title(r'$r_{\infty}$', fontsize=14)
                plt.ylabel('[kHz]', fontsize=12)
                plt.xlabel('$\mu$ [mV/ms]', fontsize=12)
                plt.ylim([0,0.14])


        if 'V_mean_ss' in quantity_names:
            plt.subplot(2, 2, 3, sharex=ax1)
            plt.plot(mu_vals[inds_mu_plot], quantities_dict['V_mean_ss'][inds_mu_plot,j],
                     label=siglabel, color=linecolor)
            for l in range(len(sigmas_plot)):
                if np.round(sigmas_plot[l],2)==np.round(sigma_vals[j],2):
                    plt.plot(mus_plot[0], quantities_dict['V_mean_ss'][mu_plot_ind,j],
                    'o', color=linecolor, markeredgewidth=0.0)
            if k_j==0:
                plt.title('$\langle V \\rangle_{\infty}$', fontsize=14)
                plt.ylabel('[mV]', fontsize=12)
                plt.xlabel('$\mu$ [mV/ms]', fontsize=12)
                #plt.ylim([-1, 46])
            if k_j==N_sigma-1:
                plt.legend(loc='best')



    # todo: only plot mu [-1, 4.]
    # todo:          sigma [0.5, 4.5]

    filename = '/home/fabian/fp_models/adex_comparison/quantities_spectral_noref.h5'
    hdf5 = tables.open_file(filename, mode='r')
    r_inf = np.rot90(hdf5.root.r_inf.read())
    V_mean_inf = np.rot90(hdf5.root.V_mean_inf.read())
    mu = hdf5.root.mu.read()
    sigma = hdf5.root.sigma.read()
    hdf5.close()

    mu_vals = [-1, 4]
    sigma_vals = [0.5, 4.5]
    mu_min = np.argmin(np.abs(mu-mu_vals[0]))
    mu_max = np.argmin(np.abs(mu-mu_vals[1]))
    sigma_min = np.argmin(np.abs(sigma-sigma_vals[0]))
    sigma_max = np.argmin(np.abs(sigma-sigma_vals[1]))




    import matplotlib
    from mpl_toolkits.axes_grid1 import AxesGrid

    def shiftedColorMap(cmap, start=0, midpoint=0.5, stop=1.0, name='shiftedcmap'):
        '''
        Function to offset the "center" of a colormap. Useful for
        data with a negative min and positive max and you want the
        middle of the colormap's dynamic range to be at zero

        Input
        -----
          cmap : The matplotlib colormap to be altered
          start : Offset from lowest point in the colormap's range.
              Defaults to 0.0 (no lower ofset). Should be between
              0.0 and `midpoint`.
          midpoint : The new center of the colormap. Defaults to
              0.5 (no shift). Should be between 0.0 and 1.0. In
              general, this should be  1 - vmax/(vmax + abs(vmin))
              For example if your data range from -15.0 to +5.0 and
              you want the center of the colormap at 0.0, `midpoint`
              should be set to  1 - 5/(5 + 15)) or 0.75
          stop : Offset from highets point in the colormap's range.
              Defaults to 1.0 (no upper ofset). Should be between
              `midpoint` and 1.0.
        '''
        cdict = {
            'red': [],
            'green': [],
            'blue': [],
            'alpha': []
        }

        # regular index to compute the colors
        reg_index = np.linspace(start, stop, 257)

        # shifted index to match the data
        shift_index = np.hstack([
            np.linspace(0.0, midpoint, 128, endpoint=False),
            np.linspace(midpoint, 1.0, 129, endpoint=True)
        ])

        for ri, si in zip(reg_index, shift_index):
            r, g, b, a = cmap(ri)

            cdict['red'].append((si, r, r))
            cdict['green'].append((si, g, g))
            cdict['blue'].append((si, b, b))
            cdict['alpha'].append((si, a, a))

        newcmap = matplotlib.colors.LinearSegmentedColormap(name, cdict)
        plt.register_cmap(cmap=newcmap)

        return newcmap




    # norm = mpl.colors.BoundaryNorm(np.linspace(0, 0.14, 256), cm.N)
    # plot r_inf colomap
    plt.subplot(2,2,2)
    plt.title('$r_{\infty}$')
    cm_orig = plt.cm.magma#seismic#rbw#RdBu_r
    # shift cm for 1st plot
    cm = shiftedColorMap(cm_orig, 0, 0.1,1.)

    plt.imshow(r_inf[sigma_min:sigma_max,mu_min:mu_max],interpolation='gaussian',
                cmap=cm, aspect='auto', extent=[mu_vals[0], mu_vals[1], sigma_vals[0], sigma_vals[1]])

    # plt.pcolormesh(mu[mu_min:mu_max], sigma[sigma_min:sigma_max], r_inf[sigma_min:sigma_max,mu_min:mu_max], cmap='spectral')
    plt.colorbar()


    # plot V_inf colormap
    plt.subplot(2,2,4)
    plt.title('$\langle V \\rangle_{\infty}$')
    # shift cm for 2nd plot
    cm = shiftedColorMap(cm_orig, 0, 0.9, 1)
    plt.imshow(V_mean_inf[sigma_min:sigma_max,mu_min:mu_max],
                 interpolation='bicubic', cmap=cm, aspect='auto',
                 extent=[mu_vals[0], mu_vals[1], sigma_vals[0], sigma_vals[1]])
    plt.colorbar()




def plot_quantities(quantities_dict, quantity_names, sigmas_plot):
    
    mu_vals = quantities_dict['mu_vals']
    sigma_vals = quantities_dict['sigma_vals']   
    
    plt.figure()
    plt.suptitle('LN quantities')
    
    mu_lim = [-1, 5]
    inds_mu_plot = [i for i in range(len(mu_vals)) if mu_lim[0] <= mu_vals[i] <= mu_lim[1]]
    inds_sigma_plot = [np.argmin(np.abs(sigma_vals-sig)) for sig in sigmas_plot]
    N_sigma = len(inds_sigma_plot)    
    
    for k_j, j in enumerate(inds_sigma_plot):
        # color    
        rgb = [0, float(k_j)/(N_sigma-1), 0]
        linecolor = rgb
        
        if 'r_ss' in quantity_names:
            ax1 = plt.subplot(4, 2, 1)
            # labels
            if k_j in [0, N_sigma//2, N_sigma-1]:
                siglabel = '$\sigma={0:.3}$ [mV/$\sqrt{{\mathrm{{ms}}}}$]'.format(sigma_vals[j])
                           
            else:
                siglabel = None
    
            plt.plot(mu_vals[inds_mu_plot], quantities_dict['r_ss'][inds_mu_plot,j], 
                     label=siglabel, color=linecolor)
    
            if k_j==0:
                plt.ylabel('$r_{\infty}$ [kHz]')
            if k_j==N_sigma-1:
                plt.legend(loc='best')
        
        if 'V_mean_ss' in quantity_names:
            plt.subplot(4, 2, 2, sharex=ax1)              
            plt.plot(mu_vals[inds_mu_plot], quantities_dict['V_mean_ss'][inds_mu_plot,j], 
                     label=siglabel, color=linecolor)
            if k_j==0:
                plt.ylabel('$\langle V \\rangle_{\infty}$ [mV]')
            
        if 'tau_mu_exp' in quantity_names:
            plt.subplot(4, 2, 3, sharex=ax1)
            plt.plot(mu_vals[inds_mu_plot], quantities_dict['tau_mu_exp'][inds_mu_plot,j], 
                     label=siglabel, color=linecolor)
            if k_j==0:
                plt.ylabel(r'$\tau_{\mu}$ [ms]')
            
        if 'tau_sigma_exp' in quantity_names:
            plt.subplot(4, 2, 4, sharex=ax1)
            plt.plot(mu_vals[inds_mu_plot], quantities_dict['tau_sigma_exp'][inds_mu_plot,j], 
                     label=siglabel, color=linecolor)
            if k_j==0:
                plt.ylabel(r'$\tau_{\sigma}$[ms]')
        
        if 'tau_mu_dosc' in quantity_names:
            plt.subplot(4, 2, 5, sharex=ax1)           
            plt.plot(mu_vals[inds_mu_plot], quantities_dict['tau_mu_dosc'][inds_mu_plot,j], 
                     label=siglabel, color=linecolor)
            if k_j==0:
                plt.ylabel(r'$\tilde{\tau}$ [ms]')
                plt.xlabel('$\mu$ [mV/ms]')
        
        if 'f0_mu_dosc' in quantity_names:
            plt.subplot(4, 2, 6, sharex=ax1)                       
            plt.plot(mu_vals[inds_mu_plot], quantities_dict['f0_mu_dosc'][inds_mu_plot,j], 
                     label=siglabel, color=linecolor)
            if k_j==0:
                plt.ylabel(r'$\tilde{f}$ [kHz]')
                plt.xlabel('$\mu$ [mV/ms]')
                
        if 'tau_sigma_dosc' in quantity_names:
            plt.subplot(4, 2, 7, sharex=ax1)           
            plt.plot(mu_vals[inds_mu_plot], quantities_dict['tau_sigma_dosc'][inds_mu_plot,j], 
                     label=siglabel, color=linecolor)
            if k_j==0:
                plt.ylabel(r'$\tilde{\tau}$ [ms]')
                plt.xlabel('$\mu$ [mV/ms]')
        
        if 'f0_sigma_dosc' in quantity_names:
            plt.subplot(4, 2, 8, sharex=ax1)                       
            plt.plot(mu_vals[inds_mu_plot], quantities_dict['f0_sigma_dosc'][inds_mu_plot,j], 
                     label=siglabel, color=linecolor)
            if k_j==0:
                plt.ylabel(r'$\tilde{f}$ [kHz]')
                plt.xlabel('$\mu$ [mV/ms]')
    
    plt.figure()
    plt.suptitle('LN quantities for bexdox variant')
    for k_j, j in enumerate(inds_sigma_plot):
        # color    
        rgb = [0, float(k_j)/(N_sigma-1), 0]
        linecolor = rgb
        if 'B_mu_bedosc' in quantity_names:
            ax2 = plt.subplot(2, 2, 2)              
            plt.plot(mu_vals[inds_mu_plot], quantities_dict['B_mu_bedosc'][inds_mu_plot,j], 
                     label=siglabel, color=linecolor)
            if k_j==0:
                plt.ylabel('$B$ [...]')
            
        if 'tau1_mu_bedosc' in quantity_names:
            plt.subplot(2, 2, 1, sharex=ax2)
            plt.plot(mu_vals[inds_mu_plot], quantities_dict['tau1_mu_bedosc'][inds_mu_plot,j], 
                     label=siglabel, color=linecolor)
            if k_j==0:
                plt.ylabel(r'$\tau_1$ [ms]')
            
        if 'tau2_mu_bedosc' in quantity_names:
            plt.subplot(2, 2, 3, sharex=ax2)
            plt.plot(mu_vals[inds_mu_plot], quantities_dict['tau2_mu_bedosc'][inds_mu_plot,j], 
                     label=siglabel, color=linecolor)
            if k_j==0:
                plt.ylabel(r'$\tau_2$ [ms]')
                plt.xlabel('$\mu$ [mV/ms]')
        
        if 'f0_mu_bedosc' in quantity_names:
            plt.subplot(2, 2, 4, sharex=ax2)           
            plt.plot(mu_vals[inds_mu_plot], quantities_dict['f0_mu_bedosc'][inds_mu_plot,j], 
                     label=siglabel, color=linecolor)
            if k_j==0:
                plt.ylabel(r'$f_0$ [kHz]')
                plt.xlabel('$\mu$ [mV/ms]')
    plt.show()
            
        
def plot_filters(output_dict, quantities_dict, output_names, params, 
                 mus_plot, sigmas_plot, recalc_filters):
    
    mu_vals = quantities_dict['mu_vals']
    sigma_vals = quantities_dict['sigma_vals']
    f_vals = quantities_dict['freq_vals']
    
    if recalc_filters:
       f_vals = np.arange(0.25, 3000.1, 0.25)/1000
       params_tmp = params.copy()
       params_tmp['freq_vals'] = f_vals
       params_tmp['N_procs'] = 1
       mumod_tmp_dict = {};  sigmamod_tmp_dict = {}
       
    df = f_vals[1] - f_vals[0]
    if any(f_vals<0):
        n = len(f_vals)+1
    else: 
        n = 2*len(f_vals)+1
    dt = 1.0/(df*n)  #ms
    t_vals = np.arange(0,n)*dt    
    tmax = 50
    # note that dt for filter is 1/(df*(2*len(f_vals)+1)) 
    # (factor 2 for appended negative frequencies, +1 for appended zero freq.)
    #  not calculating them explicitly)
    ### note that response to f=0 Hz modulation should be equal to the integral
    ### of the filter in time domain (0 to inf): D_0
    ### r1 for pos. and neg. frequencies are complex conjugates
    # TODO: hint to where can this be seen
    
    inds_mu_plot = [np.argmin(np.abs(mu_vals-mus)) for mus in mus_plot]  
    inds_sigma_plot = [np.argmin(np.abs(sigma_vals-sig)) for sig in sigmas_plot]    
    N_mu = len(inds_mu_plot)    
    N_sigma = len(inds_sigma_plot)
    
    if 'r1_mumod' in output_names:
        plt.figure()
        plt.suptitle('EIF rate response to $\mu$-modulation in in 1/V')
        for k_isig, isig in enumerate(inds_sigma_plot):        
            for k_imu, imu in enumerate(inds_mu_plot):           
                plt.subplot(N_sigma,N_mu,k_isig*N_mu+k_imu+1)                
                
                # scaling factor for rate response (rather than filter) fit:
                drdmu = output_dict['dr_ss_dmu'][imu,isig]
                
                # analytical exponential fit (using asymptotics)
                tau = params['deltaT']*drdmu/output_dict['r_ss'][imu,isig]
                A = 1.0/tau
                exp_fit = drdmu * A*tau/(1.0 + 2*np.pi*1j*f_vals*tau)
                plt.loglog(1000*f_vals[f_vals<=1], 1000*np.abs(exp_fit[f_vals<=1]), 'b--')
                    
                # semi-analytical exponential fit
                if 'tau_mu_exp' in quantities_dict.keys():
                    tau = quantities_dict['tau_mu_exp'][imu,isig]
                    A = 1.0/tau
                    exp_fit = drdmu * A*tau/(1.0 + 2*np.pi*1j*f_vals*tau)
                    plt.loglog(1000*f_vals[f_vals<=1], 1000*np.abs(exp_fit[f_vals<=1]), 'c')                
                
                # semi-analytical damped oscillator fit
                if 'tau_mu_dosc' in quantities_dict.keys():
                    tau = quantities_dict['tau_mu_dosc'][imu,isig]
                    f0 = quantities_dict['f0_mu_dosc'][imu,isig]
                    A = (1.0 + (2*np.pi*f0*tau)**2)/tau
                    dosc_fit = drdmu * A*tau/2 * \
                               ( 1.0/(1.0 + 2*np.pi*1j*tau*(f_vals-f0)) + \
                                 1.0/(1.0 + 2*np.pi*1j*tau*(f_vals+f0)) )
                    plt.loglog(1000*f_vals[f_vals<=1], 1000*np.abs(dosc_fit[f_vals<=1]), 'm')
                                                
                # semi-analytical bi-exp, damped oscillator fit
                if 'tau1_mu_bedosc' in quantities_dict.keys():
                    tau1 = quantities_dict['tau1_mu_bedosc'][imu,isig]
                    tau2 = quantities_dict['tau2_mu_bedosc'][imu,isig]
                    f0 = quantities_dict['f0_mu_bedosc'][imu,isig]
                    B = quantities_dict['B_mu_bedosc'][imu,isig]
                    A = 1.0/tau1 - B*tau2/((1.0 + (2*np.pi*f0*tau2)**2)*tau1)
                    bexdos_fit = drdmu * ( A*tau1/(1.0 + 2*np.pi*1j*f_vals*tau1) + \
                                 B*tau2/2 * ( 1.0/(1.0 + 2*np.pi*1j*tau2*(f_vals-f0)) + \
                                              1.0/(1.0 + 2*np.pi*1j*tau2*(f_vals+f0)) ) )
                    plt.loglog(1000*f_vals[f_vals<=1], 1000*np.abs(bexdos_fit[f_vals<=1]), 'g')

                # rate response
                if recalc_filters:
                    tmp_dict = {};  tmp_output_name = ['r1_mumod']
                    tmp_dict = EIF_steadystate_and_linresponse([mu_vals[imu]],
                                                               [sigma_vals[isig]],
                                                               params_tmp, tmp_dict,
                                                               tmp_output_name)
                    r1_mumod = tmp_dict['r1_mumod'][0,0,:]
                    mumod_tmp_dict[imu,isig] = r1_mumod
                else:                               
                    r1_mumod = output_dict['r1_mumod'][imu,isig,:]
                plt.loglog(1000*f_vals[f_vals<=1], 1000*np.abs(r1_mumod[f_vals<=1]), 'k')
                
                plt_min = np.min(1000*np.abs(r1_mumod[f_vals<=1]))
                plt_max = np.max(1000*np.abs(r1_mumod[f_vals<=1]))
                plt.ylim([plt_min, plt_max])
                plt.xlim([1000*f_vals[0], 1000])
#                if count<=ncols
#                  title('$|\hat{R}_{\mu}(f)|$','Interpreter','latex');
#                end
#                if mod(count,ncols)==1
#                  ylabel(['$\sigma =$' num2str(presimdata.sigmarange(is))], ...
#                          'Interpreter','latex');
#                end
#                if mod(count,ncols)==1 && count>ncols*(nrows-1)
#                  ylabel(['$\sigma =$' num2str(presimdata.sigmarange(is)) ...
#                          ' mV/$\sqrt{\mathrm{ms}}$'],'Interpreter','latex');
#                end
#                if count>ncols*(nrows-1)
#                  xlabel(['$\mu =$' num2str(presimdata.Irange(iI)/taum)], ...
#                          'Interpreter','latex');
#                  set(gca,'xlim',[0 1000],'xtick',...
#                    [1 10 100 1000],'YTickLabel',[]);   
#                else
#                  set(gca,'xlim',[0 1000],'xtick',...
#                    [1 10 100 1000],'xticklabel',[],'yticklabel',[]);
#                end
#                if count>ncols*(nrows-1) && mod(count,ncols)==0
#                  xlabel(['$\mu =$' num2str(presimdata.Irange(iI)/taum) ...
#                          ' mV/ms'],'Interpreter','latex');  
#                end
                
    if 'r1_sigmamod' in output_names:
        plt.figure()
        plt.suptitle('EIF rate response to $\sigma$-modulation in 1/(V*sqrt(ms))')
        for k_isig, isig in enumerate(inds_sigma_plot):        
            for k_imu, imu in enumerate(inds_mu_plot):           
                plt.subplot(N_sigma,N_mu,k_isig*N_mu+k_imu+1)                
                
                # scaling factor for rate response (rather than filter) fit:
                drdsigma = output_dict['dr_ss_dsigma'][imu,isig]
                   
                # semi-analytical exponential fit
                if 'tau_sigma_exp' in quantities_dict.keys():
                    if drdsigma>0:  # otherwise we use delta filter  
                        tau = quantities_dict['tau_sigma_exp'][imu,isig]
                        A = 1.0/tau
                        exp_fit = drdsigma * A*tau/(1.0 + 2*np.pi*1j*f_vals*tau)
                        plt.loglog(1000*f_vals[f_vals<=1], 1000*np.abs(exp_fit[f_vals<=1]), 'c')                
                
                # semi-analytical damped oscillator fit
                if 'tau_sigma_dosc' in quantities_dict.keys():
                    if drdsigma>0:  # otherwise we use delta filter  
                        tau = quantities_dict['tau_sigma_dosc'][imu,isig]
                        f0 = quantities_dict['f0_sigma_dosc'][imu,isig]
                        A = (1.0 + (2*np.pi*f0*tau)**2)/tau
                        dosc_fit = drdsigma * A*tau/2 * \
                                   ( 1.0/(1.0 + 2*np.pi*1j*tau*(f_vals-f0)) + \
                                     1.0/(1.0 + 2*np.pi*1j*tau*(f_vals+f0)) )
                        plt.loglog(1000*f_vals[f_vals<=1], 1000*np.abs(dosc_fit[f_vals<=1]), 'm')

                # rate response
                if recalc_filters:
                    tmp_dict = {};  tmp_output_name = ['r1_sigmamod']
                    tmp_dict = EIF_steadystate_and_linresponse([mu_vals[imu]],
                                                               [sigma_vals[isig]],
                                                               params_tmp, tmp_dict,
                                                               tmp_output_name)
                    r1_sigmamod = tmp_dict['r1_sigmamod'][0,0,:] 
                    sigmamod_tmp_dict[imu,isig] = r1_sigmamod
                else:                               
                    r1_sigmamod = output_dict['r1_sigmamod'][imu,isig,:]              
                plt.loglog(1000*f_vals[f_vals<=1], 1000*np.abs(r1_sigmamod[f_vals<=1]), 'k')
                plt_min = np.min(1000*np.abs(r1_sigmamod[f_vals<=1]))
                plt_max = np.max(1000*np.abs(r1_sigmamod[f_vals<=1]))
                plt.ylim([plt_min, plt_max])
                plt.xlim([1000*f_vals[0], 1000])

    if 'ifft_r1_mumod' in output_names:
        plt.figure()
        plt.suptitle('linear filter for $\mu$ (unnormalized) in kHz/V')
        inds = t_vals<=tmax
        for k_isig, isig in enumerate(inds_sigma_plot):        
            for k_imu, imu in enumerate(inds_mu_plot):           
                plt.subplot(N_sigma,N_mu,k_isig*N_mu+k_imu+1)                
                
                # scaling factor for rate response (rather than filter) fit:
                drdmu = output_dict['dr_ss_dmu'][imu,isig]
                
                # analytical exponential fit (using asymptotics)
                tau = params['deltaT']*drdmu/output_dict['r_ss'][imu,isig]
                A = 1.0/tau
                exp_fit = drdmu * A*np.exp(-t_vals/tau)
                plt.plot(t_vals[inds], 1000*exp_fit[inds], 'b--')
                   
                # semi-analytical exponential fit
                if 'tau_mu_exp' in quantities_dict.keys():
                    tau = quantities_dict['tau_mu_exp'][imu,isig]
                    A = 1.0/tau
                    exp_fit = drdmu * A*np.exp(-t_vals/tau)
                    plt.plot(t_vals[inds], 1000*exp_fit[inds], 'c')                
                
                # semi-analytical damped oscillator fit
                if 'tau_mu_dosc' in quantities_dict.keys():
                    tau = quantities_dict['tau_mu_dosc'][imu,isig]
                    f0 = quantities_dict['f0_mu_dosc'][imu,isig]
                    A = (1.0 + (2*np.pi*f0*tau)**2)/tau
                    dosc_fit = drdmu * A*np.exp(-t_vals/tau)*np.cos(2*np.pi*f0*t_vals)
                    plt.plot(t_vals[inds], 1000*dosc_fit[inds], 'm')
                
                # semi-analytical bi-exp, damped oscillator fit
                if 'tau1_mu_bedosc' in quantities_dict.keys():
                    tau1 = quantities_dict['tau1_mu_bedosc'][imu,isig]
                    tau2 = quantities_dict['tau2_mu_bedosc'][imu,isig]
                    f0 = quantities_dict['f0_mu_bedosc'][imu,isig]
                    B = quantities_dict['B_mu_bedosc'][imu,isig]
                    A = 1.0/tau1 - B*tau2/((1.0 + (2*np.pi*f0*tau2)**2)*tau1)
                    bexdos_fit = drdmu * ( A*np.exp(-t_vals/tau1) + \
                                 B*np.exp(-t_vals/tau2)*np.cos(2*np.pi*f0*t_vals) )
                    plt.plot(t_vals[inds], 1000*bexdos_fit[inds], 'g')

                # ifft of rate response
                if recalc_filters:
                    r1_mumod = mumod_tmp_dict[imu,isig]
                else:
                    r1_mumod = output_dict['r1_mumod'][imu,isig,:]  #1/mV
                # linear filter (ifft of rate response)
                # get mumod rate response in the right shape so that ifft can be applied
                # for info on the reshaping above, see numpy documentation on ifft 
                # (numpy.fft.ifft)
                if any(f_vals<0):
                    r1_reshaped = np.concatenate([np.array([drdmu]), r1_mumod])
                else:
                    r1_reshaped = np.concatenate([np.array([drdmu]), r1_mumod, 
                                                  np.flipud(np.real(r1_mumod)) - \
                                                  1j*np.flipud(np.imag(r1_mumod))])
                mu_filter_unnorm = np.fft.ifft(r1_reshaped)/dt  #kHz/mV
                plt.plot(t_vals[inds], 1000*mu_filter_unnorm[inds], 'k')
                
                #plt_min = np.min(mu_filter_unnorm[inds])
                plt_max = np.max(1000*mu_filter_unnorm[inds])
                plt.ylim([-0.12*plt_max, 1.2*plt_max]) 
                plt.xticks([0, tmax])        

      
    if 'ifft_r1_sigmamod' in output_names:
        plt.figure()
        plt.suptitle('linear filter for $\sigma$ (unnormalized) in kHz/(V*sqrt(ms))')
        inds = t_vals<=tmax
        for k_isig, isig in enumerate(inds_sigma_plot):        
            for k_imu, imu in enumerate(inds_mu_plot):           
                plt.subplot(N_sigma,N_mu,k_isig*N_mu+k_imu+1)                
                
                # scaling factor for rate response (rather than filter) fit:
                drdsigma = output_dict['dr_ss_dsigma'][imu,isig]
                #print mu_vals[imu], sigma_vals[isig], drdsigma  #TEMP
                
                ## analytical exponential fit (using asymptotics)
                #tau = params['deltaT']**2*drdsigma/ \
                #      (output_dict['r_ss'][imu,isig] * sigma_vals[isig])
                #A = 1.0/tau
                #exp_fit = drdsigma * A*np.exp(-t_vals/tau)
                #plt.plot(t_vals[inds], 1000*exp_fit[inds], 'b--')
                
                # semi-analytical exponential fit
                if 'tau_sigma_exp' in quantities_dict.keys():
                    if drdsigma>0:  # otherwise we use delta filter   
                        tau = quantities_dict['tau_sigma_exp'][imu,isig]
                        A = 1.0/tau
                        exp_fit = drdsigma * A*np.exp(-t_vals/tau)
                        plt.plot(t_vals[inds], 1000*exp_fit[inds], 'c')                
                
                # semi-analytical damped oscillator fit
                if 'tau_sigma_dosc' in quantities_dict.keys():
                    if drdsigma>0:  # otherwise we use delta filter   
                        tau = quantities_dict['tau_sigma_dosc'][imu,isig]
                        f0 = quantities_dict['f0_sigma_dosc'][imu,isig]
                        A = (1.0 + (2*np.pi*f0*tau)**2)/tau
                        dosc_fit = drdsigma * A*np.exp(-t_vals/tau)*np.cos(2*np.pi*f0*t_vals)
                        plt.plot(t_vals[inds], 1000*dosc_fit[inds], 'm')

                # ifft of rate response
                if recalc_filters:
                    r1_sigmamod = sigmamod_tmp_dict[imu,isig]
                else:
                    r1_sigmamod = output_dict['r1_sigmamod'][imu,isig,:]  #1/mV
                # linear filter (ifft of rate response)
                # get sigmamod rate response in the right shape so that ifft can be applied
                # for info on the reshaping above, see numpy documentation on ifft 
                # (numpy.fft.ifft)
                if any(f_vals<0):
                    r1_reshaped = np.concatenate([np.array([drdsigma]), r1_sigmamod])
                else:
                    r1_reshaped = np.concatenate([np.array([drdsigma]), r1_sigmamod, 
                                                  np.flipud(np.real(r1_sigmamod)) - \
                                                  1j*np.flipud(np.imag(r1_sigmamod))])
                sigma_filter_unnorm = np.fft.ifft(r1_reshaped)/dt  #kHz/mV
                plt.plot(t_vals[inds], 1000*sigma_filter_unnorm[inds], 'k') 
                plt_max = np.max(1000*sigma_filter_unnorm[inds])
                plt.ylim([-0.16*plt_max, 1.2*plt_max]) 
                plt.xticks([0, tmax])
    plt.show()            
                
                
                
## -----------------------------------------------------------------------------
#def FPssandmod_P(mu0, sigma0, **kwargs):
#    pdict = kwargs['pdict']
#    V_vec = np.arange(pdict['V_lb'],pdict['VT']+pdict['dV']/2,pdict['dV'])
#    kre = np.argmin(np.abs(V_vec-pdict['Vr']))
#    #mu0 *= pdict['Cs']/pdict['Cp']
#    #sigma0 *= pdict['Cs']/pdict['Cp']
#    r0, p0, q0 = FPss_P(V_vec, kre, pdict['Cp'], pdict['Gp'], mu0, sigma0, 
#                        pdict['VT'], pdict['Vr'])
#    dp0dV = 2/sigma0**2 * ((mu0-V_vec*pdict['Gp']/pdict['Cp'])*p0 - q0) 
#    # check output (optional)  
##    plt.figure()
##    ax = plt.subplot(211)
##    plt.plot(V_vec,p0,'k')
##    plt.xlim([-0.03, 0.01])
##    plt.subplot(212, sharex=ax)
##    plt.plot(V_vec,q0,'k')
##    plt.xlim([-0.03, 0.01])
#    if kwargs['only_ss']:
#        return p0, r0
#    else:           
#        w_vec = 2*np.pi*pdict['f_FP1_vec']
#        if kwargs['method'] == 'mumod':
#            mu1 = 1e-4;  inhom = mu1*p0;
#            r1_vec = FPmod_P(V_vec, kre, inhom, pdict['Cp'], pdict['Gp'], 
#                               mu0, sigma0, pdict['VT'], pdict['Vr'], w_vec)
#            r1_vec /= mu1
#        elif kwargs['method'] == 'sig2mod':
#            sig21 = 1e-4;  inhom = -sig21*dp0dV/2;
#            r1_vec = FPmod_P(V_vec, kre, inhom, pdict['Cp'], pdict['Gp'], 
#                                 mu0, sigma0, pdict['VT'], pdict['Vr'], w_vec)
#            r1_vec /= sig21                
#        return p0, r0, r1_vec
#
#@numba.njit
#def FPss_P(V, kre, C, G, mu, sigma, VT, Vr):
#    taum = C/G
#    dV = V[1]-V[0]
#    sig2 = sigma**2
#    n = len(V)
##    kre = np.argmin(np.abs(V-Vr))
#    p0 = np.zeros(n);  q0 = np.ones(n);
#    for k in xrange(n-1, kre, -1):
#        F = 2*(V[k]/taum-mu)/sig2 
#        A = np.exp(dV*F)
#        if not F==0.0:
#            p0[k-1] = p0[k] * A + (A-1.0)/F * 2/sig2 
#        else:
#            p0[k-1] = p0[k] * A + dV * 2/sig2  
#        q0[k-1] = 1.0    
#    for k in xrange(kre, 0, -1):  
#        F = 2*(V[k]/taum-mu)/sig2 
#        A = np.exp(dV*F)
#        p0[k-1] = p0[k] * A
#        q0[k-1] = 0.0
#    p0sum = np.sum(p0)   
#    r0 = 1.0/(dV*p0sum)
#    p0 *= r0;  q0 *= r0;
#    return r0, p0, q0
#    
#@numba.njit
#def FPmod_P(V, kre, inhom, C, G, mu, sigma, VT, Vr, w_vec):
#    taum = C/G
#    dV = V[1]-V[0]
#    sig2 = sigma**2
#    n = len(V)
##    kre = np.argmin(np.abs(V-Vr))
#    r1_vec = 1j*np.ones(len(w_vec))   
#    for iw in range(len(w_vec)):
#        q1a = 1.0 + 0.0*1j;  p1a = 0.0*1j;  q1b = 0.0*1j;  p1b = 0.0*1j;
#        fw = dV*1j*w_vec[iw]
#        for k in xrange(n-1, 0, -1):
#            F = 2*(V[k]/taum-mu)/sig2
#            A = np.exp(dV*F)
#            if not k==kre+1:
#                q1a_new = q1a + fw*p1a
#            else:
#                q1a_new = q1a + fw*q1a - 1.0
#            if not F==0.0:    
#                p1a_new = p1a * A + (A-1.0)/F * 2/sig2 * q1a
#                p1b_new = p1b * A + (A-1.0)/F * 2/sig2 * (q1b - inhom[k])
#            else:
#                p1a_new = p1a * A + dV * 2/sig2 * q1a
#                p1b_new = p1b * A + dV * 2/sig2 * (q1b - inhom[k])
#            q1b += fw*p1b
#            q1a = q1a_new;  p1a = p1a_new;  p1b = p1b_new;   
#        r1_vec[iw] = -q1b/q1a
#    return r1_vec    