# -*- coding: utf-8 -*-

# script for computing the steady-state, and the first order rate response 
# of an exponential integrate-and-fire neuron subject to white noise input
# to modulations of the input moments and associated quantities 
# for linear-nonlinear cascade rate models, on a rectangle of baseline 
# input moments (mu, sigma) -- written by Josef Ladenbauer in 2016 

# use the following in IPython for qt plots: %matplotlib qt

from params import get_params
import cascade_precalc_functions as cf
import numpy as np
from collections import OrderedDict
import os

folder = os.path.dirname(os.path.realpath(__file__)) # store files in the same directory as the script itself
output_filename = 'EIF_output_for_cascade_noref_speedtest.h5'
quantities_filename = 'quantities_cascade_noref_speedtest.h5'
#quantities_filename = 'quantities_cascade_noref_TEMP2.h5'  # contains also sigmod dosc fits
load_EIF_output = False
load_quantities = False
compute_EIF_output = True
compute_quantities = True
save_rate_mod = False
save_EIF_output = True
save_quantities = True
plot_filters = False
plot_quantities = False

# TODO: recalc output and quantities with 241x46 grid (instead of 261x46) and save: 
# done for Tref=0 (w/o LN_bexdos)

# PREPARING --------------------------------------------------------------------
params = get_params()

params['t_ref'] = 0.0

N_mu_vals = 241 #def.: 261 mu grid points -- from -1.5 to 5 with spacing 0.025 
N_sigma_vals = 46 #def.: 46 #sigma grid points  -- from 0.5 to 5 with spacing 0.1

d_freq = 0.25 # Hz, def.: 0.25
d_V = 0.01 # mV note: Vr should be a grid point

# For Tref<=1.5 the following limits are reasonable:
mu_vals = np.linspace(-1.0, 5.0, N_mu_vals)  #def.: np.linspace(-1.0, 5.0, N_mu_vals)
sigma_vals = np.linspace(0.5, 5.0, N_sigma_vals)  #def.: np.linspace(0.5, 5.0, N_sigma_vals)
# EIF_steadystate_and_linresponse results are not faithful for sigma<0.5 and 
# for mu<-1 when sigma is small (sigma=0.5) 
# --> this may be improved by using a slight refinement in the backwards integration:
# using evaluations of V at k-1/2 instead of k (current version), which would match then 
# with the spectral calculation scheme and (therefore) should work better for smaller mu 

# choose background mu and sigma values for filter visualization 
#num_mus_plot = np.min([np.size(mu_vals),8])    
#mus_plot = np.linspace(mu_vals[0], mu_vals[-1], num_mus_plot)    
#mus_plot = [-0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
# for paper: mu-mod [-0.5, 1.5, 3.0], sigma-mod [-0.5, 0.0, 1.5]
mus_plot = [-0.5, 1.5, 3.0]
#num_sigmas_plot = np.min([np.size(sigma_vals),4])  
#sigmas_plot = np.linspace(sigma_vals[0], sigma_vals[-1], num_sigmas_plot)      
sigmas_plot = [1.5, 3.5] #[1.2, 2.4, 3.6]  #[0.5, 1.5, 2.5, 3.5]    
# choose background sigma values for quantity visualization        
sigmas_quant_plot = np.arange(0.5, 4.501, 0.2) #[0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
#sigmas_quant_plot = sigma_vals 

# some more precalc parameters
params['N_procs'] = 10 #multiprocessing.cpu_count() # no of parallel processes (not used in calc_cascade_quantities)
params['V_vals'] = np.arange(params['Vlb'],params['Vcut']+d_V/2,d_V)
params['freq_vals'] = np.arange(d_freq, 1000+d_freq/2, d_freq)/1000  # kHz
params['d_mu'] = 1e-5 # mV/ms
params['d_sigma'] = 1e-5 # mV/sqrt(ms)

EIF_output_dict = OrderedDict()
LN_quantities_dict = OrderedDict()

EIF_output_names = ['r_ss', 'dr_ss_dmu', 'dr_ss_dsigma', 'V_mean_ss',
                    'r1_mumod', 'r1_sigmamod', 
                    #'peak_abs_r1_mumod', 'f_peak_abs_r1_mumod',  #not needed and not impl. a.t.m.
                    'peak_real_r1_mumod', 'f_peak_real_r1_mumod', 
                    'peak_imag_r1_mumod', 'f_peak_imag_r1_mumod']
                    #'peak_real_r1_sigmamod', 'f_peak_real_r1_sigmamod',  #not needed
                    #'peak_imag_r1_sigmamod', 'f_peak_imag_r1_sigmamod']  #not needed
LN_quantity_names = ['r_ss', 'V_mean_ss', 
                     'tau_mu_exp', 'tau_sigma_exp',
                     'tau_mu_dosc', 'f0_mu_dosc']
                     #'tau_sigma_dosc', 'f0_sigma_dosc']  #not needed
                     #'B_mu_bedosc', 'tau1_mu_bedosc', 'tau2_mu_bedosc', 'f0_mu_bedosc']  #not needed

plot_EIF_output_names = ['r1_mumod', 'ifft_r1_mumod', 'r1_sigmamod', 'ifft_r1_sigmamod']                     
plot_quantitiy_names =  ['r_ss', 'V_mean_ss']#,
                         #'tau_mu_exp', 'tau_sigma_exp',
                         #'tau_mu_dosc', 'f0_mu_dosc']
                         #'tau_sigma_dosc', 'f0_sigma_dosc']  #not needed
                         #'B_mu_bedosc', 'tau1_mu_bedosc', 'tau2_mu_bedosc', 'f0_mu_bedosc']  #not needed

if __name__ == '__main__':

    # LOADING ----------------------------------------------------------------------
    if load_EIF_output:
        cf.load(folder+'/'+output_filename, EIF_output_dict,
                EIF_output_names + ['mu_vals', 'sigma_vals', 'freq_vals'], params)
        # optional shortcuts
        mu_vals = EIF_output_dict['mu_vals']
        sigma_vals = EIF_output_dict['sigma_vals']
        freq_vals = EIF_output_dict['freq_vals']       
     
    if load_quantities:
        cf.load(folder+'/'+quantities_filename, LN_quantities_dict,
                LN_quantity_names + ['mu_vals', 'sigma_vals', 'freq_vals'], params)
        # optional shortcuts
        mu_vals = LN_quantities_dict['mu_vals']
        sigma_vals = LN_quantities_dict['sigma_vals']
        freq_vals = LN_quantities_dict['freq_vals'] 
    
        #print params['t_ref']  #TEMP
    
    # COMPUTING --------------------------------------------------------------------
#    if compute_EIF_output:                  
#        EIF_output_dict = cf.EIF_steadystate_and_linresponse(mu_vals,sigma_vals,params,
#                                                             EIF_output_dict,EIF_output_names)
#                                                     
#                                                             
#    if compute_quantities:                                                     
#        LN_quantities_dict = cf.calc_cascade_quantities(mu_vals,sigma_vals,params,
#                                                        EIF_output_dict,LN_quantities_dict,
#                                                        LN_quantity_names)
#        # takes a few minutes (~10 for single proc.)
                                                                                                            
    # NEW: combined EIF_steadystate_and_linresponse and calc_cascade_quantities, 
    # calculate all mu_vals per process and opt for not storing rate responses (filters)
    # in order to save memory!  
    if compute_EIF_output and compute_quantities:
        EIF_output_dict, LN_quantities_dict = cf.calc_EIF_output_and_cascade_quants(
                                                 mu_vals, sigma_vals, params, 
                                                 EIF_output_dict, EIF_output_names, save_rate_mod,
                                                 LN_quantities_dict, LN_quantity_names)                                                                                                        
    # takes 40 min. for default mu,sigma,freq grid and N_procs=8 (risha)
    # takes ~125 min. for default mu,sigma,freq grid and N_procs=2 (lenovo laptop)
    # takes 1 h for default mu,sigma,freq grid and N_procs=10 (merope)                                             
                                                 
    # SAVING -----------------------------------------------------------------------                                   
    if save_EIF_output:
        cf.save(folder+'/'+output_filename, EIF_output_dict, params) 
        print('saving EIF output done')
        
    if save_quantities:
        cf.save(folder+'/'+quantities_filename, LN_quantities_dict, params) 
        print('saving LN quantities done')
         
    # PLOTTING ---------------------------------------------------------------------
    if plot_filters:
        recalc_filters = True
        cf.plot_filters(EIF_output_dict, LN_quantities_dict, plot_EIF_output_names, 
                        params, mus_plot, sigmas_plot, recalc_filters)
        
    if plot_quantities: 
    #    cf.plot_quantities(LN_quantities_dict, plot_quantitiy_names, sigmas_quant_plot)
        cf.plot_quantities_forpaper(LN_quantities_dict, plot_quantitiy_names, 
                                    sigmas_quant_plot, mus_plot, sigmas_plot)
    

plot_addon_fabian = False
if plot_addon_fabian:
    cf.fig5_addon_Fabian(LN_quantities_dict, plot_quantitiy_names,
                                sigmas_quant_plot, mus_plot, sigmas_plot)

