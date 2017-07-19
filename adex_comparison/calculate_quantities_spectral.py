# -*- coding: utf-8 -*-

# script for computing the spectrum and associated quantities of the 
# Fokker-Planck operator for the exponential integrate-and-fire neuron model 
# on a rectangle of input mean and standard deviation (mu, sigma)
#
# the SpectralSolver class (written by Moritz Augustin in 2016-2017) is used here 
#
# author: Moritz Augustin <augustin@ni.tu-berlin.de>

# Please cite the publication which has introduced the solver if you want to use it: 
#    Augustin, Ladenbauer, Baumann, Obermayer (2017) PLOS Comput Biol
    
import sys
sys.path.insert(1, '..')
from methods_spectral import SpectralSolver, spectrum_enforce_complex_conjugation, \
                                          quantities_postprocess, inner_prod, \
                                          plot_raw_spectrum_sigma, plot_raw_spectrum_eigvals, \
                                          plot_quantities_eigvals, plot_quantities_real, \
                                          plot_quantities_complex, plot_quantities_composed
from params import get_params
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from collections import OrderedDict 
import os
from multiprocessing import cpu_count
import matplotlib
matplotlib.rcParams['text.usetex'] = True



# PARAMETERS

# default params
params = get_params()

# file containing the full spectrum (all eigenvalues), 
# the two dominant eigenvalues as well as all further quantities for 
# a rectangle of input parameter values for the mean and std dev (mu, sigma)
filename = 'quantities_spectral.h5'

folder = os.path.dirname(os.path.realpath(__file__)) # store files in the same directory as the script itself


# eigenmodes and input grid
# note that the following parameters will be overriden if the computation is not performed due to loading from file
N_eigvals = 10 # 10 # no of eigenvalues (enumeration w.r.t. the smallest mu value)
               # for up to sigma=5mV/sqrt(ms) 9 eigvals would be sufficient 
               # to extract the first two  dominant modes on the mu-sigma grid 
               # this is due to the high noise situation: diffusive modes 
               # are dominant for for small mean mu, for larger mean input 
               # then regular eigenvalues are dominant (but the latter is only 
               # the for example 9-th important mode for weak mean input)
N_mu = 461   # mu grid points -- s.t. -1.5 to 10 is spaced with distance 0.025 
N_sigma = 46 # sigma grid points  -- s.t. 0.5 to 5 is spaced with distance 0.1
N_procs = cpu_count() # no of parallel processes for the spectrum/quantity computation

# create input parameter grid (here only the mu and sigma arrays)
mu = np.linspace(-1.5, 5., N_mu)  # mu (mean input) grid. note that a value of -1 
                                  # for mu has been shown to be not small enough 
                                  # for having only real eigvals (for V_lb=-200)
sigma = np.linspace(0.5, 5., N_sigma) # sigma (input noise intensity) grid. note that 
                                      # sigma_min=0.25 is too small as it 
                                      # shows very sensitive numerical behavior 
                                      # (eigenvalue pairs' real parts are very close by)

# note that here we use mV and ms units both for (neuronal and input) parameters as well as for computed 
# quantities (eigenvalues, r_inf etc.)

# the solver initialization uses the smallest mu value (which is assumed to be chosen 
# so as to lead to purely real eigenvalues). there it (densely) evaluates the 
# eigenflux at the lower bound on the following real grid 
# attention: this grid  has to be fine enough, otherwise zeroes might be overlooked
# while lambda_1,...,lambda_{N_eigvals} should lie within this interval here our 
# code automatically enlarges the grid to finally get hold N_eigvals modes
eigenval_init_real_grid = np.linspace(-5, -1e-4, 5000) 

# TOGGLES/FLAGS FOR SAVING/LOADING/COMPUTING/POSTPROCESSING
save_spec = True # save (and overwrite file) if spectrum computation or postprocessing happened
load_spec = True # loading spectrum from file skips computation unless loading fails
compute_spec = False # computes if not loaded
postprocess_spectrum = False # enforce complex conjugation

save_quant = True # save (and overwrite file) if quantity computation or postprocessing happened
load_quant = True # loading quantities from file skips quantity calculation unless loading fails
compute_quant = False # whether to compute quantities at all 
postprocess_quant = False # remove numerical artefacts from quantities
obtain_fluxlb = True # whether to load or compute if not in file lambda -> q(V_lb) for smallest mu

load_params = True # when loading spectrum or quantities the params dict values gets updated from file


# PLOTTING PARAMETERS

plot_paper_quantities = True            # the visualization used for Figure 7 of Augustin et al 2017

plot_full_spectrum_sigma = False        # the spectrum (mu, eigenvalue index) visualized with sigma running over subplots
                                        # note that this is also contained in plot_paper_quantities
                                        
plot_full_spectrum_eigvals = False      # the spectrum (mu, sigma) visualized with eigenvalue index running over subplots

plot_quantities = ['eigvals', 'real', 'composed'] # which quantitie types to plot 
                                                  # additionally, choose no or any from 
                                                  # ['eigvals', 'real', 'complex', 'composed']

plot_validation = True # plot available quantities that were calculated by 
                       # another method (here only the stationary quantities, 
                       # i.e., the steady state spike rate and mean membrane 
                       # potential obtained with the code of the cascade models 
                       # that is based on scripts from [Richardson 2007, Phys Rev E])

plot_real_inits = False # plot the eigenfluxes at the lower bound for the densely 
                        # evaluated grid of real eigenvalue candidates lambda 
                        # (at a sufficiently small mean input) 
                        # note that this is also contained in plot_paper_quantities

plot_eigenfunctions = False # plot some eigenfunctions
                            # note that this is also contained in plot_paper_quantities


# parameters shared by several plotting parts of this file
sigma_smaller = 1.5 # smaller sigma value (used when plotting for two sigma values)
sigma_larger = 3.5  # larger sigma value (used when plotting for two sigma values)
mu_min = -1.0 # to be compatible with cascade models (only used for visualizations)
colormap_sigma = 'winter' # colormap used for the (plotting parts except those plot_paper_quantities)
no_sigma_quantities_plot = 'all' # min(4, len(sigma)) # which sigma values to plot (not used in plot_paper_quantities)
# more plotting parameters are at the respective plotting function calls further below


# I. SPECTRUM OF FP OPERATOR -- COMPUTATION/LOADING/POSTPROCESSING
# calculate (or load) the spectrum, i.e., all nonstationary eigenvalues as indexed 
# 1, 2, ..., N_eigvals where the index refers to the ordering w.r.t. 
# increasingly negative real part at the smallest mean input mu 

specsolv = SpectralSolver(params)

quantities_dict = OrderedDict() # for inserting quantities in order in hdf5 file 

print('filename={}'.format(filename))

# SPECTRUM LOADING
spec_loaded = False
if load_spec:
    specsolv.load_quantities(folder+'/'+filename, quantities_dict,
                                        quantities=['lambda_all', 'mu', 'sigma'], 
                                        load_params=load_params)
    mu = quantities_dict['mu']
    sigma = quantities_dict['sigma']
    lambda_all = quantities_dict['lambda_all']
    N_eigvals = lambda_all.shape[0]
    N_mu = lambda_all.shape[1]
    N_sigma = lambda_all.shape[2]
    
    spec_loaded = True
    


# SPECTRUM COMPUTATION
spec_computed = False
if compute_spec and not spec_loaded:

    # do the actual computation for N_eigvals eigenvalues on the mu sigma 
    # rectangle via the following method callinitialized with all eigenvalues 
    # found by dense evaluation of the eigenvalue candidate array eigenval_init_real_grid
    lambda_all = specsolv.compute_eigenvalue_rect(mu, sigma, N_eigvals, 
                                                  eigenval_init_real_grid, N_procs=N_procs)
    
    quantities_dict['lambda_all'] = lambda_all
    quantities_dict['mu'] = mu
    quantities_dict['sigma'] = sigma
    
    # saving spectrum
    if save_spec:
    
        specsolv.save_quantities(folder+'/'+filename, quantities_dict) 
    
        print('saving spectrum after computing done.')

# POSTPROCESSING after the raw spectral solver output:
# enforcing complex conjugation and selecting pointwise the two dominant eigenvalues

if postprocess_spectrum:
    
    
    # enforcing complex conjugate pairs of eigenvalues (from the crossing to the right)
    # (since from the iterative solution procedure the sign of the imaginary part is random)
    # it is furthermore not guaranteed that eigenvalue curves corresponding to complex conjugate pairs are following 
    # each other, e.g. k=0 could correspond to a regular mode, k=1 could be diffusive (purely real), 
    # k=2 could be the complex conjugate to k=0 (also regular)

    conjugation_first_imag_negative = False # sign of imaginary part for first of the conjugate couple
    
    print('enforcing complex conjugated pairs for lambda_all')
    spectrum_enforce_complex_conjugation(lambda_all, mu, sigma, 
                                  tolerance_conjugation=params['tolerance_conjugation'], # sec units: # tolerance_conjugation=1e-1,
                                  conjugation_first_imag_negative=conjugation_first_imag_negative)
    # now lambda_all satisfies the above property of complex conjugation
    
        
    
    # select lambda_1 and lambda_2 -- the first two dominante eigenvalues for each mu, sigma
    # for neurons with lower bound == reset this is simply taking the first two of lambda_all
    # here the situation is more complicated due to the switching between dominant diffusive and regular modes
    # as explained in the paper [Augustin et al 2017, PLOS Comput Biol]
    lambda_1 = np.zeros((N_mu, N_sigma), dtype=np.complex)
    lambda_2 = np.zeros_like(lambda_1)
    
    print('extracting lambda_1 and lambda_2 from raw spectrum lambda_all')
    for i in range(N_mu):
        for j in range(N_sigma):
            m_dominant_1 = np.argmin(np.abs(lambda_all[:, i, j].real)) #smallest absolute real part == slowest mode
            lambda_1[i, j] = lambda_all[m_dominant_1, i, j]
            
            # if the first eigenvalue is complex the second is defined as its complex conjugate            
            # additionally we ensure that conjugation_first_imag_negative holds
            is_complex_lambda1 = abs(lambda_1[i, j].imag) >= params['tolerance_conjugation']
            if is_complex_lambda1:
                if (lambda_1[i, j].imag > 0 and conjugation_first_imag_negative) or \
                   (lambda_1[i, j].imag < 0 and not conjugation_first_imag_negative): 
                    lambda_1[i, j] = lambda_1[i, j].conjugate()
                # now it is ensured that lambda_1 has correct sign in imag. part and we can set lambda_2
                lambda_2[i, j] = lambda_1[i, j].conjugate()
            
            else: # lambda_1 is real. therefore we must ensure lambda_2 is real as well
    
                m_inds_wo1 = [m for m in range(N_eigvals) if m!=m_dominant_1]
                            
                found_m_dominant_2 = False
                while not found_m_dominant_2:
                    m_dominant_2 = m_inds_wo1[np.argmin(np.abs(lambda_all[m_inds_wo1, i, j].real))]
                    lambda_2_candidate = lambda_all[m_dominant_2, i, j]
                    # ensure that lambda_2 is real, too
                    if abs(lambda_2_candidate.imag) <= params['tolerance_conjugation']:
                        found_m_dominant_2 = True
                    else:
                        m_inds_wo1.remove(m_dominant_2)
                if found_m_dominant_2:
                    lambda_2[i, j] = lambda_2_candidate
                else:
                    print('ERROR: there is no second dominant real eigenvalue for mu={}, sigma={}'.format(mu[i], sigma[i]))
                    exit()

    # now we have extracted the first two dominant eigenvalues with 
    # the property that for complex conjugate pairs (regular modes) 
    # they are indeed complex conjugates and lambda_1 has positive imag. part
    quantities_dict['lambda_1'] = lambda_1
    quantities_dict['lambda_2'] = lambda_2
    
    if save_spec:
        specsolv.save_quantities(folder+'/'+filename, quantities_dict)     
        print('saving spectrum after postprocessing done.')

else:
    specsolv.load_quantities(folder+'/'+filename, quantities_dict, 
                             quantities=['lambda_1', 'lambda_2'], load_params=False)      


# II. QUANTITIES CORRESPONDING TO SPECTRUM ABOVE -- COMPUTATION/LOADING

# after having computed or loaded the first two dominant eigenvalues 
# lambda_1 and lambda_2 we use them to compute the (nonstationary) 
# quantities (f_1, f_2, c_mu_1, c_mu_2, c_sigma_1, c_sigma_2 
# [and psi_r_1, psi_r_2 that are not used in the current models]
# and furthermore we compute the stationary quantities, too: r_inf, V_mean_inf 
# and derivatives of tha latter two w.r.t. mu and sigma

# QUANTITY LOADING
quant_loaded = False
if load_quant:
                        
    quant_names = [ 'lambda_1', 'lambda_2', 
                    'r_inf', 'dr_inf_dmu', 'dr_inf_dsigma',
                    'V_mean_inf', 'dV_mean_inf_dmu', 'dV_mean_inf_dsigma',
                    'f_1', 'f_2', 'psi_r_1', 'psi_r_2',
                    'c_mu_1', 'c_mu_2', 'c_sigma_1', 'c_sigma_2',
                    'mu', 'sigma']

    specsolv.load_quantities(folder+'/'+filename, quantities_dict, 
                             quantities=quant_names, load_params=load_params)
    quant_loaded = True

quant_computed = False
if compute_quant and not quant_loaded:  
    
    assert 'lambda_1' and 'lambda_2' in quantities_dict # we need to find lambda_1 and lambda_2 before this
    
    # do the actual quantity computation of the mu sigma rectangle via the following method call
    specsolv.compute_quantities_rect(quantities_dict, 
                            quant_names=['r_inf', 'dr_inf_dmu', 'dr_inf_dsigma',
                                         'V_mean_inf', 'dV_mean_inf_dmu', 'dV_mean_inf_dsigma',
                                         'f_1', 'f_2', 'psi_r_1', 'psi_r_2',
                                         'c_mu_1', 'c_mu_2', 'c_sigma_1', 'c_sigma_2'
                                        ], N_procs=N_procs)

    # SAVING
    # spectrum: saving into hdf5 file: mu, sigma, lambda_all and params
    # quantities: saving into hdf5 file: lambda_1, lambda_2, r_inf, V_mean_inf 
    # and those coefficients required for the spectral_mattia_diff model
    if save_quant:

        specsolv.save_quantities(folder+'/'+filename, quantities_dict)
    
        print('saving quantities after computing done.')

if postprocess_quant:
    
    # remove artefacts due to proximity to double eigenvalues at the transition from real to complex
    # by taking the value of the nearest neighbor for those mu, sigma values
    
    quantities_postprocess(quantities_dict, 
                           quant_names=['lambda_1', 'lambda_2',
                                        'f_1', 'f_2', 
                                        'psi_r_1', 'psi_r_2',
                                        'c_mu_1', 'c_mu_2', 
                                        'c_sigma_1', 'c_sigma_2'], 
                            minsigma_interp=0.5, maxsigma_interp=5., maxmu_interp=0.52, 
                            tolerance_conjugation=params['tolerance_conjugation'])
    
    if save_quant:
        specsolv.save_quantities(folder+'/'+filename, quantities_dict)
        print('saving quantities after postprocessing done.')

# the following code serves only the purpose to be able plotting curves similar to those 
# of Fig 7A (left attached plot) of the paper  demonstrating the initialization of the algorithm
# the flux at the lower bound is obtained to be able to plot the _real_ initialization of the 
# spectral solver at the smallest mu
if obtain_fluxlb:
    
    lambda_real_grid = np.linspace(-0.6, 1e-5, 500) # note this is only used when not loading
    N_eigs_min = 8 # in that interval above: sigma=1.5 => 10+1 eigvals, sigma=3.5 => 8+1 eigvals
    
    fluxlb_quants = {}    
    
    specsolv.load_quantities(folder+'/'+filename, fluxlb_quants, 
                             quantities=
                                 ['lambda_real_grid'] 
                                 +['lambda_real_found_sigma{:.1f}'.format(sig) for sig 
                                     in [sigma_smaller, sigma_larger]]
                                 +['qlb_real_sigma{:.1f}'.format(sig) for sig 
                                     in [sigma_smaller, sigma_larger]], load_params=False)
                                 
    if 'lambda_real_grid' in fluxlb_quants:
        print('loading of the (]aw) real lambda/qlb data was sucessful')
    else:
        print('loading of the (]aw) real lambda/qlb data was NOT sucessful => computing!')                        
        
        fluxlb_quants['lambda_real_grid'] = lambda_real_grid
        
        specsolv.params['mu'] = mu_min  
        for sig in [sigma_smaller, sigma_larger]:
            specsolv.params['sigma'] = sig
            lambda_real_found, qs_real, qlb_real = specsolv.real_eigenvalues(lambda_real_grid, min_ev=N_eigs_min)    
            fluxlb_quants['qlb_real_sigma{:.1f}'.format(sig)] = qlb_real
            fluxlb_quants['lambda_real_found_sigma{:.1f}'.format(sig)] = lambda_real_found
            
        quantities_dict.update(fluxlb_quants) # merge into other quantities as saving overwrites the whole file
        specsolv.save_quantities(folder+'/'+filename, quantities_dict)
        
        print('saving obtained fluxes at lower bound after computing them')   
            

# PLOTTING

if no_sigma_quantities_plot == 'all':
    no_sigma_quantities_plot = sigma.shape[0]

# choose which sigma values are plotted for quantities & eigenvalues visualiation
sigmas_plot = np.linspace(sigma[0], sigma[-1], no_sigma_quantities_plot) #sigma #np.linspace(0.5, 2.5, 11) #sigma #np.arange(0.5, 5.0001, 0.1) #[2.0, 4.0] # argmin choice
inds_sigma_plot = [np.argmin(np.abs(sigma-sig)) for sig in sigmas_plot]


# FULL SPECTRUM PLOTTING   
if plot_full_spectrum_sigma:
    sigmas_plot_raw = [1.5, 3.5] 
    inds_sigma_plot_raw = [np.argmin(np.abs(sigma-sig)) for sig in sigmas_plot_raw]    
    plot_raw_spectrum_sigma(lambda_all, mu, sigma, inds_sigma_plot_raw)
    
if plot_full_spectrum_eigvals:
    plot_raw_spectrum_eigvals(lambda_all, mu, sigma)

quantities_validation = {}
if plot_validation:
    
    file_josef_ref = 'precalc05_int.mat' # Tref=1.4    
    file_josef_noref = 'quantities_cascade.h5' # Tref=0
    if abs(params['t_ref']-1.5) < 1e-10:
        mat_josef = scipy.io.loadmat(file_josef_ref)['presimdata'][0,0]
        
        quantities_validation = {'mu': mat_josef['Irange'].flatten()/20., # mu = I/C
                                 'sigma': mat_josef['sigmarange'].flatten(),
                                 'r_inf': mat_josef['r_raw'],
                                 'dr_inf_dmu': mat_josef['drdmu_raw'],
                                 'dr_inf_dsigma': mat_josef['drdsigma_raw'],
                                 'V_mean_inf': mat_josef['Vmean_raw'],
                                }
    elif params['t_ref'] == 0.:
        quant_names = [ 'r_ss', 'V_mean_ss', 'mu_vals', 'sigma_vals']
        specsolv.load_quantities(folder+'/'+file_josef_noref, quantities_validation, 
                                 quantities=quant_names, load_params=False)
        # use own names 
        quantities_validation['mu'] = quantities_validation['mu_vals']
        quantities_validation['sigma'] = quantities_validation['sigma_vals']
        quantities_validation['r_inf'] = quantities_validation['r_ss']
        quantities_validation['V_mean_inf'] = quantities_validation['V_mean_ss']

if plot_quantities and 'eigvals' in plot_quantities:
    
    plot_quantities_eigvals(quantities_dict, inds_sigma_plot, colormap_sigma=colormap_sigma,
                            plot_validation=plot_validation, quantities_validation=quantities_validation)

    
if plot_quantities and 'real' in plot_quantities:
    
    plot_quantities_real(quantities_dict, inds_sigma_plot, colormap_sigma=colormap_sigma,
                         plot_validation=plot_validation, quantities_validation=quantities_validation)
    
    

if plot_quantities and 'complex' in plot_quantities:
    
    # the following lines contain some examples of quantities
    # which could be plotted by the plot_quantities entry 'complex'
    complex_quants_plot = ['f_1', 'f_1*c_mu_1', 'c_mu_1']
    #complex_quants_plot = ['f_1', 'f_1*psi_r_1', 'psi_r_1']
    #complex_quants_plot = ['f_2', 'f_2*psi_r_2', 'psi_r_2']
    #complex_quants_plot = ['f_1', 'f_1*c_mu_1', 'c_mu_1']
    #complex_quants_plot = ['f_2', 'f_2*c_mu_2', 'c_mu_2']
    #complex_quants_plot = ['f_1', 'f_1*c_sigma_1', 'c_sigma_1']
    #complex_quants_plot = ['f_2', 'f_2*c_sigma_2', 'c_sigma_2']
    #complex_quants_plot = ['f_1', 'f_2', 'psi_r_1', 'psi_r_2', 'c_mu_1', 'c_mu_2', 'c_sigma_1', 'c_sigma_2']
    #complex_quants_plot = ['f_1*psi_r_1', 'f_2*psi_r_2', 'f_1*c_mu_1', 'f_2*c_mu_2', 'f_1*c_sigma_1', 'f_2*c_sigma_2']
    plot_quantities_complex(complex_quants_plot, quantities_dict, inds_sigma_plot, colormap_sigma=colormap_sigma,
                            plot_validation=plot_validation, quantities_validation=quantities_validation)


if plot_quantities and 'composed' in plot_quantities:
    
    composed_quantities = {'f.cmu' : quantities_dict['f_1']*quantities_dict['c_mu_1'] + quantities_dict['f_2']*quantities_dict['c_mu_2'],
                           'f.csigma': quantities_dict['f_1']*quantities_dict['c_sigma_1'] + quantities_dict['f_2']*quantities_dict['c_sigma_2'],
                           'f.(Lambda*cmu)' : quantities_dict['f_1']*quantities_dict['lambda_1']*quantities_dict['c_mu_1'] + quantities_dict['f_2']*quantities_dict['lambda_2']*quantities_dict['c_mu_2'],
                           'f.(Lambda*csigma)' : quantities_dict['f_1']*quantities_dict['lambda_1']*quantities_dict['c_sigma_1'] + quantities_dict['f_2']*quantities_dict['lambda_2']*quantities_dict['c_sigma_2'],
                          }
    comp_quants_validat = {}

    plot_quantities_composed(composed_quantities, quantities_dict, inds_sigma_plot, colormap_sigma=colormap_sigma,
                            plot_validation=plot_validation, quantities_validation=quantities_validation, 
                            comp_quants_validat=comp_quants_validat)  



if plot_real_inits:
           
    
    
    plt.figure()
    sid = 1
    for sig in [sigma_smaller, sigma_larger]:
        
        lambda_real_grid = fluxlb_quants['lambda_real_grid']
        lambda_real_found = fluxlb_quants['lambda_real_found_sigma{:.1f}'.format(sig)]
        
        plt.subplot(1, 2, sid)
        plt.title('$\mu={}, \sigma={}$'.format(mu_min, sig))
        plt.plot(fluxlb_quants['qlb_real_sigma{:.1f}'.format(sig)], lambda_real_grid)
        plt.plot(np.zeros_like(lambda_real_found), lambda_real_found, 'rx', markersize=5)
        plt.plot(np.zeros_like(lambda_real_grid), lambda_real_grid, '--', color='gray')
        plt.xscale('symlog', linthreshx=1e-10)
        plt.ylim(lambda_real_grid[0], 0)
        plt.xlim(-10**-2, 10**-2)
        plt.xlabel('$q(V_\mathrm{lb})$ [kHz]')
        plt.ylabel('$\lambda$ [kHz]')
            
        
        sid += 1

if plot_eigenfunctions:
    
    # corresponding params
    mu_smaller_eigfun = 0.25
    mu_larger_eigfun = 1.5
    xlim = [-100, -40]
#    xlim = [-200, -40]
    
    plt.figure()

    plt.suptitle('eigenfunctions in a.u. truncated at V={}'.format(xlim[0]))
    
    plt.subplot(2, 2, 1)
    
    # REGULAR MODE -- REAL
    # small mu, small sigma
    mu_i = mu_smaller_eigfun
    sigma_j = sigma_smaller
    plt.title('$\mu={}, \sigma={}$'.format(mu_i, sigma_j))
    
    specsolv.params['mu'] = mu_i
    specsolv.params['sigma'] = sigma_j
    i = np.argmin(np.abs(quantities_dict['mu'] - mu_i))
    j = np.argmin(np.abs(quantities_dict['sigma'] - sigma_j))
    lambda1 = quantities_dict['lambda_1'][i,j]
    Vgrid, phi1, q = specsolv.eigenfunction(lambda1, mu_i, sigma_j)
    Vgrid, psi1, dpsi = specsolv.eigenfunction(lambda1, mu_i, sigma_j, adjoint=True)
    Vgrid, phi0, q = specsolv.eigenfunction(0., mu_i, sigma_j)
    
    # binormalize
    phi1 /= inner_prod(psi1, phi1, Vgrid)
    
    plotrange = Vgrid>=xlim[0]
    Vgrid = Vgrid[plotrange]
    phi1 = phi1[plotrange]
    psi1 = psi1[plotrange]
    phi0 = phi0[plotrange]
    phi1 /= np.abs(phi1).max()
    psi1 /= np.abs(psi1).max()
    phi0 /= np.abs(phi0).max()
    plt.plot(Vgrid, phi0.real, label='$\phi_0$')
    plt.plot(Vgrid, psi1.real, label='$\psi_1$')
    plt.plot(Vgrid, phi1.real, label='$\phi_1$')
    plt.legend(loc='best')
    plt.xlabel('V [mV]') 
    plt.ylabel('real eigfunc. [a.u.]')
    
    
    plt.subplot(2, 2, 3)
    
    # DIFFUSIVE MODE
    # larger mu, larger sigma
    mu_i = mu_larger_eigfun
    sigma_j = sigma_larger
    plt.title('$\mu={}, \sigma={}$'.format(mu_i, sigma_j))
    
    specsolv.params['mu'] = mu_i
    specsolv.params['sigma'] = sigma_j
    i = np.argmin(np.abs(quantities_dict['mu'] - mu_i))
    j = np.argmin(np.abs(quantities_dict['sigma'] - sigma_j))
    lambda1 = quantities_dict['lambda_1'][i,j]
    Vgrid, phi1, q = specsolv.eigenfunction(lambda1, mu_i, sigma_j)
    Vgrid, psi1, dpsi = specsolv.eigenfunction(lambda1, mu_i, sigma_j, adjoint=True)
    Vgrid, phi0, q = specsolv.eigenfunction(0., mu_i, sigma_j)
    
    # binormalize
    phi1 /= inner_prod(psi1, phi1, Vgrid)
    
    plotrange = Vgrid>=xlim[0]
    Vgrid = Vgrid[plotrange]
    phi1 = phi1[plotrange]
    psi1 = psi1[plotrange]
    phi0 = phi0[plotrange]
    phi1 /= np.abs(phi1).max()
    psi1 /= np.abs(psi1).max()
    phi0 /= np.abs(phi0).max()
    plt.plot(Vgrid, phi0.real, label='$\phi_0$')
    plt.plot(Vgrid, psi1.real, label='$\psi_1$')
    plt.plot(Vgrid, phi1.real, label='$\phi_1$')
#    plt.legend()
    plt.xlabel('V [mV]') 
    plt.ylabel('real eigfunc. [a.u.] -- diffusive')

    plt.subplot(2, 2, 2)
    
    # REGULAR MODE -- COMPLEX
    # larger mu, smaller sigma
    mu_i = mu_larger_eigfun
    sigma_j = sigma_smaller
    plt.title('$\mu={}, \sigma={}$'.format(mu_i, sigma_j))
    
    specsolv.params['mu'] = mu_i
    specsolv.params['sigma'] = sigma_j
    i = np.argmin(np.abs(quantities_dict['mu'] - mu_i))
    j = np.argmin(np.abs(quantities_dict['sigma'] - sigma_j))
    lambda1 = quantities_dict['lambda_1'][i,j]
    Vgrid, phi1, q = specsolv.eigenfunction(lambda1, mu_i, sigma_j)
    Vgrid, psi1, dpsi = specsolv.eigenfunction(lambda1, mu_i, sigma_j, adjoint=True)
    Vgrid, phi0, q = specsolv.eigenfunction(0., mu_i, sigma_j)

    # binormalize
    phi1 /= inner_prod(psi1, phi1, Vgrid)
    
    plotrange = Vgrid>=xlim[0]
    Vgrid = Vgrid[plotrange]
    phi1 = phi1[plotrange]
    psi1 = psi1[plotrange]
    phi0 = phi0[plotrange]
    
    
    phi1 /= np.abs(phi1.real).max()
    psi1 /= np.abs(psi1.real).max()
    phi0 /= np.abs(phi0.real).max()
    plt.plot(Vgrid, phi0.real, label='$\phi_0$')
    plt.plot(Vgrid, psi1.real, label='$\psi_1$')
    plt.plot(Vgrid, phi1.real, label='$\phi_1$')
#    plt.legend()
    plt.xlabel('V [mV]') 
    plt.ylabel('real part [a.u.]')
    
    plt.subplot(2, 2, 4)
    
    phi1 /= np.abs(phi1.imag).max()
    psi1 /= np.abs(psi1.imag).max()
    plt.plot(Vgrid, phi0.imag, label='$\phi_0$')
    plt.plot(Vgrid, psi1.imag, label='$\psi_1$')
    plt.plot(Vgrid, phi1.imag, label='$\phi_1$')
#    plt.legend()
    plt.xlabel('V [mV]') 
    plt.ylabel('imag. part [a.u.]')
    
    


if plot_paper_quantities:        
    
    sigmas_quant_plot = np.arange(0.5, 4.501, 0.2)
    sigmas_plot = [sigma_smaller, sigma_larger] # used by plots shown for two sigma values
    mus_plot = [1.5] # 0.25
    mu_smaller_eigfun = 0.25
    mu_larger_eigfun = 1.5
    mu_vals = quantities_dict['mu']
    sigma_vals = quantities_dict['sigma']   
    
    plt.figure()
    plt.suptitle('spectral quantities')
    
    mu_lim = [-1, 4]
    inds_mu_plot = [i for i in range(len(mu_vals)) if mu_lim[0] <= mu_vals[i] <= mu_lim[1]]
    inds_sigma_plot = [np.argmin(np.abs(sigma_vals-sig)) for sig in sigmas_quant_plot]
    mu_plot_ind = np.argmin(np.abs(mu_vals-mus_plot[0]))
    N_sigma = len(inds_sigma_plot)    
    
    for k_j, j in enumerate(inds_sigma_plot):
        # color black-green    
        rgb = [0, float(k_j)/(N_sigma-1), 0]
        linecolor = rgb
        # color red
        linecolor2 =  [float(k_j)/(N_sigma-1), 0,  0.] #0.2 + float(k_j)/(N_sigma-1)*0.8]
    
        ax1 = plt.subplot(4, 4, 1)
        # labels
        if k_j in [0, N_sigma//2, N_sigma-1]:
            siglabel = '$\sigma={0:.3}$ [mV/$\sqrt{{\mathrm{{ms}}}}$]'.format(sigma_vals[j])
                       
        else:
            siglabel = None

        ylim = [0,50]
        plotinds_lim = -1/quantities_dict['lambda_1'].real[inds_mu_plot,j] <= ylim[1]
        plt.plot(mu_vals[inds_mu_plot][plotinds_lim], -1/quantities_dict['lambda_1'].real[inds_mu_plot,j][plotinds_lim], 
                 label=siglabel, color=linecolor)
        for l in range(len(sigmas_plot)): 
            if np.round(sigmas_plot[l],2)==np.round(sigma_vals[j],2):
                plt.plot(mus_plot[0], -1/quantities_dict['lambda_1'].real[mu_plot_ind,j], 
                'o', color=linecolor, markersize=5)
        if k_j==0:
            plt.title(r'$-1/\mathrm{Re}\lambda_1$', fontsize=14)
            plt.ylabel('[ms]', fontsize=12)
            plt.ylim(ylim)
            plt.yticks(ylim)


        if k_j==len(inds_sigma_plot)-1:
            plt.legend()
            
            
        siglabel = None 
        
        
        plt.subplot(4, 4, 2)

        plt.plot(mu_vals[inds_mu_plot], quantities_dict['lambda_1'].imag[inds_mu_plot,j]/(2*np.pi), 
                 label=siglabel, color=linecolor)
        for l in range(len(sigmas_plot)): 
            if np.round(sigmas_plot[l],2)==np.round(sigma_vals[j],2):
                plt.plot(mus_plot[0], quantities_dict['lambda_1'].imag[mu_plot_ind,j]/(2*np.pi), 
                'o', color=linecolor, markersize=5)
        if k_j==0:
            plt.title(r'$\mathrm{Im}\lambda_1 / (2\pi)$', fontsize=14)
            plt.ylabel('[kHz]', fontsize=12)
            ylim = [-.005,0.14]
            plt.ylim(ylim)
            plt.yticks([0,.14])
            
            
        plt.subplot(4, 4, 5)
                
        if k_j>0:
            plotinds_lim = -1/quantities_dict['lambda_2'].real[inds_mu_plot,j] <= ylim[1]
        else: 
            plotinds_lim = -1/quantities_dict['lambda_2'].real[inds_mu_plot,j] <= 1000.
        plt.plot(mu_vals[inds_mu_plot][plotinds_lim], -1/quantities_dict['lambda_2'].real[inds_mu_plot,j][plotinds_lim], 
                 label=siglabel, color=linecolor)
        for l in range(len(sigmas_plot)): 
            if np.round(sigmas_plot[l],2)==np.round(sigma_vals[j],2):
                plt.plot(mus_plot[0], -1/quantities_dict['lambda_2'].real[mu_plot_ind,j], 
                'o', color=linecolor, markersize=5)
        if k_j==0:
            plt.title(r'$-1/\mathrm{Re}\lambda_2$', fontsize=14)
            plt.ylabel('[ms]', fontsize=12)
            plt.xlabel('$\mu$ [mV/ms]', fontsize=12)
            ylim = [0,50]
            plt.ylim(ylim)
            plt.yticks(ylim)
            
        
        plt.subplot(4, 4, 6)

        plt.plot(mu_vals[inds_mu_plot], quantities_dict['lambda_2'].imag[inds_mu_plot,j]/(2*np.pi), 
                 label=siglabel, color=linecolor)
        for l in range(len(sigmas_plot)): 
            if np.round(sigmas_plot[l],2)==np.round(sigma_vals[j],2):
                plt.plot(mus_plot[0], quantities_dict['lambda_2'].imag[mu_plot_ind,j]/(2*np.pi), 
                'o', color=linecolor, markersize=5)
        if k_j==0:
            plt.title(r'$\mathrm{Im}\lambda_2 / (2\pi)$', fontsize=14)
            plt.ylabel('[kHz]', fontsize=12)
            plt.xlabel('$\mu$ [mV/ms]', fontsize=12)
            ylim = [-0.14,.005]
            plt.ylim(ylim)
            plt.yticks([-.14, 0])
            
        
        fcmu = (quantities_dict['f_1']*quantities_dict['c_mu_1'] + 
                quantities_dict['f_2']*quantities_dict['c_mu_2']).real
                
        plt.subplot(4, 4, 3)

        plt.plot(mu_vals[inds_mu_plot], (fcmu + quantities_dict['dr_inf_dmu'])[inds_mu_plot,j], 
                 label=siglabel, color=linecolor)
        for l in range(len(sigmas_plot)): 
            if np.round(sigmas_plot[l],2)==np.round(sigma_vals[j],2):
                plt.plot(mus_plot[0], (fcmu + quantities_dict['dr_inf_dmu'])[mu_plot_ind,j], 
                'o', color=linecolor, markersize=5)
        if k_j==0:
            plt.title(r'$M$', fontsize=14)
            plt.ylabel('[1/mV]', fontsize=12)
            ylim = [0.,0.035]
            plt.ylim(ylim)
            plt.yticks(ylim)
        
                
        fcsigma2 = (quantities_dict['f_1']*quantities_dict['c_sigma_1'] + 
                quantities_dict['f_2']*quantities_dict['c_sigma_2']).real / (2*sigma_vals[j]) # scale for sigma->sigma^2
        dr_inf_dsigma2 = quantities_dict['dr_inf_dsigma'] / (2*sigma_vals[j])
        plt.subplot(4, 4, 7)

        plt.plot(mu_vals[inds_mu_plot], (fcsigma2+dr_inf_dsigma2)[inds_mu_plot,j], 
                 label=siglabel, color=linecolor)
        for l in range(len(sigmas_plot)): 
            if np.round(sigmas_plot[l],2)==np.round(sigma_vals[j],2):
                plt.plot(mus_plot[0], (fcsigma2+dr_inf_dsigma2)[mu_plot_ind,j], 
                'o', color=linecolor, markersize=5)
        if k_j==0:
            plt.title(r'$S$', fontsize=14)
            plt.ylabel('[1/mV$^2$]', fontsize=12)
            plt.xlabel('$\mu$ [mV/ms]', fontsize=12)
            ylim = [-0.0005,0.009]
            plt.ylim(ylim)
            plt.yticks([0,0.009])
            
            
        
        F_mu = (quantities_dict['f_1']*quantities_dict['c_mu_1']*quantities_dict['lambda_1'] + 
                quantities_dict['f_2']*quantities_dict['c_mu_2']*quantities_dict['lambda_2']).real
        plt.subplot(4, 4, 4)

        plt.plot(mu_vals[inds_mu_plot], F_mu[inds_mu_plot,j], 
                 label=siglabel, color=linecolor)
        for l in range(len(sigmas_plot)): 
            if np.round(sigmas_plot[l],2)==np.round(sigma_vals[j],2):
                plt.plot(mus_plot[0], F_mu[mu_plot_ind,j], 
                'o', color=linecolor, markersize=5)
        if k_j==0:
            plt.title(r'$F_\mu$', fontsize=14)
            plt.ylabel('[kHz/mV]', fontsize=12)
            ylim = [-.00001,0.005]
            plt.ylim(ylim)
            plt.yticks([.0,0.005])
            
            
        
        F_sigma2 = (quantities_dict['f_1']*quantities_dict['c_sigma_1']*quantities_dict['lambda_1'] + 
                    quantities_dict['f_2']*quantities_dict['c_sigma_2']*quantities_dict['lambda_2']).real / (2*sigma_vals[j]) # scale for dsigma->dsigma^2
        plt.subplot(4, 4, 8)

        plt.plot(mu_vals[inds_mu_plot], F_sigma2[inds_mu_plot,j], 
                 label=siglabel, color=linecolor)
        for l in range(len(sigmas_plot)): 
            if np.round(sigmas_plot[l],2)==np.round(sigma_vals[j],2):
                plt.plot(mus_plot[0], F_sigma2[mu_plot_ind,j], 
                'o', color=linecolor, markersize=5)
        if k_j==0:
            plt.title(r'$F_{\sigma^2}$', fontsize=14)
            plt.ylabel('[kHz/mV$^2$]', fontsize=12)
            plt.xlabel('$\mu$ [mV/ms]', fontsize=12)
            ylim = [-.001,0.001]
            plt.ylim(ylim)
            plt.yticks(ylim)
    
    
    N_eigvals = 10    
    sigma_fig = [1.5, 3.5]
    xlim = [-1,4]
    plotinds = (mu >= xlim[0]) & (mu <= xlim[1])
    sigma_inds_fig = [np.argmin(np.abs(sigma-sig)) for sig in sigma_fig]    
    plt.figure()
    plt.suptitle('raw spectrum')
    subplotid = 1 # left/right real/imag, rows: eigvals
    N_plotcols = len(sigma_inds_fig) # N_sigma//sigma_skip_inds+(1 if N_sigma % sigma_skip_inds > 0 else 0)
    # axis sharing
    ax_real = plt.subplot(2, N_plotcols, subplotid, sharex=None, sharey=None)
    ax_imag = plt.subplot(2, N_plotcols, subplotid+N_plotcols, sharex=ax_real, sharey=None)
    for l_j, j in enumerate(sigma_inds_fig):
        
        
        for k in range(N_eigvals):
            
            color_regular = [0, 0, 1 - float(k)/(N_eigvals-1)*0.66]
            color_diffusive = [1-float(k)/(N_eigvals-1)*0.33, 0, 0]
            if np.sum(np.abs(lambda_all[k, :, j].imag)) > 1e-5:
                linecolor = color_regular
            else:
                linecolor = color_diffusive
            
            ylim_real = [-0.6, 0.01]
            # remove pts outside of limit
            if l_j==1 or k < 8:
                plotinds_real = (lambda_all[k, plotinds, j].real >= ylim_real[0]) & (lambda_all[k, plotinds, j].real <= ylim_real[1]) 
            else:
                plotinds_real = lambda_all[k, plotinds, j].real >= -10000.
            
            # eigenval: real part
            plt.subplot(2, N_plotcols, subplotid, 
                        sharex=ax_real if subplotid > 1 else None, 
                        sharey=ax_real if subplotid > 1 else None)
            plt.plot(mu[plotinds][plotinds_real], lambda_all[k, plotinds, j].real[plotinds_real], color=linecolor)
            if l_j==0:
                if k==0:
                    plt.ylabel('$\mathrm{Re}\lambda_n$ [kHz]')
            if k==0:
                plt.title('$\sigma={0:.3}$ [mV/$\sqrt{{\mathrm{{ms}}}}$]'.format(sigma[j]))
                plt.ylim(ylim_real)
                plt.yticks([-0.6,0])
                plt.plot(mu[plotinds][plotinds_real], np.zeros_like(mu[plotinds])[plotinds_real], color='gray', lw=2)
                plt.xlim(xlim)
            
            # dots for relation with other fig                
            if k==0 and l_j==0:
                i = np.argmin(np.abs(mu-0.25))
                plt.plot(mu[i], lambda_all[k, i, j].real, '^b', ms=5)
            if k==0 and l_j==0:
                i = np.argmin(np.abs(mu-1.5))
                plt.plot(mu[i], lambda_all[k, i, j].real, 'ob', ms=5)
            if k==0 and l_j==1:
                i = np.argmin(np.abs(mu-1.5))
                plt.plot(mu[i], lambda_all[k, i, j].real, 'or', ms=5)
            
            # eigenval: imag part
            plt.subplot(2, N_plotcols, subplotid+N_plotcols, 
                        sharex=ax_real, 
                        sharey=ax_imag if subplotid > 1 else None)
            plt.plot(mu[plotinds], lambda_all[k, plotinds, j].imag, color=linecolor)
            if l_j==0:
                if k==0:
                    plt.ylabel('$\mathrm{Im}\lambda_n$ [kHz]')
            if k==0: 
                ylim_i = [-2.7, 2.7]
                plt.ylim(ylim_i)
                plt.yticks(ylim_i)
                plt.xlabel('$\mu$ [mV/ms]')
                plt.xlim(xlim)
                
            if k==0 and l_j==0:
                i = np.argmin(np.abs(mu-1.5))
                plt.plot(mu[i], lambda_all[k, i, j].imag, 'ob', ms=5)
        
        if l_j==0:
            plt.subplot(2, N_plotcols, subplotid, 
                        sharex=ax_real if subplotid > 1 else None, 
                        sharey=ax_real if subplotid > 1 else None)
            plt.legend(loc='best')
        
        subplotid += 1

    mu_min = -1.0 # to be compatible with cascade models
    lambda_real_grid = np.linspace(-0.54, 1e-5, 500)
    N_eigs_min = 8 # in that interval above: sigma=1.5 => 10+1 eigvals, sigma=3.5 => 8+1 eigvals
    
    
    plt.figure()
    sid = 1
    for sig in [sigma_smaller, sigma_larger]:
        
        lambda_real_grid = fluxlb_quants['lambda_real_grid']
        lambda_real_found = fluxlb_quants['lambda_real_found_sigma{:.1f}'.format(sig)]
        
        plt.subplot(1, 2, sid)
        plt.title('$\mu={}, \sigma={}$'.format(mu_min, sig))
        plt.plot(np.zeros_like(lambda_real_grid), lambda_real_grid, '--', color='gray')
        plt.plot(fluxlb_quants['qlb_real_sigma{:.1f}'.format(sig)], lambda_real_grid, color='black')
        plt.plot(np.zeros_like(lambda_real_found), lambda_real_found, 'o', markersize=5, color='gray')
        plt.xscale('symlog', linthreshx=1e-10)
        plt.ylim(lambda_real_grid[0], 0)
        plt.xlabel('$q(V_\mathrm{lb})$ [kHz]')
        plt.ylabel('$\lambda$ [kHz]')
        plt.xticks([-10**-2, -10**-10, 0, 10**-10, 10**-2])
        plt.xlim(-10**-2, 10**-2)
        plt.ylim(-0.6, 0)
        plt.yticks([-0.6,0])
            
        
        sid += 1
    
    xlim = [-100, -40]
    skippts = 50
    xticks = [-100, -70, -40]
    ylim = [-1,1]
    col_phi0 = 'gray'
    col_regular = 'blue'
    col_diffusive = 'red'
    
    
    plt.figure()

    plt.suptitle('eigenfunctions in a.u. truncated at V={}'.format(xlim[0]))
    
    plt.subplot(2, 2, 1)
    
    # REGULAR MODE -- REAL
    # small mu, small sigma
    mu_i = mu_smaller_eigfun
    sigma_j = sigma_smaller
    plt.title('$\mu={}, \sigma={}$'.format(mu_i, sigma_j))
    
    specsolv.params['mu'] = mu_i
    specsolv.params['sigma'] = sigma_j
    i = np.argmin(np.abs(quantities_dict['mu'] - mu_i))
    j = np.argmin(np.abs(quantities_dict['sigma'] - sigma_j))
    lambda1 = quantities_dict['lambda_1'][i,j]
    Vgrid, phi1, q = specsolv.eigenfunction(lambda1, mu_i, sigma_j)
    Vgrid, psi1, dpsi = specsolv.eigenfunction(lambda1, mu_i, sigma_j, adjoint=True)
    Vgrid, phi0, q = specsolv.eigenfunction(0., mu_i, sigma_j)
    
    # binormalize
    phi1 /= inner_prod(psi1, phi1, Vgrid)
    
    plotrange = Vgrid>=xlim[0]    
    
    Vgrid = Vgrid[plotrange]
    phi1 = phi1[plotrange]
    psi1 = psi1[plotrange]
    phi0 = phi0[plotrange]
    plotinds = np.concatenate([ np.arange(0, len(Vgrid)-1, skippts), np.array([len(Vgrid)-1]) ])
    phi1 /= np.abs(phi1).max()
    psi1 /= np.abs(psi1).max()
    phi0 /= np.abs(phi0).max()
    plt.plot(Vgrid[plotinds], phi0.real[plotinds], label='$\phi_0$', color=col_phi0)
    plt.plot(Vgrid[plotinds], psi1.real[plotinds], '--', label='$\psi_1$', color=col_regular)
    plt.plot(Vgrid[plotinds], phi1.real[plotinds], label='$\phi_1$', color=col_regular)
    plt.legend(loc='best')
#    plt.xlabel('V [mV]') 
    plt.ylabel('real function [a.u.]')
    plt.xticks(xticks)
    plt.ylim(ylim)
    plt.yticks(ylim)
    
    
    plt.subplot(2, 2, 3)
    
    # DIFFUSIVE MODE
    # larger mu, larger sigma
    mu_i = mu_larger_eigfun
    sigma_j = sigma_larger
    plt.title('$\mu={}, \sigma={}$'.format(mu_i, sigma_j))
    
    specsolv.params['mu'] = mu_i
    specsolv.params['sigma'] = sigma_j
    i = np.argmin(np.abs(quantities_dict['mu'] - mu_i))
    j = np.argmin(np.abs(quantities_dict['sigma'] - sigma_j))
    lambda1 = quantities_dict['lambda_1'][i,j]
    Vgrid, phi1, q = specsolv.eigenfunction(lambda1, mu_i, sigma_j)
    Vgrid, psi1, dpsi = specsolv.eigenfunction(lambda1, mu_i, sigma_j, adjoint=True)
    Vgrid, phi0, q = specsolv.eigenfunction(0., mu_i, sigma_j)
    
    # binormalize
    phi1 /= inner_prod(psi1, phi1, Vgrid)
    
    plotrange = Vgrid>=xlim[0]
    Vgrid = Vgrid[plotrange]
    phi1 = phi1[plotrange]
    psi1 = psi1[plotrange]
    phi0 = phi0[plotrange]
    phi1 /= np.abs(phi1).max()
    psi1 /= np.abs(psi1).max()
    phi0 /= np.abs(phi0).max()
    plt.plot(Vgrid[plotinds], phi0.real[plotinds], label='$\phi_0$', color=col_phi0)
    plt.plot(Vgrid[plotinds], psi1.real[plotinds], '--', label='$\psi_1$', color=col_diffusive)
    plt.plot(Vgrid[plotinds], phi1.real[plotinds], label='$\phi_1$', color=col_diffusive)
    plt.xlabel('V [mV]') 
    plt.ylabel('real function [a.u.]')
    plt.xticks(xticks)
    plt.ylim(ylim)
    plt.yticks(ylim)    
    
    plt.subplot(2, 2, 2)
    
    # REGULAR MODE -- COMPLEX
    # larger mu, smaller sigma
    mu_i = mu_larger_eigfun
    sigma_j = sigma_smaller
    plt.title('$\mu={}, \sigma={}$'.format(mu_i, sigma_j))
    
    specsolv.params['mu'] = mu_i
    specsolv.params['sigma'] = sigma_j
    i = np.argmin(np.abs(quantities_dict['mu'] - mu_i))
    j = np.argmin(np.abs(quantities_dict['sigma'] - sigma_j))
    lambda1 = quantities_dict['lambda_1'][i,j]
    Vgrid, phi1, q = specsolv.eigenfunction(lambda1, mu_i, sigma_j)
    Vgrid, psi1, dpsi = specsolv.eigenfunction(lambda1, mu_i, sigma_j, adjoint=True)
    Vgrid, phi0, q = specsolv.eigenfunction(0., mu_i, sigma_j)

    # binormalize
    phi1 /= inner_prod(psi1, phi1, Vgrid)
    
    plotrange = Vgrid>=xlim[0]
    Vgrid = Vgrid[plotrange]
    phi1 = phi1[plotrange]
    psi1 = psi1[plotrange]
    phi0 = phi0[plotrange]
    
    
    phi1 /= np.abs(phi1.real).max()
    psi1 /= np.abs(psi1.real).max()
    phi0 /= np.abs(phi0.real).max()
    plt.plot(Vgrid[plotinds], phi0.real[plotinds], label='$\phi_0$', color=col_phi0)
    plt.plot(Vgrid[plotinds], psi1.real[plotinds], '--', label='$\psi_1$', color=col_regular)
    plt.plot(Vgrid[plotinds], phi1.real[plotinds], label='$\phi_1$', color=col_regular)
    plt.ylabel('real part [a.u.]')
    plt.xticks(xticks)
    plt.ylim(ylim)
    plt.yticks(ylim)
    
    
    plt.subplot(2, 2, 4)
    
    phi1 /= np.abs(phi1.imag).max()
    psi1 /= np.abs(psi1.imag).max()
    plt.plot(Vgrid[plotinds], phi0.imag[plotinds], label='$\phi_0$', color=col_phi0)
    plt.plot(Vgrid[plotinds], psi1.imag[plotinds], '--', label='$\psi_1$', color=col_regular)
    plt.plot(Vgrid[plotinds], phi1.imag[plotinds], label='$\phi_1$', color=col_regular)
    plt.xlabel('V [mV]') 
    plt.ylabel('imag. part [a.u.]')
    plt.xticks(xticks)
    plt.ylim(ylim)
    plt.yticks(ylim)
        
plt.show()
