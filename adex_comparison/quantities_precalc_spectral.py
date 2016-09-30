# -*- coding: utf-8 -*-

# script for computing the spectrum and associated quantities for an 
# exponential integrate-and-fire neuron with a lower bound different from the reset
# on a rectangle of input moments (mu, sigma)
# uses the spectralsolver package -- written by Moritz Augustin in 2016

import sys
sys.path.insert(1, '..')
from spectralsolver.spectralsolver import SpectralSolver, spectrum_enforce_complex_conjugation, \
                                          quantities_postprocess, inner_prod
from spectralsolver.plotting import plot_raw_spectrum_sigma, plot_raw_spectrum_eigvals, \
                                    plot_quantities_eigvals, plot_quantities_real, \
                                    plot_quantities_complex, plot_quantities_composed
from params import get_params
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from collections import OrderedDict 
import os
from multiprocessing import cpu_count
#import seaborn
import matplotlib
#matplotlib.rcParams['svg.fonttype'] = 'none'
matplotlib.rcParams['text.usetex'] = True



# PARAMETERS

# default params
params = get_params()

# input/output
filename = 'quantities_spectral_noref.h5'
#filename = 'quantities_spectral_noref_newflux.h5'

folder = os.path.dirname(os.path.realpath(__file__)) # store files in the same directory as the script itself


# eigenmodes and input grid
# note that the following parameters will be overriden if the computation is not performed due to loading from file
N_eigvals = 10 # 10 # no of eigenvalues (enumeration w.r.t. the smallest mu value)
               # for up to sigma=5mV/sqrt(ms) 9 eigvals are sufficient to extract first two 
               # dominant ones on the mu-sigma grid 
N_mu = 261 #461 # mu grid points -- s.t. -1.5 to 10 is spaced with distance 0.025 
N_sigma = 46 #sigma grid points  -- s.t. 0.5 to 5 is spaced with distance 0.1
N_procs = 12 #cpu_count() # no of parallel processes for spectrum computation

mu = np.linspace(-1.5, 5., N_mu)  # mu (mean input) grid. note that mu_min=-1 seems not to be small 
                                   # enough to have real only eigvals (for V_lb=-200)
sigma = np.linspace(0.5, 5., N_sigma) # sigma (noise intensity) grid. note that sigma_min=0.25 
                                      # is too sensitive (eigenvalue pairs' real parts are very close by)

# the spectralsolver uses mV and ms units both for (neuronal and input) parameters as well as for computed 
# quantities (e.g. eigenvalues, r_inf, other coefficients dep on eigenfunctions)

# the solver initialization uses the mu_min value (which is assumed to lead to purely real 
# eigenvalues) and (densely) evaluatess the eigenflux_lb for the following real grid 
# note: pay attention that this grid is not too coarse otherwise zeroes might be overlooked
eigenval_init_real_grid = np.linspace(-5, -1e-4, 5000) # lambda_1,...,lambda_{N_eigvals} must lie separated here  -- though the grid is automatically extended to hold N_eigvals (see real_eigenvalues)
#eigenval_init_real_grid = np.linspace(-2500.0, -1e-1, 10000) # lambda_1,...,lambda_{N_eigvals} must lie separated here 

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

# bring visualization to spectralsolver -- copy those then later for manuscript figures but genearl version is very nice

plot_paper_quantities = True

plot_full_spectrum_sigma = False
plot_full_spectrum_eigvals = False
plot_full_spectrum_complexplane = False # TODO, see maurizios draft for inspiration

plot_quantities = ['eigvals', 'real', 'composed'] #['eigvals', 'real', 'complex', 'composed'] #, 'eigvals', 'complex'] #['composed', 'eigvals', 'complex'] #['eigvals', 'real'] #['composed', 'complex', 'eigvals', 'real'] #['eigvals', 'real', 'complex'] #['eigvals', 'real', 'complex'] # or None
#complex_quants_plot = ['f_1', 'f_1*psi_r_1', 'psi_r_1']
#complex_quants_plot = ['f_2', 'f_2*psi_r_2', 'psi_r_2']
#complex_quants_plot = ['f_1', 'f_1*c_mu_1', 'c_mu_1']
#complex_quants_plot = ['f_2', 'f_2*c_mu_2', 'c_mu_2']
#complex_quants_plot = ['f_1', 'f_1*c_sigma_1', 'c_sigma_1']
#complex_quants_plot = ['f_2', 'f_2*c_sigma_2', 'c_sigma_2']
#complex_quants_plot = ['f_1', 'f_2', 'psi_r_1', 'psi_r_2', 'c_mu_1', 'c_mu_2', 'c_sigma_1', 'c_sigma_2']
#complex_quants_plot = ['f_1*psi_r_1', 'f_2*psi_r_2', 'f_1*c_mu_1', 'f_2*c_mu_2', 'f_1*c_sigma_1', 'f_2*c_sigma_2']
complex_quants_plot = ['f_1', 'f_1*c_mu_1', 'c_mu_1']

plot_summed_quantities = True
plot_validation = True # plot available quantities that were calculated by another method, as comparison

plot_real_inits = False
# corresponding params
sigma_smaller_raw = 1.5
sigma_larger_raw = 3.5
mu_min_raw = -1.0 # to be compatible with cascade models
lambda_real_grid = np.linspace(-0.6, 1e-5, 500)
N_eigs_min = 8 # in that interval above: sigma=1.5 => 10+1 eigvals, sigma=3.5 => 8+1 eigvals


plot_eigenfunctions = False
# corresponding params
mu_smaller_eigfun = 0.25
mu_larger_eigfun = 1.5
# furthermore use sigma_smaller_raw and sigma_larger_raw from above

# plotting parameters
colormap_sigma = 'winter'
no_sigma_quantities_plot = 'all' # min(4, len(sigma))  # 'all'
sigmas_plot_raw = [1.5, 3.5] #[1.2, 2.4]
# some more plotting parameters below (i.e., after loading)


# I. SPECTRUM OF FP OPERATOR -- COMPUTATION/LOADING/POSTPROCESSING

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
# enforcing complex conjugate pairs of eigenvalues (from the crossing to the right)
# it cannot be guaranteed that eigenvalue curves corresponding to complex conjugate pairs are following 
# each other, e.g. k=0 could correspond to one complex ev, k=1 is purely real, k=2 is the complex conj. to k=0
# transformation: lambda_all -> lambda_conjpairs

if postprocess_spectrum:
    
    conjugation_first_imag_negative = False
    
    print('enforcing complex conjugated pairs for lambda_all')
    spectrum_enforce_complex_conjugation(lambda_all, mu, sigma, 
                                  tolerance_conjugation=params['tolerance_conjugation'], # sec units: # tolerance_conjugation=1e-1,
                                  conjugation_first_imag_negative=conjugation_first_imag_negative)
    
        
    
    # select lambda_1 and lambda_2 -- the first two dominante eigenvalues for each mu, sigma
    # for neurons with lower bound == reset this is simply taking the first two lambda_all matrices
    # here the situation is more complicated due to the switching between dominant diffusive and regular modes
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

    quantities_dict['lambda_1'] = lambda_1
    quantities_dict['lambda_2'] = lambda_2
    
    if save_spec:

        specsolv.save_quantities(folder+'/'+filename, quantities_dict) 
    
        print('saving spectrum after postprocessing done.')

else:

    specsolv.load_quantities(folder+'/'+filename, quantities_dict, 
                             quantities=['lambda_1', 'lambda_2'], load_params=False)      


# II. QUANTITIES CORRESPONDING TO SPECTRUM ABOVE -- COMPUTATION/LOADING

# QUANTITY LOADING
quant_loaded = False
if load_quant:
                        
    quant_names = [ 'lambda_1', 'lambda_2', 
                    'r_inf', 'dr_inf_dmu', 'dr_inf_dsigma',
                    'V_mean_inf', 'dV_mean_inf_dmu', 'dV_mean_inf_dsigma',
                    'f_1', 'f_2', 'psi_r_1', 'psi_r_2',
                    'c_mu_1', 'c_mu_2', 'c_sigma_1', 'c_sigma_2',
                    'mu', 'sigma'] # load mu/sigma in case we have not loaded earlier (might be redundant but no problem)

    specsolv.load_quantities(folder+'/'+filename, quantities_dict, 
                             quantities=quant_names, load_params=load_params)
    quant_loaded = True

quant_computed = False
if compute_quant and not quant_loaded:  
    
    assert 'lambda_1' and 'lambda_2' in quantities_dict #postprocess_spectrum # we need to find lambda_1 and lambda_2 before this
    
    
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
    # note the following min/max/tol params can be used for all: Tref=0 (Vlb<Vr and Vlb=Vr), Tref=1.5
    quantities_postprocess(quantities_dict, 
                           quant_names=['lambda_1', 'lambda_2',
                                        'f_1', 'f_2', 
                                        'psi_r_1', 'psi_r_2',
                                        'c_mu_1', 'c_mu_2', 
                                        'c_sigma_1', 'c_sigma_2'], 
                            minsigma_interp=0.5, maxsigma_interp=5., maxmu_interp=0.52, 
                            tolerance_conjugation=params['tolerance_conjugation'])
    print(quantities_dict.keys())
    
    if save_quant:

        specsolv.save_quantities(folder+'/'+filename, quantities_dict)
        print(quantities_dict.keys())
    
        print('saving quantities after postprocessing done.')

# here we obtain the flux at the lower bound to be able to plot the _real_ initialization of the 
# spectral solver at mu_min
if obtain_fluxlb:
    fluxlb_quants = {}    
    
    specsolv.load_quantities(folder+'/'+filename, fluxlb_quants, 
                             quantities=
                                 ['lambda_real_grid'] 
                                 +['lambda_real_found_sigma{:.1f}'.format(sig) for sig 
                                     in [sigma_smaller_raw, sigma_larger_raw]]
                                 +['qlb_real_sigma{:.1f}'.format(sig) for sig 
                                     in [sigma_smaller_raw, sigma_larger_raw]], load_params=False)
                                 
    if 'lambda_real_grid' in fluxlb_quants:
        print('loading of the (]aw) real lambda/qlb data was sucessful')
    else:
        print('loading of the (]aw) real lambda/qlb data was NOT sucessful => computing!')                        
        
        fluxlb_quants['lambda_real_grid'] = lambda_real_grid
        
        specsolv.params['mu'] = mu_min_raw  
        for sig in [sigma_smaller_raw, sigma_larger_raw]:
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
    inds_sigma_plot_raw = [np.argmin(np.abs(sigma-sig)) for sig in sigmas_plot_raw]    
    plot_raw_spectrum_sigma(lambda_all, mu, sigma, inds_sigma_plot_raw)
    
if plot_full_spectrum_eigvals:
    plot_raw_spectrum_eigvals(lambda_all, mu, sigma)

if plot_full_spectrum_complexplane:
    raise NotImplementedError('complexplane plotting of the full spectrum is not yet implemented')

quantities_validation = {}
if plot_validation:
    
    file_josef_ref = 'precalc05_int.mat' # Tref=1.4    
    file_josef_noref = 'quantities_cascade_noref.h5' # Tref=0
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
        
#
#    # use precalc05_int.mat data for validation (of mean(V; p_inf(ref)) and r_inf as well as dr_dmu, dr_dsigma, ) first
#
#
#    # compare vs. maurizio matfiles (todo: compare with matfile from josef r_inf modelxomp)
#    assert params['conjugation_first_imag_negative']
#    
#    
#    
#    load_maurizio = lambda filename: scipy.io.loadmat(folder_maurizio+'/'+filename)['Expression1'].T
#    
#    quantities_validation = {'mu': load_maurizio('F1_X.mat').flatten() / 1000.,
#                             'sigma': load_maurizio('F1_Y.mat').flatten() / math.sqrt(1000.),
#                             'lambda_1': load_maurizio('Lambda1_Z.mat') / 1000.,
#                             'lambda_2': load_maurizio('Lambda-1_Z.mat') / 1000.,                             
#                             'r_inf': load_maurizio('Phin_Z.mat') / 1000.,
#                             'dr_inf_dmu': load_maurizio('DPhiDMun_Z.mat'),
#                             'dr_inf_dsigma': load_maurizio('DPhiDSigman_Z.mat') / math.sqrt(1000.), # * sigma?
#                             # V_mean and its derivatives w.r.t mu/sigma not yet available from Maurizio
#                             'f_1': load_maurizio('F1_Z.mat') / 1000.,
#                             'f_2': load_maurizio('F-1_Z.mat') / 1000.,
#                             'psi_r_1': load_maurizio('Psi1_Z.mat'),
#                             'psi_r_2': load_maurizio('Psi-1_Z.mat'),
#                             'c_mu_1': load_maurizio('cMu1_Z.mat') * 1000.,
#                             'c_mu_2': load_maurizio('cMu-1_Z.mat') * 1000.,
#                             'c_sigma_1': load_maurizio('DcDsigma1_Z.mat') * math.sqrt(1000.),
#                             'c_sigma_2': load_maurizio('DcDsigma-1_Z.mat') * math.sqrt(1000.)
#                            }
#    # phi_inf can be computed analytically due to Mattia02 PRE
#    mu_ana = quantities_validation['mu'] * 1000. # here we need seconds units
#    sigma_ana = quantities_validation['sigma'] * math.sqrt(1000.)
#    mu_ana = mu_ana.reshape(-1, 1)
#    sigma_ana = sigma_ana.reshape(1, -1)
#    xi = mu_ana/sigma_ana**2
#    c = 1./(sigma_ana**2/(2*mu_ana**2) * (2*mu_ana/sigma_ana**2 - 1 + np.exp(-2*mu_ana/sigma_ana**2)))
#    V_ana = np.linspace(0, 1, params['grid_V_points'])
#    V_mean_inf_ana = np.zeros_like(quantities_validation['r_inf'])    
#    for i in range(len(mu_ana)):
#        for j in range(len(sigma_ana)):
#            phi_ana_ij = c[i,j]/mu_ana[i] * (1-np.exp(-2*xi[i,j]*(1-V_ana)))
#            V_mean_inf_ana[i,j] = np.sum(np.diff(V_ana)*0.5*((V_ana*phi_ana_ij)[:-1]+(V_ana*phi_ana_ij)[1:]))
#    quantities_validation['V_mean_inf'] = V_mean_inf_ana
    # the mu/sigma derivatives of V_mean_inf could be added here too using, e.g., central finite differences
    
# the following multiplied quantities are generated in plot_quantities_complex
#    quantities_validation['f_1*psi_r_1'] = quantities_validation['f_1'] * quantities_validation['psi_r_1']
#    quantities_validation['f_2*psi_r_2'] = quantities_validation['f_2'] * quantities_validation['psi_r_2']
#    quantities_validation['f_1*c_mu_1'] = quantities_validation['f_1'] * quantities_validation['c_mu_1']
#    quantities_validation['f_2*c_mu_2'] = quantities_validation['f_2'] * quantities_validation['c_mu_2']
#    quantities_validation['f_1*c_sigma_1'] = quantities_validation['f_1'] * quantities_validation['c_sigma_1']
#    quantities_validation['f_2*c_sigma_2'] = quantities_validation['f_2'] * quantities_validation['c_sigma_2']
                            

if plot_quantities and 'eigvals' in plot_quantities:
    
    plot_quantities_eigvals(quantities_dict, inds_sigma_plot, colormap_sigma=colormap_sigma,
                            plot_validation=plot_validation, quantities_validation=quantities_validation)

    
if plot_quantities and 'real' in plot_quantities:
    
    plot_quantities_real(quantities_dict, inds_sigma_plot, colormap_sigma=colormap_sigma,
                         plot_validation=plot_validation, quantities_validation=quantities_validation)
    
    

if plot_quantities and 'complex' in plot_quantities:
    
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
    for sig in [sigma_smaller_raw, sigma_larger_raw]:
        
        lambda_real_grid = fluxlb_quants['lambda_real_grid']
        lambda_real_found = fluxlb_quants['lambda_real_found_sigma{:.1f}'.format(sig)]
        
        plt.subplot(1, 2, sid)
        plt.title('$\mu={}, \sigma={}$'.format(mu_min_raw, sig))
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
    
    xlim = [-100, -40]
#    xlim = [-200, -40]
    
    plt.figure()

    plt.suptitle('eigenfunctions in a.u. truncated at V={}'.format(xlim[0]))
    
    plt.subplot(2, 2, 1)
    
    # REGULAR MODE -- REAL
    # small mu, small sigma
    mu_i = mu_smaller_eigfun
    sigma_j = sigma_smaller_raw
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
    sigma_j = sigma_larger_raw
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

    # validation
#    print('|phi1.imag|={}'.format(np.linalg.norm(phi1.imag)))
#    print('|psi1.imag|={}'.format(np.linalg.norm(psi1.imag)))
#    print('|phi0.imag|={}'.format(np.linalg.norm(phi0.imag)))
    
    
    plt.subplot(2, 2, 2)
    
    # REGULAR MODE -- COMPLEX
    # larger mu, smaller sigma
    mu_i = mu_larger_eigfun
    sigma_j = sigma_smaller_raw
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
    sigmas_plot = [1.5, 3.5]
    mus_plot = [1.5] # 0.25
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
#        plt.plot(mu_vals[inds_mu_plot], -1/quantities_dict['lambda_2'].real[inds_mu_plot,j], 
#                 color=linecolor2)
        plt.plot(mu_vals[inds_mu_plot][plotinds_lim], -1/quantities_dict['lambda_1'].real[inds_mu_plot,j][plotinds_lim], 
                 label=siglabel, color=linecolor)
        for l in range(len(sigmas_plot)): 
            if np.round(sigmas_plot[l],2)==np.round(sigma_vals[j],2):
                plt.plot(mus_plot[0], -1/quantities_dict['lambda_1'].real[mu_plot_ind,j], 
                'o', color=linecolor, markersize=5)#, markeredgewidth=0.0)
        if k_j==0:
            plt.title(r'$-1/\mathrm{Re}\lambda_1$', fontsize=14)
            plt.ylabel('[ms]', fontsize=12)
#            plt.xlabel('$\mu$ [mV/ms]', fontsize=12)
            plt.ylim(ylim)
            plt.yticks(ylim)
#        plt.plot(mu_vals[inds_mu_plot], quantities_dict['lambda_1'].real[inds_mu_plot,j], 
#                 label=siglabel, color=linecolor)
#        for l in range(len(sigmas_plot)): 
#            if np.round(sigmas_plot[l],2)==np.round(sigma_vals[j],2):
#                plt.plot(mus_plot[0], quantities_dict['lambda_1'].real[mu_plot_ind,j], 
#                'o', color=linecolor, markersize=5)#, markeredgewidth=0.0)
#        if k_j==0:
#            plt.title(r'$\mathrm{Re}\lambda_1$', fontsize=14)
#            plt.ylabel('[kHz]', fontsize=12)
##            plt.xlabel('$\mu$ [mV/ms]', fontsize=12)
#            ylim = [-0.7,0.0]
#            plt.ylim(ylim)
#            plt.yticks(ylim)


        if k_j==len(inds_sigma_plot)-1:
            plt.legend()
            
            
        siglabel = None 
        
        
        plt.subplot(4, 4, 2)

        plt.plot(mu_vals[inds_mu_plot], quantities_dict['lambda_1'].imag[inds_mu_plot,j]/(2*np.pi), 
                 label=siglabel, color=linecolor)
        for l in range(len(sigmas_plot)): 
            if np.round(sigmas_plot[l],2)==np.round(sigma_vals[j],2):
                plt.plot(mus_plot[0], quantities_dict['lambda_1'].imag[mu_plot_ind,j]/(2*np.pi), 
                'o', color=linecolor, markersize=5)#, markeredgewidth=0.0)
        if k_j==0:
            plt.title(r'$\mathrm{Im}\lambda_1 / (2\pi)$', fontsize=14)
            plt.ylabel('[kHz]', fontsize=12)
#            plt.xlabel('$\mu$ [mV/ms]', fontsize=12)
            ylim = [-.005,0.14]
            plt.ylim(ylim)
            plt.yticks([0,.14])
            
#        plt.plot(mu_vals[inds_mu_plot], quantities_dict['lambda_1'].imag[inds_mu_plot,j], 
#                 label=siglabel, color=linecolor)
#        for l in range(len(sigmas_plot)): 
#            if np.round(sigmas_plot[l],2)==np.round(sigma_vals[j],2):
#                plt.plot(mus_plot[0], quantities_dict['lambda_1'].imag[mu_plot_ind,j], 
#                'o', color=linecolor, markersize=5)#, markeredgewidth=0.0)
#        if k_j==0:
#            plt.title(r'$\mathrm{Im}\lambda_1$', fontsize=14)
#            plt.ylabel('[kHz]', fontsize=12)
##            plt.xlabel('$\mu$ [mV/ms]', fontsize=12)
#            ylim = [-.02,.9]
#            plt.ylim(ylim)
#            plt.yticks([0,.9])
            
            
        plt.subplot(4, 4, 5)
                
        if k_j>0:
            plotinds_lim = -1/quantities_dict['lambda_2'].real[inds_mu_plot,j] <= ylim[1]
        else: 
            plotinds_lim = -1/quantities_dict['lambda_2'].real[inds_mu_plot,j] <= 1000.
#        plt.plot(mu_vals[inds_mu_plot], quantities_dict['lambda_2'].imag[inds_mu_plot,j]/(2*np.pi), 
#                 label=siglabel, color=linecolor2)
#        plt.plot(mu_vals[inds_mu_plot], quantities_dict['lambda_1'].imag[inds_mu_plot,j]/(2*np.pi), 
#                 label=siglabel, color=linecolor)
        plt.plot(mu_vals[inds_mu_plot][plotinds_lim], -1/quantities_dict['lambda_2'].real[inds_mu_plot,j][plotinds_lim], 
                 label=siglabel, color=linecolor)
        for l in range(len(sigmas_plot)): 
            if np.round(sigmas_plot[l],2)==np.round(sigma_vals[j],2):
                plt.plot(mus_plot[0], -1/quantities_dict['lambda_2'].real[mu_plot_ind,j], 
                'o', color=linecolor, markersize=5)#, markeredgewidth=0.0)
        if k_j==0:
            plt.title(r'$-1/\mathrm{Re}\lambda_2$', fontsize=14)
            plt.ylabel('[ms]', fontsize=12)
            plt.xlabel('$\mu$ [mV/ms]', fontsize=12)
            ylim = [0,50]
            plt.ylim(ylim)
            plt.yticks(ylim)
#        plt.plot(mu_vals[inds_mu_plot], quantities_dict['lambda_2'].real[inds_mu_plot,j], 
#                 label=siglabel, color=linecolor)
#        for l in range(len(sigmas_plot)): 
#            if np.round(sigmas_plot[l],2)==np.round(sigma_vals[j],2):
#                plt.plot(mus_plot[0], quantities_dict['lambda_2'].real[mu_plot_ind,j], 
#                'o', color=linecolor, markersize=5)#, markeredgewidth=0.0)
#        if k_j==0:
#            plt.title(r'$\mathrm{Re}\lambda_2$', fontsize=14)
#            plt.ylabel('[kHz]', fontsize=12)
#            plt.xlabel('$\mu$ [mV/ms]', fontsize=12)
#            ylim = [-0.7,0.0]
#            plt.ylim(ylim)
#            plt.yticks(ylim)
            
        
        plt.subplot(4, 4, 6)

        plt.plot(mu_vals[inds_mu_plot], quantities_dict['lambda_2'].imag[inds_mu_plot,j]/(2*np.pi), 
                 label=siglabel, color=linecolor)
        for l in range(len(sigmas_plot)): 
            if np.round(sigmas_plot[l],2)==np.round(sigma_vals[j],2):
                plt.plot(mus_plot[0], quantities_dict['lambda_2'].imag[mu_plot_ind,j]/(2*np.pi), 
                'o', color=linecolor, markersize=5)#, markeredgewidth=0.0)
        if k_j==0:
            plt.title(r'$\mathrm{Im}\lambda_2 / (2\pi)$', fontsize=14)
            plt.ylabel('[kHz]', fontsize=12)
            plt.xlabel('$\mu$ [mV/ms]', fontsize=12)
            ylim = [-0.14,.005]
            plt.ylim(ylim)
            plt.yticks([-.14, 0])
            
        
        fcmu = (quantities_dict['f_1']*quantities_dict['c_mu_1'] + 
                quantities_dict['f_2']*quantities_dict['c_mu_2']).real
#        plt.subplot(2, 6, 3)
#
#        plt.plot(mu_vals[inds_mu_plot], fcmu[inds_mu_plot,j], 
#                 label=siglabel, color=linecolor)
#        for l in range(len(sigmas_plot)): 
#            if np.round(sigmas_plot[l],2)==np.round(sigma_vals[j],2):
#                plt.plot(mus_plot[0], fcmu[mu_plot_ind,j], 
#                'o', color=linecolor, markersize=5)#, markeredgewidth=0.0)
#        if k_j==0:
#            plt.title(r'$\mathbf{f}\cdot\mathbf{c}_\mu$', fontsize=14)
#            plt.ylabel('[1/mV]', fontsize=12)
##            plt.xlabel('$\mu$ [mV/ms]', fontsize=12)
#            ylim = [-.06,0.001]
#            plt.ylim(ylim)
#            plt.yticks([-.06,0.])
                
        plt.subplot(4, 4, 3)

        plt.plot(mu_vals[inds_mu_plot], (fcmu + quantities_dict['dr_inf_dmu'])[inds_mu_plot,j], 
                 label=siglabel, color=linecolor)
        for l in range(len(sigmas_plot)): 
            if np.round(sigmas_plot[l],2)==np.round(sigma_vals[j],2):
                plt.plot(mus_plot[0], (fcmu + quantities_dict['dr_inf_dmu'])[mu_plot_ind,j], 
                'o', color=linecolor, markersize=5)#, markeredgewidth=0.0)
        if k_j==0:
            plt.title(r'$M$', fontsize=14)
            plt.ylabel('[1/mV]', fontsize=12)
#            plt.xlabel('$\mu$ [mV/ms]', fontsize=12)
            ylim = [0.,0.035]
            plt.ylim(ylim)
            plt.yticks(ylim)
        
#        
#        fcsigma2 = (quantities_dict['f_1']*quantities_dict['c_sigma_1'] + 
#                quantities_dict['f_2']*quantities_dict['c_sigma_2']).real / (2*sigma_vals[j]) # scale for dsigma->dsigma2
#        plt.subplot(2, 6, 9)
#
#        plt.plot(mu_vals[inds_mu_plot], fcsigma2[inds_mu_plot,j], 
#                 label=siglabel, color=linecolor)
#        for l in range(len(sigmas_plot)): 
#            if np.round(sigmas_plot[l],2)==np.round(sigma_vals[j],2):
#                plt.plot(mus_plot[0], fcsigma2[mu_plot_ind,j], 
#                'o', color=linecolor, markersize=5)#, markeredgewidth=0.0)
#        if k_j==0:
#            plt.title(r'$\mathbf{f}\cdot\mathbf{c}_{\sigma^2}$', fontsize=14)
#            plt.ylabel('[1/mV$^2$]', fontsize=12)
#            plt.xlabel('$\mu$ [mV/ms]', fontsize=12)
#            ylim = [-.007,0.008]
#            plt.ylim(ylim)
#            plt.yticks(ylim)
        
                
        fcsigma2 = (quantities_dict['f_1']*quantities_dict['c_sigma_1'] + 
                quantities_dict['f_2']*quantities_dict['c_sigma_2']).real / (2*sigma_vals[j]) # scale for dsigma->dsigma2
        dr_inf_dsigma2 = quantities_dict['dr_inf_dsigma'] / (2*sigma_vals[j])
        plt.subplot(4, 4, 7)

        plt.plot(mu_vals[inds_mu_plot], (fcsigma2+dr_inf_dsigma2)[inds_mu_plot,j], 
                 label=siglabel, color=linecolor)
        for l in range(len(sigmas_plot)): 
            if np.round(sigmas_plot[l],2)==np.round(sigma_vals[j],2):
                plt.plot(mus_plot[0], (fcsigma2+dr_inf_dsigma2)[mu_plot_ind,j], 
                'o', color=linecolor, markersize=5)#, markeredgewidth=0.0)
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
                'o', color=linecolor, markersize=5)#, markeredgewidth=0.0)
        if k_j==0:
            plt.title(r'$F_\mu$', fontsize=14)
            plt.ylabel('[kHz/mV]', fontsize=12)
#            plt.xlabel('$\mu$ [mV/ms]', fontsize=12)
            ylim = [-.00001,0.005]
            plt.ylim(ylim)
            plt.yticks([.0,0.005])
            
            
        
        F_sigma2 = (quantities_dict['f_1']*quantities_dict['c_sigma_1']*quantities_dict['lambda_1'] + 
                    quantities_dict['f_2']*quantities_dict['c_sigma_2']*quantities_dict['lambda_2']).real / (2*sigma_vals[j]) # scale for dsigma->dsigma2
        plt.subplot(4, 4, 8)

        plt.plot(mu_vals[inds_mu_plot], F_sigma2[inds_mu_plot,j], 
                 label=siglabel, color=linecolor)
        for l in range(len(sigmas_plot)): 
            if np.round(sigmas_plot[l],2)==np.round(sigma_vals[j],2):
                plt.plot(mus_plot[0], F_sigma2[mu_plot_ind,j], 
                'o', color=linecolor, markersize=5)#, markeredgewidth=0.0)
        if k_j==0:
            plt.title(r'$F_{\sigma^2}$', fontsize=14)
            plt.ylabel('[kHz/mV$^2$]', fontsize=12)
            plt.xlabel('$\mu$ [mV/ms]', fontsize=12)
            ylim = [-.001,0.001]
            plt.ylim(ylim)
            plt.yticks(ylim)
            
            
            
#        plt.subplot(2, 6, 5)
#
#        plt.plot(mu_vals[inds_mu_plot], quantities_dict['dr_inf_dmu'][inds_mu_plot,j], 
#                 label=siglabel, color=linecolor, lw=1.5)
#        for l in range(len(sigmas_plot)): 
#            if np.round(sigmas_plot[l],2)==np.round(sigma_vals[j],2):
#                plt.plot(mus_plot[0], quantities_dict['dr_inf_dmu'][mu_plot_ind,j], 
#                'o', color=linecolor, markersize=5)#, markeredgewidth=0.0)
#        if k_j==0:
#            plt.title(r'$\partial_\mu r_\infty$', fontsize=14)
#            plt.ylabel('[1/mV]', fontsize=12)
##            plt.xlabel('$\mu$ [mV/ms]', fontsize=12)
#            ylim = [-.0005,0.08]
#            plt.ylim(ylim)
#            plt.yticks([.0,0.08])
#            
#            
#        dr_inf_dsigma2 = quantities_dict['dr_inf_dsigma'] / (2*sigma_vals[j])
#        plt.subplot(2, 6, 11)
#
#        plt.plot(mu_vals[inds_mu_plot], dr_inf_dsigma2[inds_mu_plot,j], 
#                 label=siglabel, color=linecolor, lw=1.5)
#        for l in range(len(sigmas_plot)): 
#            if np.round(sigmas_plot[l],2)==np.round(sigma_vals[j],2):
#                plt.plot(mus_plot[0], dr_inf_dsigma2[mu_plot_ind,j], 
#                'o', color=linecolor, markersize=5)#, markeredgewidth=0.0)
#        if k_j==0:
#            plt.title(r'$\partial_{\sigma^2} r_\infty$', fontsize=14)
#            plt.ylabel('[1/mV$^2$]', fontsize=12)
#            plt.xlabel('$\mu$ [mV/ms]', fontsize=12)
#            ylim = [-.0005,0.011]
#            plt.ylim(ylim)
#            plt.yticks([.0,0.011])
#            
#            
#        plt.subplot(2, 6, 6)
#
#        plt.plot(mu_vals[inds_mu_plot], quantities_dict['dV_mean_inf_dmu'][inds_mu_plot,j], 
#                 label=siglabel, color=linecolor, lw=1.5)
#        for l in range(len(sigmas_plot)): 
#            if np.round(sigmas_plot[l],2)==np.round(sigma_vals[j],2):
#                plt.plot(mus_plot[0], quantities_dict['dV_mean_inf_dmu'][mu_plot_ind,j], 
#                'o', color=linecolor, markersize=5)#, markeredgewidth=0.0)
#        if k_j==0:
#            plt.title(r'$\partial_\mu \langle V \rangle_\infty$', fontsize=14)
#            plt.ylabel('[ms]', fontsize=12)
#            plt.xlabel('$\mu$ [mV/ms]', fontsize=12)
#            ylim = [-11,22]
#            plt.ylim(ylim)
#            plt.yticks(ylim)
#            
#            
#        dV_mean_dsigma2 = quantities_dict['dV_mean_inf_dsigma'] / (2*sigma_vals[j])
#        plt.subplot(2, 6, 12)
#
#        plt.plot(mu_vals[inds_mu_plot], dV_mean_dsigma2[inds_mu_plot,j], 
#                 label=siglabel, color=linecolor, lw=1.5)
#        for l in range(len(sigmas_plot)): 
#            if np.round(sigmas_plot[l],2)==np.round(sigma_vals[j],2):
#                plt.plot(mus_plot[0], dV_mean_dsigma2[mu_plot_ind,j], 
#                'o', color=linecolor, markersize=5)#, markeredgewidth=0.0)
#        if k_j==0:
#            plt.title(r'$\partial_{\sigma^2} \langle V \rangle_\infty$', fontsize=14)
#            plt.ylabel('[ms/mV]', fontsize=12)
#            plt.xlabel('$\mu$ [mV/ms]', fontsize=12)
#            ylim = [-5,0.2]
#            plt.ylim(ylim)
#            plt.yticks(ylim)
    
#    plt.savefig('fig_spec_quantities.svg')
    
    
    
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
    for l_j, j in enumerate(sigma_inds_fig): #  range(0, N_sigma, sigma_skip_inds):
        
        
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

    
#    plt.savefig('fig_spec_rawspectrum.svg')



    
    sigma_smaller_raw = 1.5
    sigma_larger_raw = 3.5
    mu_min_raw = -1.0 # to be compatible with cascade models
    lambda_real_grid = np.linspace(-0.54, 1e-5, 500)
    N_eigs_min = 8 # in that interval above: sigma=1.5 => 10+1 eigvals, sigma=3.5 => 8+1 eigvals
    
    
    plt.figure()
    sid = 1
    for sig in [sigma_smaller_raw, sigma_larger_raw]:
        
        lambda_real_grid = fluxlb_quants['lambda_real_grid']
        lambda_real_found = fluxlb_quants['lambda_real_found_sigma{:.1f}'.format(sig)]
        
        plt.subplot(1, 2, sid)
        plt.title('$\mu={}, \sigma={}$'.format(mu_min_raw, sig))
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
            
#    plt.savefig('fig_spec_fluxlb.svg')
    
    
    
    xlim = [-100, -40]
    skippts = 50
    xticks = [-100, -70, -40]
    ylim = [-1,1]
    col_phi0 = 'gray'
    col_regular = 'blue'
    col_diffusive = 'red'
#    xlim = [-200, -40]
    
    
    plt.figure()

    plt.suptitle('eigenfunctions in a.u. truncated at V={}'.format(xlim[0]))
    
    plt.subplot(2, 2, 1)
    
    # REGULAR MODE -- REAL
    # small mu, small sigma
    mu_i = mu_smaller_eigfun
    sigma_j = sigma_smaller_raw
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
    print(plotinds)
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
    sigma_j = sigma_larger_raw
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
#    plt.legend()
    plt.xlabel('V [mV]') 
    plt.ylabel('real function [a.u.]')
    plt.xticks(xticks)
    plt.ylim(ylim)
    plt.yticks(ylim)

    # validation
#    print('|phi1.imag|={}'.format(np.linalg.norm(phi1.imag)))
#    print('|psi1.imag|={}'.format(np.linalg.norm(psi1.imag)))
#    print('|phi0.imag|={}'.format(np.linalg.norm(phi0.imag)))
    
    
    plt.subplot(2, 2, 2)
    
    # REGULAR MODE -- COMPLEX
    # larger mu, smaller sigma
    mu_i = mu_larger_eigfun
    sigma_j = sigma_smaller_raw
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
#    plt.legend()
#    plt.xlabel('V [mV]') 
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
#    plt.legend()
    plt.xlabel('V [mV]') 
    plt.ylabel('imag. part [a.u.]')
    plt.xticks(xticks)
    plt.ylim(ylim)
    plt.yticks(ylim)
    
#    plt.savefig('fig_spec_eigfuncs.svg')
        
plt.show()
