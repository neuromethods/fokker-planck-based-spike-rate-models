from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(1,'..') # allow parent modules to be imported
sys.path.insert(1,'../..') # allow parent modules to be imported
sys.path.insert(1,'../../..') # allow parent modules to be imported
import time
import params
from misc.utils import generate_OUinput, x_filter, get_changing_input, interpolate_input
import models.brian2.network_sim as net
import models.fvm_sg.fokker_planck as fp_sg
import models.ln_exp.ln_exp_model as lnexp
import models.ln_dos.ln_dosc_model as lndosc
import models.ln_bexdos.ln_bexdos_model as lnbexdos
import models.spec1.spectral1_model as s1
import models.spec2.spectral2m_model as s2m
import models.spec2_red.spectral2m_simple_model as s2m_simple

# use the following in IPython for qt plots: %matplotlib qt


# what will be computed

# network simulation
run_network =  True
# full fokker planck model
run_fp =       True

# reduced models
# ln cascade
run_ln_exp =   True
run_ln_dosc=   True
run_ln_bexdos = False

# spectral
run_spec1    =    True
run_spec2m   =    True
run_spec2m_simple = False


# use as default the parameters from file params.py
# if not specified else below
params = params.get_params()

# runtime options
# run simulation of uncoupled (rec=False) or recurrently coupled simulation (rec=True)
rec = True

params['runtime'] = 3000.
# number of neurons
params['N_total'] = 4000 #50000
# time steps for models
params['uni_dt'] = 0.01 # [ms]
params['fp_dt'] = 0.05
params['net_dt'] = 0.05





# coupling (and delay) params in the case of recurrency, i.e. rec = True
params['K'] =  100
params['J'] = 0.05
params['delay_type'] = 2
params['taud'] = 3.
params['const_delay'] = 5.


# adaptation params as scalars
params['a'] = 4.
params['b'] = 40.


# [only for reduced models] switch between two different time integration schemes: (1) Euler, (2) Heun
params['uni_int_order'] = 2


# for generating the input; for all models which do
# not have the same resolution we have to interpolate
params['min_dt'] = min(params['uni_dt'], params['net_dt'],params['fp_dt'])


ln_data = 'quantities_cascade.h5'
spec_data = 'quantities_spectral.h5'
params['t_ref'] = 0.0

# plotting section
plot_rates = True
plot_input = True

plot_adapt = True and (params['a'] > 0 or params['b'] > 0)

# external input mean
# for the external input mean and the standard deviation any type of input may be defined, such as constant, step, ramp

input_mean = 'steps' # similar to Fig1 of manuscript
# input_mean = 'osc'
# input_mean = 'const'
# input_mean = 'OU'
# input_mean = 'ramp'

# filter input mean (necessary for spectral_2m model)
filter_mean = True

#input_std = 'const'
#input_std = 'step'
#input_std = 'OU'
input_std = 'ramp'
filter_std = True

# external time trace used for generating input and plotting
# if time step is unequal to model_dt input gets interpolated for
# the respective model
steps = int(params['runtime']/params['min_dt'])
t_ext = np.linspace(0., params['runtime'], steps+1)

# time trace computed with min_dt
params['t_ext'] = t_ext

# for filter testing set seed
# np.random.seed(3)

# mu_ext variants
if input_mean == 'const':
    mu_ext = np.ones(steps+1) * 4.0

# mu = OU process, sigma = const
elif input_mean == 'OU':
    params['ou_X0'] = 0.
    params['ou_mean']  = 6.0
    params['ou_sigma'] = .5
    params['ou_tau']   = 50.
    mu_ext = generate_OUinput(params)

# oscillating input
elif input_mean == 'osc':
    freq = 0.005 #kHz
    amp = 0.1  #mV/ms
    offset = 0.5  #mV/ms
    mu_ext = offset*np.ones(len(t_ext)) + amp*np.sin(2*np.pi*freq*t_ext)

# input is ramped over a certain time interval from mu_start to mu_end
elif input_mean == 'ramp':
    # define parameters for input
    ramp_start = 500.
    assert ramp_start < params['runtime']
    ramp_duration = 30.
    mu_start = 2.
    mu_end = 4.
    mu_ext = get_changing_input(params['runtime'],
                                ramp_start,params['min_dt'],mu_start,
                                mu_end,duration_change=ramp_duration)

# step input scenario for mean input
elif input_mean == 'steps':
    # vals for steps
    vals = [1, 1, 1, 1, 1, 1.7,
            1.3,2.7, 2.4, 3.5,
            3,3.4, 4.1, 3.7, 3.5,
            2.5,3,3.5, 2, 2.5]

    params['vals'] = vals
    params['duration_vals'] = 150.


    def step_plateaus_up_down(params):
        steps = int(params['runtime']/params['min_dt'])
        trace = np.zeros(steps+1)
        val_idx = int(params['duration_vals']/params['min_dt'])
        assert params['runtime']%params['duration_vals']==0
        assert len(vals)*params['duration_vals'] == params['runtime']
        for i in xrange(len(params['vals'])):
            trace[i*val_idx:i*val_idx+val_idx] = params['vals'][i]
        return trace

    mu_ext=step_plateaus_up_down(params)


# sigma_ext variants
if input_std == 'const':
    sigma_ext = np.ones(steps+1) * 2.


elif input_std == 'step':
    sigma_ext = np.ones(steps+1)* 4.0
    sigma_ext[int(steps/3):int(2*steps/3)] = 3.0
    sigma_ext[int(2*steps/3):] = 1.5
    

# mu = const, sigma = OU process
elif input_std == 'OU':
    params['ou_X0'] =  0. #only relevant if params['ou_stationary'] = False
    params['ou_mean']  = 3.0
    params['ou_sigma'] = 1.2
    params['ou_tau']   = 1.
    sigma_ext = generate_OUinput(params)

elif input_std == 'ramp':
    # define parameters for input
    ramp_start = 1500.
    assert ramp_start < params['runtime']
    ramp_duration = 100.
    sigma_start = 3.5
    sigma_end = 1.5
    sigma_ext = get_changing_input(params['runtime'],ramp_start, params['min_dt'],sigma_start,
                                   sigma_end,duration_change=ramp_duration)
else:
    raise NotImplementedError


# enforce in any case sufficiently large input
mu_min = -1.0
mu_ext[mu_ext < mu_min] = mu_min - (mu_ext[mu_ext < mu_min] - mu_min)
mu_max = 5.
mu_ext[mu_ext > mu_max] = mu_max - (mu_ext[mu_ext > mu_max] - mu_max)
sigma_min = 0.5
sigma_ext[sigma_ext < sigma_min] = sigma_min - (sigma_ext[sigma_ext < sigma_min] - sigma_min)
sigma_max = 5.
sigma_ext[sigma_ext > sigma_max] = sigma_max - (sigma_ext[sigma_ext > sigma_max] - sigma_max)

# filter the input in order to have not sharp edges 
# filter params
params['filter_type'] = 'gauss'
# filter width in time domain ~ 6*filter_gauss_sigma
# -> keep that in mind for resolution issues

params['filter_gauss_sigma'] = 1. #1 for ramps, 0.1-0.5 for OU
if filter_mean:
    mu_ext_orig = mu_ext
    mu_ext = x_filter(mu_ext_orig, params)
if filter_std:
    sigma_ext_orig = sigma_ext
    sigma_ext = x_filter(sigma_ext_orig, params)

# collect ext input for model wrappers
ext_input0 = [mu_ext, sigma_ext]

# saving results in global results dict
model_results = dict()
model_results['input_mean'] = mu_ext
model_results['input_sigma']= sigma_ext
model_results['run_model'] = list()



print('\nModels run in {} mode.\n'.format('recurrent' if rec else 'feedforward'))

# simulation of the brian network
# input is computed with respect to the smallest time step (min(uni_dt, net_dt, fp_dt)) and subsequently interpolated for all other models

if run_network:
    ext_input = interpolate_input(ext_input0,params,'net')
    # use the brian simulation with a,b= array
    model_results['results_net'] = \
        net.network_sim(ext_input, params, rec = rec)
    model_results['run_model'].append('net')

#fokker planck equation solved using the Scharfetter Gummel representation
if run_fp:
    ext_input = interpolate_input(ext_input0, params, 'fp')
    model_results['results_fp'] = \
        fp_sg.sim_fp_sg(ext_input, params, rec=rec)
    model_results['run_model'].append('fp')

#reduced models
if run_ln_exp:
    ext_input = interpolate_input(ext_input0, params, 'reduced')
    model_results['results_ln_exp'] = \
        lnexp.run_ln_exp(ext_input, params, ln_data,
                         rec_vars= params['rec_lne'], rec= rec)
    model_results['run_model'].append('ln_exp')

if run_ln_dosc:
    ext_input = interpolate_input(ext_input0, params, 'reduced')
    # currently uses the 'old' ln dosc version
    model_results['results_ln_dosc'] = \
        lndosc.run_ln_dosc(ext_input, params,ln_data,
                           rec_vars= params['rec_lnd'],
                           rec= rec)
    model_results['run_model'].append('ln_dosc')

if run_ln_bexdos:
    ext_input = interpolate_input(ext_input0, params, 'reduced')
    model_results['results_ln_bexdos'] = \
        lnbexdos.run_ln_bexdos(ext_input, params,ln_data,
                               rec_vars=['wm'], rec = rec)
    model_results['run_model'].append('ln_bexdos')


if run_spec1:
    ext_input = interpolate_input(ext_input0, params, 'reduced')
    model_results['results_spec1'] = \
        s1.run_spec1(ext_input, params, spec_data,
                     rec_vars=params['rec_s1'],
                     rec = rec)
    model_results['run_model'].append('spec1')

if run_spec2m:
    ext_input = interpolate_input(ext_input0, params, 'reduced')
    model_results['results_spec2m'] = \
        s2m.run_spec2m(ext_input, params, spec_data,
                       rec_vars=['wm'],
                       rec=rec)
    model_results['run_model'].append('spec2m')

if run_spec2m_simple:
    ext_input = interpolate_input(ext_input0, params, 'reduced')
    model_results['results_spec2m_simple'] = \
        s2m_simple.run_spec2m_simple(ext_input, params, rec_vars=params['rec_sm'],
                                     rec=rec, filename_h5 = spec_data)
    model_results['run_model'].append('spec2m_simple')




# plotting section
# what will be plottet
nr_p = plot_rates + plot_adapt + plot_input
fig = plt.figure(); pidx = 1
if plot_input:
    ax_mu = fig.add_subplot(nr_p, 1, pidx)
    plt.plot(t_ext, mu_ext_orig, color = 'k', lw=1.5) if filter_mean else 0
    line_mu_final = plt.plot(t_ext, ext_input0[0], color = 'm', lw=1.5, label='$\mu_\mathrm{final}$')
    plt.ylabel('$\mu_{ext}$ [mV/ms]', fontsize=15)
    ax_sig = plt.twinx()    
    plt.plot(t_ext, sigma_ext_orig, color = 'g', lw=1.5) if filter_std else 0
    line_sig_final = plt.plot(t_ext, ext_input0[1], color = 'b', lw=1.5, label='$\sigma_\mathrm{final}$')
    plt.ylabel('$\sigma_{ext}$ [$\sqrt{mV}$/ms]', fontsize=15)
    plt.legend([line_mu_final[0], line_sig_final[0]], 
               [line_mu_final[0].get_label(), line_sig_final[0].get_label()])
    pidx +=1

if plot_rates:
    ax_rate = fig.add_subplot(nr_p, 1, pidx, sharex=ax_mu)
    for model in model_results['run_model']:
        color = params['color'][model]
        lw = params['lw'][model]
        key = 'results_{}'.format(model)
        time = model_results[key]['t']
        rates = model_results[key]['r']
        plt.plot(time, rates, label = model, color = color, lw=lw)
        plt.ylabel('r [Hz]')
        plt.legend()
    pidx += 1

# plot adaptation current
if plot_adapt:
    ax_adapt = fig.add_subplot(nr_p, 1, pidx, sharex=ax_mu)
    for model in model_results['run_model']:
        color = params['color'][model]
        lw = params['lw'][model]
        key = 'results_{}'.format(model)
        time = model_results[key]['t']
        print(model)
        wm = model_results[key]['wm']
        wm_shape = wm.shape
        time_shape = time.shape
        plt.ylabel('<wm> [pA]')
        plt.plot(time, wm, color = color, lw = lw)

    # plot also mean+std/mean-std if net was computed
    if 'net' in model_results['run_model']:
        time = model_results['results_net']['t']
        wm = model_results['results_net']['wm']
        w_std = model_results['results_net']['w_std']
        wm_plus = wm + w_std
        wm_minus = wm - w_std
        plt.fill_between(time,wm_minus, wm_plus, color = 'lightpink')


if nr_p: plt.show()





