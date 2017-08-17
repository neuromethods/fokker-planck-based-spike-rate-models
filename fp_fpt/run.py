from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(1,'..') # allow parent modules to be imported
sys.path.insert(1,'../..') # allow parent modules to be imported
sys.path.insert(1,'../../..') # allow parent modules to be imported
from params import get_params
import models.fp.fokker_planck_fpt as fp
import time


# use the following in IPython for qt plots: %matplotlib qt


# full fokker planck model
run_fp = True

# use as default the parameters from file params.py
# if not specified else below
params = get_params()
params['neuron_model'] = 'LIF'
params['integration_method'] = 'implicit'
params['Vlb'] = -150.0
params['N_centers_fp'] = 1000

params['taum'] = 20.0
params['DeltaT'] = 0.0
params['T_ref'] = 0.0

params['fp_v_init'] = 'delta'
params['fp_delta_peak'] = params['Vr']

# time steps for models
params['fp_dt'] = 0.05

t_grid = np.arange(0, 200, params['fp_dt'])
params['t_grid'] = t_grid

# mu input
mu_ext = 1.5 * np.ones_like(t_grid)
# sigma input
sigma_ext = 2.5 * np.ones_like(t_grid)

spike_times = np.array([0, 8]) #[50, 100, 200, 400]


# plotting section
plot_rates = True
plot_input = True



# saving results in global results dict
results = dict()
# results['input_mean'] = mu_ext
# results['input_sigma']= sigma_ext
results['model_results'] = dict()

start = time.time()

#fokker planck equation solved using the Scharfetter-Gummel-flux approximation
if run_fp:
    # ext_input = interpolate_input(ext_input, params, 'fp')
    results['model_results']['fp'] = \
        fp.sim_fp_sg_fpt(mu_ext, sigma_ext, params, fpt=True, rt=spike_times)

print('FVM took {dur}s'.format(dur=np.round(time.time() - start,2)))

# plotting section
nr_p = plot_rates + plot_input
fig = plt.figure(); pidx = 1

# plot inputs
if plot_input:
    ax_mu = fig.add_subplot(nr_p, 1, pidx)
    plt.plot(t_grid, mu_ext, color = 'k', lw=1.5)
    line_mu_final = plt.plot(t_grid, mu_ext, color = 'm', lw=1.5, label='$\mu_\mathrm{final}$')
    plt.ylabel('$\mu_{ext}$ [mV/ms]', fontsize=15)
    ax_sig = plt.twinx()
    plt.plot(t_grid, sigma_ext, color = 'g', lw=1.5)
    line_sig_final = plt.plot(t_grid, sigma_ext, color = 'b', lw=1.5, label='$\sigma_\mathrm{final}$')
    plt.ylabel('$\sigma_{ext}$ [$\sqrt{mV}$/ms]', fontsize=15)
    plt.legend([line_mu_final[0], line_sig_final[0]],
               [line_mu_final[0].get_label(), line_sig_final[0].get_label()])
    pidx +=1

# plot rates
if plot_rates:
    ax_rate = fig.add_subplot(nr_p, 1, pidx, sharex=ax_mu)
    for model in results['model_results']:
        color = params['color'][model]
        lw = params['lw'][model]
        rates = results['model_results'][model]['r']
        plt.plot(t_grid, rates, label = model, color = color, lw=lw)
        plt.ylabel('spike likelihood [kHz]')
        plt.legend()
    pidx += 1

if nr_p: plt.show()





