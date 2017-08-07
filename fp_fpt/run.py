from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(1,'..') # allow parent modules to be imported
sys.path.insert(1,'../..') # allow parent modules to be imported
sys.path.insert(1,'../../..') # allow parent modules to be imported
import adex_comparison.params as params
import models.fp.fokker_planck_fpt as fp


# use the following in IPython for qt plots: %matplotlib qt


# full fokker planck model
run_fp =       True


# use as default the parameters from file params.py
# if not specified else below
params = params.get_params()

# time steps for models
params['fp_dt'] = 0.05


# mu input
mu_ext = np.ones(10000)
# sigma input
sigma_val = 2.
sigma_ext = np.ones_like(mu_ext)*sigma_val

t_ext = np.arange(0, len(mu_ext)*params['fp_dt'], params['fp_dt'])
params['t_ext'] = t_ext

ext_input = [mu_ext, sigma_ext]




# plotting section
plot_rates = True
plot_input = True



# saving results in global results dict
results = dict()
# results['input_mean'] = mu_ext
# results['input_sigma']= sigma_ext
results['model_results'] = dict()


#fokker planck equation solved using the Scharfetter-Gummel-flux approximation
if run_fp:
    # ext_input = interpolate_input(ext_input, params, 'fp')
    results['model_results']['fp'] = \
        fp.sim_fp_sg_fpt(ext_input, params, fpt=True, rt=[50, 100, 200, 400])


# plotting section
nr_p = plot_rates + plot_input
fig = plt.figure(); pidx = 1

# plot inputs
if plot_input:
    ax_mu = fig.add_subplot(nr_p, 1, pidx)
    plt.plot(t_ext, mu_ext, color = 'k', lw=1.5)
    line_mu_final = plt.plot(t_ext, ext_input[0], color = 'm', lw=1.5, label='$\mu_\mathrm{final}$')
    plt.ylabel('$\mu_{ext}$ [mV/ms]', fontsize=15)
    ax_sig = plt.twinx()
    plt.plot(t_ext, sigma_ext, color = 'g', lw=1.5)
    line_sig_final = plt.plot(t_ext, ext_input[1], color = 'b', lw=1.5, label='$\sigma_\mathrm{final}$')
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
        plt.plot(t_ext, rates, label = model, color = color, lw=lw)
        plt.ylabel('r [Hz]')
        plt.legend()
    pidx += 1

if nr_p: plt.show()





