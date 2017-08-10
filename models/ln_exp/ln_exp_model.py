# imports
import numpy as np
import tables
from misc.utils import interpolate_xy, lookup_xy, get_mu_syn, get_sigma_syn, outside_grid_warning
# try to import numba
# or define dummy decorator
try:
    from numba import njit
except:
    def njit(func):
        return func

# longer description of the model
@njit
def sim_ln_exp(mu_ext, sigma_ext, mu_range, sigma_range,
               map_Vmean, map_r, map_tau_mu_f, map_tau_sigma_f,
               L, muf0, sigma_f0, w_m0, a, b, C, dt, tauW, Ew,
               rec, K, J, delay_type, const_delay, taud, uni_int_order, grid_warn = True):

    # small optimization(s)
    b_tauW = b * tauW
    
    # initialize arrays
    r_d = np.zeros(L+1)
    mu_f = np.zeros(L+1)
    sigma_f = np.zeros(L+1)
    mu_syn = np.zeros(L+1)
    mu_total = np.zeros(L+1)
    sigma_syn = np.zeros(L+1)
    w_m = np.zeros(L+1)
    rates = np.zeros(L+1)
    tau_mu_f = np.zeros(L+1)
    tau_sigma_f = np.zeros(L+1)
    Vm = np.zeros(L+1)

    # set initial values
    w_m[0] = w_m0
    mu_f[0] = muf0
    sigma_f[0] = sigma_f0

    for i in xrange(L):
        for j in xrange(int(uni_int_order)):
            # interpolate
            mu_f_eff = mu_f[i+j]-w_m[i+j]/C
            # save for param exploration
            # todo remove this again
            mu_total[i+j] = mu_f_eff

            # grid warning
            if grid_warn and j == 0:
                outside_grid_warning(mu_f_eff, sigma_f[i+j], mu_range, sigma_range, dt*i)
            # interpolate
            weights = interpolate_xy(mu_f_eff, sigma_f[i+j], mu_range, sigma_range)

            # lookup
            Vm[i+j] = lookup_xy(map_Vmean, weights)
            rates[i+j] = lookup_xy(map_r, weights)
            tau_mu_f[i+j] = lookup_xy(map_tau_mu_f, weights)
            tau_sigma_f[i+j] = lookup_xy(map_tau_sigma_f, weights)

            if rec:
                mu_syn[i+j] = get_mu_syn(K,J,mu_ext,delay_type,i+j,
                                         rates,r_d,const_delay,taud,dt)
                sigma_syn[i+j] = get_sigma_syn(K,J,sigma_ext,delay_type,i+j,
                                               rates,r_d,const_delay,taud,dt)

            # this case corresponds to an UNCOUPLED POPULATION of neurons, i.e. K=0.
            else:
                mu_syn[i+j] = mu_ext[i+j]
                sigma_syn[i+j] = sigma_ext[i+j]


            # j == 0: corresponds to a simple Euler method. 
            # If j reaches 1 (-> j!=0) the ELSE block is executed 
            # which updates using the HEUN scheme.
            if j == 0:
                mu_f[i+1] = mu_f[i] \
                           + dt * (mu_syn[i] - mu_f[i])/tau_mu_f[i]
                sigma_f[i+1] = sigma_f[i] \
                              + dt * (sigma_syn[i] - sigma_f[i])/tau_sigma_f[i]
                w_m[i+1] = w_m[i] + dt \
                                  * (a[i]*(Vm[i] - Ew) - w_m[i]
                                     + b_tauW[i] * (rates[i]))/tauW
            # only perform Heun integration step if 'uni_int_order' == 2
            else:
                mu_f[i+1] = mu_f[i] \
                            + dt/2. * ((mu_syn[i] - mu_f[i])/tau_mu_f[i]
                                       +(mu_syn[i+1] - mu_f[i+1])/tau_mu_f[i+1])

                sigma_f[i+1] = sigma_f[i] \
                               + dt/2. * ((sigma_syn[i] - sigma_f[i])/tau_sigma_f[i]
                                          +(sigma_syn[i+1] - sigma_f[i+1])/tau_sigma_f[i+1])
                w_m[i+1] = w_m[i] \
                           + dt/2. * (((a[i]*(Vm[i] - Ew) - w_m[i]
                                       + b_tauW[i] * (rates[i]))/tauW)
                                      +((a[i+1]*(Vm[i+1] - Ew) - w_m[i+1]
                                     + b_tauW[i+1] * (rates[i+1]))/tauW))
    
    #order of return tuple should always be rates, w_m, Vm, etc...
    return (rates*1000., w_m, Vm, tau_mu_f, tau_sigma_f, mu_f, sigma_f, mu_total)

# wrapper function without numba
# the input filename will become obsolet after the issue with the mu/sigma tables has been cleared
def run_ln_exp(ext_signal, params,filename,
               rec_vars = ['wm'],
               rec = False, FS = False):


    if FS:
        raise NotImplementedError('FS-effects not implemented for LNexp model!')

    print('==================== integrating LNexp-model ====================')

    # runtime parameters
    runtime = params['runtime']
    dt = params['uni_dt']
    steps = int(runtime/dt)
    t = np.linspace(0., runtime, steps+1)

    # external inputs
    mu_ext = ext_signal[0]      #[mV/ms]
    sigma_ext = ext_signal[1]   #[mV/sqrt(ms)]

    # adaptation parameters
    a = params['a']
    b = params['b']
    # convert to array if adapt params are scalar values
    if type(a) in [int,float]:
        a = np.ones(steps+1)*a
    if type(b) in [int,float]:
        b = np.ones(steps+1)*b
    tauW = params['tauw']
    Ew = params['Ew']
    have_adap = True if (a.any() or b.any()) else False

    # coupling parameters
    K = params['K']
    J = params['J']
    taud = params['taud']
    ndt = int(params['const_delay']/dt)
    delay_type = params['delay_type']
    # time integration method
    uni_int_order = params['uni_int_order']
    # outside grid warning
    grid_warn = params['grid_warn']

    # membrane capacitance
    C = params['C']

    
    
    # extract numerical QUANTITIES from the hdf5 - file
    h5file = tables.open_file(filename, mode='r')
    mu_vals = np.array(h5file.root.mu_vals)
    sigma_vals = np.array(h5file.root.sigma_vals)
    V_mean_ss = np.array(h5file.root.V_mean_ss)
    r_ss = np.array(h5file.root.r_ss)
    tau_mu_exp = np.array(h5file.root.tau_mu_exp)
    tau_sigma_exp = np.array(h5file.root.tau_sigma_exp)
    tau_sigma_exp[tau_sigma_exp < dt] = dt
    tau_mu_exp[tau_mu_exp < dt] = dt
    h5file.close()

    
    # set initial values for mu/sigma
    mu_f0 = 0# 0.5
    sigma_f0 = 0#  1.6

    # initial value != 0 if we have adaptation
    wm0 = params['wm_init'] if have_adap else 0.

    # run the model and save results in the results dictionary 
    results = sim_ln_exp(mu_ext, sigma_ext, mu_vals, sigma_vals,
                         V_mean_ss, r_ss, tau_mu_exp,tau_sigma_exp                             ,
                         steps, mu_f0, sigma_f0,  wm0, a, b, C, dt,
                         tauW, Ew, rec,K,J, delay_type, ndt, taud,
                         uni_int_order, grid_warn)
    
    #depending on the variable record_variables certain data arrays get saved in the results_dict
    results_dict = dict()

    # return arrays without the last element
    results_dict['r'] = results[0]
    results_dict['t'] = t
    results_dict['mu_total'] = results[7]
    results_dict['sigma_f'] = results[6]
    # take all elements except last one so that len(time)==len(wm)
    if 'wm' in rec_vars: results_dict['wm'] = results[1]
    if 'Vm' in rec_vars: results_dict['Vm'] = results[2]
    return results_dict




