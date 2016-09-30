# imports
import numpy as np
import tables
from misc.utils import interpolate_xy, lookup_xy, get_mu_syn, get_sigma_syn
# try to import numba
# or define dummy decorator
try:
    from numba import njit
except:
    def njit(func):
        return func


#FUNCTION TO SIMULATE aLN exp MODEL ACCELERATED (NUMBA)
@njit
def sim_ln_exp(mu_ext, sigma_ext, mu_range, sigma_range,
               map_Vmean, map_r, map_tau_mu_f, map_tau_sigma_f,
               L, muf0, sigma_f0, w_m0, a, b, C, dt, tauW, Ew,
               rec, K, J, delay_type, const_delay, taud, uni_int_order):

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


            # euler step
            if j == 0:
                mu_f[i+1] = mu_f[i] \
                           + dt * (mu_syn[i] - mu_f[i])/tau_mu_f[i]
                sigma_f[i+1] = sigma_f[i] \
                              + dt * (sigma_syn[i] - sigma_f[i])/tau_sigma_f[i]
                w_m[i+1] = w_m[i] + dt \
                                  * (a[i]*(Vm[i] - Ew) - w_m[i]
                                     + b_tauW[i] * (rates[i]))/tauW
            if j == 1:
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

    # print('running lnexp')

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

    # membrane capacitance
    C = params['C']

    ##################################################################################
    # Vl = params['ALN_v_lower_bound']/mV # lower bound for voltage
    # Vs = -40.0 #mV
    # dVs = 0.5 #mV
    # EL = params['EL']/mV
    # Vr = params['Vr']/mV
    # VT = params['VT']/mV
    # DeltaT = params['deltaT']
    # N = params['N_total']
    # mu_f0, sigma_f0 = get_initial_values(N, Vs, Vl, dVs, taum, EL, Vr, VT, DeltaT, C)
    ##################################################################################

    # remove this before publishing the code

    # lookup tables for quantities
    # filename_matlab = 'precalc05_int.mat'
    # data = loadmat(filename_matlab)
    # mu_range_mat = data['presimdata'][0, 0]['Irange'][0]/(params['taum'])
    # sigma_range_mat = data['presimdata'][0, 0]['sigmarange'][0]
    # map_r = data['presimdata'][0, 0]['r_raw']
    # map_Vmean = data['presimdata'][0, 0]['Vmean_raw']
    # map_tau_mu_f    = data['presimdata'][0, 0]['mu_exp_tau']
    # map_tau_mu_f[np.where(map_tau_mu_f < dt)] = dt # dt = 0.05 ms
    # map_tau_sigma_f = data['presimdata'][0, 0]['sigma_exp_tau_deltacorrection']
    # map_tau_sigma_f[np.where(map_tau_sigma_f < dt)] = dt # dt = 0.05 ms
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



    # initial values for mu/sigma
    mu_f0 = 0# 0.5
    sigma_f0 = 0#  1.6

    #initial value != 0 if we have adaptation
    wm0 = params['wm_init'] if have_adap else 0.

    results = sim_ln_exp(mu_ext, sigma_ext, mu_vals, sigma_vals,
                         V_mean_ss, r_ss, tau_mu_exp,tau_sigma_exp                             ,
                         steps, mu_f0, sigma_f0,  wm0, a, b, C, dt, tauW,
                         Ew, rec,K,J, delay_type, ndt, taud, uni_int_order)
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




#################################################################################
# old version of the ln_dosc funciton wihtout HEUN integration
@njit
def sim_ln_exp_without_HEUN_OLD(mu_ext, sigma_ext, mu_range, sigma_range,
               map_Vmean, map_r, map_tau_mu_f, map_tau_sigma_f,
               L, muf0, sigma_f0, w_m0, a, b, C, dt, tauW, Ew,
               rec, K, J, delay_type, const_delay, taud):

    b_tauW = b * tauW

    # initialize arrays
    r_d = np.zeros(L+1)
    mu_f = np.zeros(L+1)
    sigma_f = np.zeros(L+1)
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

        # interpolate
        mu_f_eff = mu_f[i]-w_m[i]/C
        weights = interpolate_xy(mu_f_eff, sigma_f[i], mu_range, sigma_range)

        # lookup
        Vm[i] = lookup_xy(map_Vmean, weights)
        rates[i] = lookup_xy(map_r, weights)
        tau_mu_f[i] = lookup_xy(map_tau_mu_f, weights)
        tau_sigma_f[i] = lookup_xy(map_tau_sigma_f, weights)

        if rec:
            mu_syn = get_mu_syn(K,J,mu_ext,delay_type,i,rates,r_d,const_delay,taud,dt)
            sigma_syn = get_sigma_syn(K,J,sigma_ext,delay_type,i,rates,r_d,const_delay,taud,dt)
        else:
            mu_syn = mu_ext[i]
            sigma_syn = sigma_ext[i]

        # rhs of equations
        mu_f_rhs = (mu_syn - mu_f[i])/tau_mu_f[i]
        sigma_f_rhs = (sigma_syn - sigma_f[i])/tau_sigma_f[i]
        w_m_rhs = (a[i]*(Vm[i] - Ew) - w_m[i] + b_tauW[i] * (rates[i]))/tauW

        # euler step
        mu_f[i+1]= mu_f[i] + dt * mu_f_rhs
        sigma_f[i+1]= sigma_f[i] + dt * sigma_f_rhs
        w_m[i+1]=w_m[i]+ dt * w_m_rhs
    #order of return tuple should always be rates, w_m, Vm, etc...
    return (rates*1000., w_m, Vm, tau_mu_f, tau_sigma_f, mu_f, sigma_f)

# wrapper function without numba
# the input filename will become obsolet after the issue with the mu/sigma tables has been cleared
def run_ln_exp_without_HEUN_OLD(ext_signal, params,filename,
               rec_vars = ['wm'],
               rec = False, FS = False):


    if FS:
        raise NotImplementedError('FS-effects not implemented for LNexp model!')

    # print('running lnexp')

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

    # membrane capacitance
    C = params['C']

    ##################################################################################
    # Vl = params['ALN_v_lower_bound']/mV # lower bound for voltage
    # Vs = -40.0 #mV
    # dVs = 0.5 #mV
    # EL = params['EL']/mV
    # Vr = params['Vr']/mV
    # VT = params['VT']/mV
    # DeltaT = params['deltaT']
    # N = params['N_total']
    # mu_f0, sigma_f0 = get_initial_values(N, Vs, Vl, dVs, taum, EL, Vr, VT, DeltaT, C)
    ##################################################################################


    h5file = tables.open_file(filename, mode='r')
    mu_vals = np.array(h5file.root.mu_vals)
    sigma_vals = np.array(h5file.root.sigma_vals)
    V_mean_ss = np.array(h5file.root.V_mean_ss)
    r_ss = np.array(h5file.root.r_ss)
    tau_mu_exp = np.array(h5file.root.tau_mu_exp)
    tau_sigma_exp = np.array(h5file.root.tau_sigma_exp)
    tau_sigma_exp[tau_sigma_exp<dt] = dt
    tau_mu_exp[tau_mu_exp<dt] = dt
    h5file.close()
    # lookup tables for quantities
    # data = loadmat(filename_mat)
    # mu_range_mat = data['presimdata'][0, 0]['Irange'][0]/(params['taum'])
    # sigma_range_mat = data['presimdata'][0, 0]['sigmarange'][0]
    # map_r = data['presimdata'][0, 0]['r_raw']
    # map_Vmean = data['presimdata'][0, 0]['Vmean_raw']
    # map_tau_mu_f    = data['presimdata'][0, 0]['mu_exp_tau']
    # map_tau_mu_f[np.where(map_tau_mu_f < dt)] = dt # dt = 0.05 ms
    # map_tau_sigma_f = data['presimdata'][0, 0]['sigma_exp_tau_deltacorrection']
    # map_tau_sigma_f[np.where(map_tau_sigma_f < dt)] = dt # dt = 0.05 ms

    # initial values for mu/sigma
    mu_f0 = 0# 0.5
    sigma_f0 = 0#  1.6

    #initial value != 0 if we have adaptation
    wm0 = params['wm_init'] if have_adap else 0.

    results = sim_ln_exp_without_HEUN_OLD(mu_ext, sigma_ext, mu_vals, sigma_vals,
                         V_mean_ss, r_ss, tau_mu_exp, tau_sigma_exp,
                         steps, mu_f0, sigma_f0,  wm0, a, b, C, dt, tauW,
                         Ew, rec,K,J, delay_type, ndt, taud)
    #depending on the variable record_variables certain data arrays get saved in the results_dict
    results_dict = dict()

    # return arrays without the last element
    results_dict['r'] = results[0]
    results_dict['t'] = t
    # take all elements except last one so that len(time)==len(wm)
    if 'wm' in rec_vars: results_dict['wm'] = results[1]
    if 'Vm' in rec_vars: results_dict['Vm'] = results[2]
    return results_dict





