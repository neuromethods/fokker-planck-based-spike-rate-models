# imports
import tables
import numpy as np
from misc.utils import interpolate_xy, lookup_xy, get_mu_syn, get_sigma_syn, outside_grid_warning

# try to import numba
# or define dummy decorator
try:
    import numba
    from numba import njit
except:
    def njit(func):
        return func






# new model version with HEUN method
@njit
def sim_spec1(mu_ext, sigma_ext, map1_Vmean, map_r_inf, mu_range,
              sigma_range, map_lambda_1, tauw, dt, steps, C,a ,b, EW,
              s0, r0,w0, rec,K,J,delay_type,n_d,taud, uni_int_order,
              rectify, grid_warn = True):

    # small optimization(s)
    dt_tauw = dt/tauw
    b_tauw = b*tauw

    # initialize arrays
    r   = np.zeros(steps+1)
    r_d = np.zeros(steps+1)
    s   = np.zeros(steps+1)
    wm  = np.zeros(steps+1)
    Re = np.zeros(steps+1)
    Im = np.zeros(steps+1)
    Vm_inf = np.zeros(steps+1)
    r_inf = np.zeros(steps+1)
    r_diff = np.zeros(steps+1)
    lambda_1_real = np.zeros(steps+1)
    mu_total = np.zeros(steps+1)

    # write initial condition into array
    r[0] = r0
    s[0] = s0
    wm[0] = w0

    for i in range(steps):
        for j in xrange(int(uni_int_order)): 

            if rec:
                mu_syn = get_mu_syn(K, J, mu_ext,delay_type,i+j,r,r_d,n_d,taud,dt)
                sigma_syn = get_sigma_syn(K, J, sigma_ext,delay_type,i+j,r,r_d,n_d,taud,dt)

            # this case corresponds to an UNCOUPLED POPULATION of neurons, i.e. K=0.
            else:
                mu_syn = mu_ext[i+j]
                sigma_syn = sigma_ext[i+j]

            #effective mu
            mu_eff = mu_syn - wm[i+j] / C

            # grid warning
            if grid_warn and j == 0:
                outside_grid_warning(mu_eff, sigma_syn, mu_range, sigma_range, dt*i)

            # interpolate
            weights = interpolate_xy(mu_eff, sigma_syn, mu_range, sigma_range)
            # lookup
            Vm_inf[i+j] = lookup_xy(map1_Vmean, weights)
            r_inf[i+j] =  lookup_xy(map_r_inf, weights)
            lambda_1 = lookup_xy(map_lambda_1, weights)
            lambda_1_real[i+j] = lambda_1.real # save real part of lambda(t)

            # split lambda in real and imaginary part
            Re[i+j] = lambda_1.real
            Im[i+j] = lambda_1.imag
            r_diff[i+j] = r[i+j] - r_inf[i+j]
            mu_total[i+j] = mu_eff


            # j == 0: corresponds to a simple Euler method. 
            # If j reaches 1 (-> j!=0) the ELSE block is executed 
            # which updates using the HEUN scheme.
            if j == 0:
                r[i+1] = r[i] + dt * (Re[i] * r_diff[i] - Im[i] * s[i])
                s[i+1] = s[i] + dt * (Im[i] * r_diff[i] + Re[i] * s[i])

                wm[i+1] = wm[i] + dt_tauw * \
                                  (a[i] * (Vm_inf[i] - EW) - wm[i] + b_tauw[i] * r[i])

            # only perform Heun integration step if 'uni_int_order' == 2
            else:
                r[i+1] = r[i] + dt/2 * ( (Re[i] * r_diff[i] - Im[i] * s[i])
                                    + (Re[i+1]  * r_diff[i+1] - Im[i+1] * s[i+1]))
                s[i+1] = s[i] + dt/2 * ( (Im[i] * r_diff[i] + Re[i] * s[i])
                                    + (Im[i+1]  * r_diff[i+1] + Re[i+1] * s[i+1]))

                wm[i+1] = wm[i] + dt_tauw/2  * ( (a[i] * (Vm_inf[i] - EW) -
                                        wm[i] + b_tauw[i] * r[i])
                                        +(a[i+1] * (Vm_inf[i+1] - EW) -
                                        wm[i+1] + b_tauw[i+1] * r[i+1]))

        # set variables to zero if they would be integrated below 0
        if rectify and r[i+1] < 0.:
            r[i+1] = 0.
            s[i+1] = 0.

    return (r*1000, wm, lambda_1_real, mu_total, Vm_inf)





def run_spec1(ext_signal, params, filename_h5,
              rec_vars = ['wm'], rec = False, FS = False):

    if FS:
        raise NotImplementedError('FS-effects not implemented for spectral1 model!')

    print('==================== integrating spec1-model ====================')


    # runtime parameters
    dt = params['uni_dt']
    runtime = params['runtime']
    steps = int(runtime/dt)
    t = np.linspace(0., runtime, steps+1)

    #external signal
    mu_ext = ext_signal[0]
    sigma_ext = ext_signal[1]

    # adaptation params
    a = params['a']
    b = params['b']
    # convert to array if adapt params are scalar values
    if type(a) in [int,float]:
        a = np.ones(steps+1)*a
    if type(b) in [int,float]:
        b = np.ones(steps+1)*b
    tauw = params['tauw']
    Ew = params['Ew']
    have_adap = True if (a.any() or b.any()) else False

    # coupling parameters
    K = params['K']
    J = params['J']
    taud = params['taud']
    ndt = int(params['const_delay']/dt)
    delay_type = params['delay_type']
    # integration order
    uni_int_order = params['uni_int_order']
    # boolean for rectification & grid warning
    rectify = params['rectify_spec_models']#
    grid_warn = params['grid_warn']

    # membrane capacitance
    C = params['C']

    # load quantities from hdf5-file
    h5file           =  tables.open_file(filename_h5, mode = 'r')
    sig_tab        =  h5file.root.sigma.read()
    mu_tab         =  h5file.root.mu.read()
    lambda_1       =  h5file.root.lambda_1.read()
    map_r          =  h5file.root.r_inf.read()
    map_Vmean      =  h5file.root.V_mean_inf.read()
    h5file.close()

    #initial values
    s0 = 0.
    r0 = 0.
    w0 = params['wm_init'] if have_adap else 0.

    # run simulation loop
    results = sim_spec1(mu_ext, sigma_ext,map_Vmean,map_r,mu_tab,sig_tab,
                        lambda_1,tauw, dt, steps,C, a,b,Ew, s0, r0,w0,
                        rec, K, J, delay_type, ndt, taud, uni_int_order,
                        rectify, grid_warn)

    #store results in dictionary which is returned
    results_dict = dict()
    results_dict['r'] = results[0]
    results_dict['t'] = t
    results_dict['mu_total'] = results[3]
    if 'wm' in rec_vars: results_dict['wm'] = results[1]
    if 'Vm' in rec_vars: results_dict['Vm'] = results[4]
    if 'lambda_1_real' in rec_vars: results_dict['lambda_1_real'] = results[2]
    #put more
    # if 'Vm' in record_variables:
    # if 'XX' in record_variables:
    # if 'XX' in record_variables:
    return results_dict


