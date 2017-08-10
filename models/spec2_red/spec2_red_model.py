# imports
import tables
import numpy as np
from misc.utils import interpolate_xy, lookup_xy, get_mu_syn, get_sigma_syn, outside_grid_warning
# try to import numba
# or define dummy decorator
try:
    from numba import njit
except:
    def njit(func):
        return func



# model version with HEUN integration method
@njit
def sim_spec2_red(mu_ext, sigma_ext,steps,dt,EW, a,b,C, tauw,
                      map_Vmean, map_rss, mu_range2, sigma_range2,
                      map_eig1, map_eig_mattia, rec,K,J,delay_type,n_d,
                      taud, r0, s0, w0, uni_int_order, grid_warn = True):

    # optimazations
    dt_tauw = dt/tauw
    b_tauw = b*tauw

    # initialize arrays
    r = np.zeros(steps+1)
    dr = np.zeros(steps+1)
    r_d = np.zeros(steps+1)
    s = np.zeros(steps+1)
    wm = np.zeros(steps+1)
    # more arrays for HEUN method
    Vm_inf = np.zeros(steps+1)
    r_inf = np.zeros(steps+1)
    lambda1_plus_lambda2 = np.zeros(steps+1)
    lambda1_times_lambda2 = np.zeros(steps+1)



    # write initial condition into array
    r[0] = r0
    s[0] = s0
    wm[0] = w0

    for i in xrange(steps):
        for j in xrange(int(uni_int_order)): # i --> i + j = i + 1 (for second round)

            if rec:
                mu_syn = get_mu_syn(K,J,mu_ext,delay_type,i+j,r,r_d,n_d,taud,dt)
                sigma_syn = get_sigma_syn(K,J,sigma_ext,delay_type,i+j,r,r_d,n_d,taud,dt)
            else:
                mu_syn = mu_ext[i+j]
                sigma_syn = sigma_ext[i+j]

            # compute effective mu
            mu_tot = mu_syn  - wm[i+j] / C

            # grid warning
            if grid_warn and j == 0:
                outside_grid_warning(mu_tot, sigma_syn, mu_range2, sigma_range2, dt*i)

            # interpolate
            # weights_quant = interpolate_xy(mu_tot, sigma_syn, mu_range1, sigma_range1)
            weights_eig = interpolate_xy(mu_tot, sigma_syn, mu_range2, sigma_range2)

            # lookup
            Vm_inf[i+j]  = lookup_xy(map_Vmean, weights_eig)
            r_inf[i+j]  = lookup_xy(map_rss, weights_eig)
            lambda1  = lookup_xy(map_eig1, weights_eig)
            lambda2  = lookup_xy(map_eig_mattia, weights_eig)
            # results of lambda1+lambda2 and lambda1*lambda2
            # are always real --> eliminate imaginary parts
            lambda1_plus_lambda2[i+j] = (lambda1+lambda2).real
            lambda1_times_lambda2[i+j] = (lambda1*lambda2).real

            # TODO: add rectify of r via toggle (def: on)

            if j == 0:
                dr[i+1] = dr[i] + dt * (lambda1_times_lambda2[i] * (r_inf[i]-r[i])
                                        + lambda1_plus_lambda2[i] * dr[i])

                r[i+1]  = r[i]  + dt * dr[i]

                wm[i+1] = wm[i] + dt_tauw * (a[i] * (Vm_inf[i] - EW)
                                             - wm[i] + b_tauw[i] * r[i])

            # only perform Heun integration step if 'uni_int_order' == 2
            # to compute the updates state variables with the trapezian rule
            elif j == 1:
                dr[i+1] = dr[i] + dt/2. * ( (lambda1_times_lambda2[i] * (r_inf[i]-r[i])
                                               + lambda1_plus_lambda2[i] * dr[i])
                                        + (lambda1_times_lambda2[i+1] * (r_inf[i+1]-r[i+1])
                                            + lambda1_plus_lambda2[i+1] * dr[i+1]))

                r[i+1]  = r[i]  + dt/2. * (dr[i]+dr[i+1])

                wm[i+1] = wm[i] + dt_tauw/2.  * ( (a[i] * (Vm_inf[i] - EW)
                                                   - wm[i] + b_tauw[i] * r[i])
                                                  +(a[i+1] * (Vm_inf[i+1] - EW)
                                                    - wm[i+1] + b_tauw[i+1] * r[i+1]))
    # re
    return (r*1000., wm)


def run_spec2_red(ext_signal, params, filename_h5,
                      rec_vars = ['wm'], rec = False, FS = False):

    if FS:
        raise NotImplementedError('FS-effects not implemented for LNexp model!')

    print('=============== integrating (reduced) spec2-model ===============')

    # runtime parameters
    dt = params['uni_dt'] # time resolution for spectral mattia model
    runtime = params['runtime']
    steps = int(runtime/dt)
    t = np.linspace(0, runtime, steps+1)

    #external signal
    mu_ext = ext_signal[0]
    sigma_ext = ext_signal[1]

    # adaptation parameters
    a    = params['a']
    b    = params['b']
    # convert to array if adapt params are scalar values
    if type(a) in [int,float]:
        a = np.ones(steps+1)*a
    if type(b) in [int,float]:
        b = np.ones(steps+1)*b
    tauw = params['tauw']
    Ew   = params['Ew']
    have_adap = True if (a.any() or b.any()) else False

    # coupling parameters
    K = params['K']
    J = params['J']
    n_d = int(params['const_delay']//dt)
    taud = params['taud']
    delay_type = params['delay_type']
    # integration method
    uni_int_order = params['uni_int_order']
    grid_warn = params['grid_warn']

    # membrane capacitance
    C = params['C']

    # loading data from hdf5-file
    file            = tables.open_file(filename_h5, mode = 'r')
    mu_tab     = file.root.mu.read()
    sig_tab  = file.root.sigma.read()
    lambda_1        = file.root.lambda_1.read()
    lambda_2_mattia = file.root.lambda_2.read()
    map_r           = file.root.r_inf.read()
    map_Vmean       = file.root.V_mean_inf.read()
    file.close()

    #initial values
    r0 = 0.
    s0 = 0.
    w0 = params['wm_init'] if have_adap else 0.

    results = sim_spec2_red(mu_ext,sigma_ext,steps,dt,Ew,a,b,C,tauw,map_Vmean,
                                map_r,mu_tab,sig_tab,lambda_1,
                                lambda_2_mattia,rec,K,J,delay_type,n_d,
                                taud, r0, s0, w0, uni_int_order, grid_warn)
    #store results in dictionary which is returned
    results_dict = dict()
    results_dict['r'] = results[0]
    results_dict['t'] = t
    if 'wm' in rec_vars: results_dict['wm'] = results[1]
    #put more
    # if 'Vm' in record_variables:
    # if 'XX' in record_variables:
    # if 'XX' in record_variables:
    return results_dict