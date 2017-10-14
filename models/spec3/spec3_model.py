# spectral 3 model using in total three eigenvalues
# -> two diffusive and one regular mode

from __future__ import print_function
import sys
from scipy.misc import derivative
import matplotlib.pyplot as plt
import time
from misc.utils import interpolate_xy, lookup_xy, x_filter
import tables
import numpy as np
import math
import scipy.interpolate

# try to import numba
# or define dummy decorator
try:
    from numba import njit
except:
    def njit(func):
        return func




def sim_spec3(t, dt, steps, uni_int_order, mu_range, sig_range,
              D_map, M_map, S_map, P_map, T_map, Fmu_map,
              Fsigma_map, Gmu_map, Gsigma_map, Vm_inf_map,
              dmu, ddmu, dddmu , dsigma2, ddsigma2, dddsigma2,
              Ew, tauw, mu_ext, sigma_ext):

    # arrays initialization
    mu_tot = np.zeros_like(t)
    sigma_tot = np.zeros_like(t)
    r = np.zeros_like(t)
    dr = np.zeros_like(t)
    ddr = np.zeros_like(t)
    wm = np.zeros_like(t)
    a = np.zeros_like(t)
    b = np.zeros_like(t)



    # optimizations
    dt_tauw = dt/tauw

    for i in xrange(steps):


        # feedforward case without any recurrent couplings
        mu_tot[i] = mu_ext[i]
        sigma_tot[i] = sigma_ext[i]

        #interpolate the quantities with total mu and sigma
        weights = interpolate_xy(mu_tot[i], sigma_tot[i], mu_range, sig_range)
        # lookup quantities(i) from the passed quantity maps
        D = lookup_xy(D_map, weights)
        M = lookup_xy(M_map, weights)
        S = lookup_xy(S_map, weights) # keep S-factor in mind
        P = lookup_xy(P_map, weights)
        T = lookup_xy(T_map, weights)
        Fmu = lookup_xy(Fmu_map, weights)
        Fsigma = lookup_xy(Fsigma_map, weights)
        Gmu = lookup_xy(Gmu_map, weights)
        Gsigma = lookup_xy(Gsigma_map, weights)
        Vm_inf = lookup_xy(Vm_inf_map, weights)

        # scale all ("sigma")-quantities with 1/(2*sig_tot)




        # rewrite third order equation in #1, #2, #3
        # EULER LOOP
        #1
        ddr[i+1] = ddr[i] \
                   + dt*(dddmu[i]*D*M + dddsigma2[i]*D*S - ddmu[i]*(P*M-D*Fmu)
                         - ddsigma2[i]*(P*S-D*Fsigma) - dmu[i]*(P*Fmu-T*M-D*Gmu)
                         - dsigma2[i]*(P*Fmu-T*M-D*Gsigma) - T*dr[i] + P*ddr[i])
        #2
        dr[i+1] = dr[i] + dt*(ddr[i])
        #3
        r[i+1] = r[i] + dt*dr[i]

        # adaptation wm
        wm[i+1] = wm[i] + dt_tauw*(a[i]*(Vm_inf-Ew)- wm[i] + b[i] * tauw * r[i])


    return (r*1000, wm)






def run_spec3(ext_signal, params, filename_h5,
              rec_vars=['wm'], rec=False,
              FS=False, filter_input=False, filetype='old'):
    if FS:
        raise NotImplementedError('FS-effects not implemented for spectral 2m model!')

    print('running spec3')

    # runtime parameters
    dt = params['uni_dt']
    runtime = params['runtime']
    steps = int(runtime / dt)
    t = np.linspace(0., runtime, steps + 1)

    # external signal
    mu_ext = ext_signal[0]
    sig_ext = ext_signal[1]

    # adaptation parameters
    a = params['a']
    b = params['b']
    # convert to array if adapt params are scalar values
    if type(a) in [int, float]:
        a = np.ones(steps + 1) * a
    if type(b) in [int, float]:
        b = np.ones(steps + 1) * b
    tauw = params['tauw']
    Ew = params['Ew']
    have_adapt = True if (a.any() or b.any()) else False

    # coupling parameters
    K = params['K']
    J = params['J']
    taud = params['taud']
    delay_type = int(params['delay_type'])
    const_delay = params['const_delay']
    # time integration method
    uni_int_order = params['uni_int_order']
    # boolean for rectification
    rectify = params['rectify_spec_models']

    # membrane capacitance
    C = params['C']

    # todo: file toggle
    # load quantities from hdf5-file
    h5file = tables.open_file(filename_h5, mode='r')
    mu_range = h5file.root.mu.read()
    sig_range = h5file.root.sigma.read()


    lambda_reg_diff = h5file.root.lambda_reg_diff.read()
    lambda_1 = lambda_reg_diff[0, :, :]
    lambda_2 = lambda_reg_diff[1, :, :]
    lambda_3 = lambda_reg_diff[2, :, :]
    f = h5file.root.f.read()
    f1 = f[0, :, :]
    f2 = f[1, :, :]
    f3 = f[2, :, :]
    c_mu = h5file.root.c_mu.read()
    c_mu_1 = c_mu[0, :, :]
    c_mu_2 = c_mu[1, :, :]
    c_mu_3 = c_mu[2, :, :]
    c_sigma = h5file.root.c_sigma.read()
    c_sigma_1 = c_sigma[0, :, :]
    c_sigma_2 = c_sigma[1, :, :]
    c_sigma_3 = c_sigma[2, :, :]



    dr_inf_dmu = h5file.root.dr_inf_dmu.read()
    dr_inf_dsigma = h5file.root.dr_inf_dsigma.read()
    dVm_dmu = h5file.root.dV_mean_inf_dmu.read()
    dVm_dsig = h5file.root.dV_mean_inf_dsigma.read()
    V_mean_inf = h5file.root.V_mean_inf.read()
    r_inf = h5file.root.r_inf.read()
    h5file.close()



    # filter the input (typically done before calling here, thus disabled)
    # NOTE: this should result in a smooth input which
    # should be differentiable twice; maybe this function
    # will be transfered out of this function
    filter_input = False  # from now on assume calling with smooth input
    if filter_input:
        # filter input moments
        print('filtering inputs...')
        start = time.time()
        mu_ext = x_filter(mu_ext, params)
        sig_ext = x_filter(sig_ext, params)
        print('filtering inputs done in {} s.'
              .format(time.time() - start))

    # compute 1st/2nd derivatives of input variance (not std!)
    sig_ext2 = sig_ext ** 2
    dsig_ext2_dt = np.diff(sig_ext2) / dt
    dsig_ext2_dt = np.insert(dsig_ext2_dt, 0, dsig_ext2_dt[-1])  # 1st order backwards differences
    d2sig_ext2_dt2 = np.diff(dsig_ext2_dt) / dt  #
    d2sig_ext2_dt2 = np.append(d2sig_ext2_dt2, d2sig_ext2_dt2[-1])  # 2nd order central diffs
    d3sigma_ext2_dt3 = np.diff(d2sig_ext2_dt2) / dt
    d3sigma_ext2_dt3 = np.append(d3sigma_ext2_dt3, d3sigma_ext2_dt3[-1])

    # compute 1st/2nd derivatives of input mean
    dmu_ext_dt = np.diff(mu_ext) / dt
    dmu_ext_dt = np.insert(dmu_ext_dt, 0, dmu_ext_dt[-1])  # 1st order backwards differences
    d2mu_ext_dt2 = np.diff(dmu_ext_dt) / dt
    d2mu_ext_dt2 = np.append(d2mu_ext_dt2, d2mu_ext_dt2[-1])  # 2nd order central diffs
    d3mu_ext_dt3 = np.diff(d2mu_ext_dt2) / dt
    d3mu_ext_dt3 = np.append(d3mu_ext_dt3, d3mu_ext_dt3[-1])

    # lumped quantities #############
    T = (lambda_1 ** -1 + lambda_2 ** -1 + lambda_3 ** -1).real
    D = ((lambda_1 * lambda_2 * lambda_3) ** -1).real
    P = ((lambda_1 * lambda_2) ** -1 + (lambda_1 * lambda_3) ** -1 + (lambda_2 * lambda_3) ** -1).real
    F_mu = (lambda_1 * f1 * c_mu_1 + lambda_2 * f2 * c_mu_2 + lambda_3 * f3 * c_mu_3).real
    F_sig = (lambda_1 * f1 * c_sigma_1 + lambda_2 * f2 * c_sigma_2 + lambda_3 * f3 * c_sigma_3).real
    S = (dr_inf_dsigma + (f1 * c_sigma_1 + f2 * c_sigma_2 + f3 * c_sigma_3)).real
    M = (dr_inf_dmu + (f1 * c_mu_1 + f2 * c_mu_2 + f3 * c_mu_3)).real
    G_mu = (f1 * lambda_1**2 * c_mu_1 + f2 * lambda_2**2 * c_mu_2 + f3 * lambda_3**2 * c_mu_3).real
    G_sigma = (f1 * lambda_1**2 * c_sigma_1 + f2 * lambda_2**2 * c_sigma_2 + f3 * lambda_3**2 * c_sigma_3).real



    # initial values
    wm0 = params['wm_init'] if have_adapt else 0.


    results = sim_spec3(t, dt, steps, 1, mu_range, sig_range, D, M, S, P,
                        T, F_mu, F_sig, G_mu, G_sigma, V_mean_inf, dmu_ext_dt,
                        d2mu_ext_dt2, d3mu_ext_dt3, dsig_ext2_dt, d2sig_ext2_dt2,
                        d3sigma_ext2_dt3,Ew, tauw, mu_ext, sig_ext)

    results_dict = dict()
    results_dict['t'] = t
    results_dict['r'] = results[0]
    if 'wm' in rec_vars: results_dict['wm'] = results[1]

    # return results
    return results_dict
