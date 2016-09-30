import numpy as np
import tables
from misc.utils import interpolate_xy, lookup_xy, get_mu_syn, get_sigma_syn

try:
    import numba
    from numba import njit
    use_numba = True
except:
    def njit(func):
        return func
    use_numba = False

from matplotlib.pyplot import *


# current ln_dosc model version with HEUN time intergation method
###################################################################################################
#@njit
def sim_ln_bexdos(mu_ext, sig_ext, mu_range, sigma_range,
                  B_mu_grid, tau_mu_1_grid, tau_mu_2_grid,
                  f0_mu_grid, tau_mua_grid, tau_sigma_grid,
                  Vm_grid, r_grid, dmu_ext_dt,
                  C, K, J, const_delay, delay_type,
                  rec, taud,
                  a,b,tauw, EW,
                  t, dt, steps, uni_int_order):

    # todo: figure out how to deal with


    # initialize arrays
    x_mu = np.zeros_like(t)
    y_mu = np.zeros_like(t)
    z_mu = np.zeros_like(t)

    mu_f = np.zeros_like(t)
    mu_a = np.zeros_like(t)
    mu_syn = np.zeros_like(t)
    dmu_syn_dt = np.zeros_like(t)
    sigma_syn = np.zeros_like(t)
    sigma_f = np.zeros_like(t)
    wm = np.zeros_like(t)
    Vm = np.zeros_like(t)
    r = np.zeros_like(t)
    r_d = np.zeros_like(t)
    inv_tau1 = np.zeros_like(t)


    B_mu = np.zeros_like(t)
    tau_mu_1 = np.zeros_like(t)
    tau_mu_2 = np.zeros_like(t)
    f0_mu = np.zeros_like(t)
    tau_mua = np.zeros_like(t)
    tau_sigma = np.zeros_like(t)
    A_mu = np.zeros_like(t)





    for i in xrange(steps):
        for j in range(int(uni_int_order)):
            # mu_a lookup
            weights_1 = interpolate_xy(mu_a[i+j]-wm[i+j]/C, sigma_f[i+j], mu_range, sigma_range)
            B_mu[i+j] = lookup_xy(B_mu_grid, weights_1)
            tau_mu_1[i+j] = lookup_xy(tau_mu_1_grid,weights_1)
            inv_tau1[i+j] = tau_mu_1[i+j]
            tau_mu_2[i+j] = lookup_xy(tau_mu_2_grid,weights_1)
            # todo change this to omega
            f0_mu[i+j] = lookup_xy(f0_mu_grid, weights_1)
            # this is tau_mu_exp
            tau_mua[i+j] = lookup_xy(tau_mua_grid, weights_1)
            # this is tau_sigma_exp
            tau_sigma[i+j] = lookup_xy(tau_sigma_grid, weights_1)

            A_mu[i+j] = 1./tau_mu_1[i+j] - B_mu[i+j]*tau_mu_2[i+j]/\
                                           ((1.+(2.*np.pi*f0_mu[i+j]*tau_mu_2[i+j])**2)
                                            *tau_mu_1[i+j])



            # mu_f lookup
            weights_2 = interpolate_xy(mu_f[i+j]-wm[i+j]/C, sigma_f[i+j], mu_range, sigma_range)
            Vm[i+j] = lookup_xy(Vm_grid, weights_2)
            r[i+j] = lookup_xy(r_grid, weights_2)

            if rec:
                mu_syn[i+j] = get_mu_syn(K,J,mu_ext,delay_type,i+j,
                                    r,r_d,const_delay,taud,dt)
                sigma_syn[i+j] = get_sigma_syn(K,J,sig_ext,delay_type,i+j,
                                          r,r_d,const_delay,taud,dt)

                # also compute the derivatives of the synaptic input
                # in the case of exponentially distributed
                # delays; compute dmu_syn_dt analytically
                if delay_type == 2:
                    dmu_syn_dt[i+j] = dmu_ext_dt[i+j] + (r[i+j]-r_d[i+j])/taud
                # else finite differences mu_syn
                else:
                    dmu_syn_dt[i+j] = (mu_syn[i+j]-mu_syn[i+j-1])/dt




            # this case corresponds to an UNCOUPLED POPULATION of neurons, i.e. K=0.
            #todo: double check thist comutation
            else:
                mu_syn[i+j] = mu_ext[i+j]
                sigma_syn[i+j] = sig_ext[i+j]
                dmu_syn_dt[i+j] = dmu_ext_dt[i+j]


            #euler step
            if j == 0:
                x_mu[i+1] = x_mu[i] + dt*(A_mu[i]*mu_syn[i] - x_mu[i]/tau_mu_1[i])
                y_mu[i+1] = y_mu[i] + dt*z_mu[i]
                z_mu[i+1] = z_mu[i] + dt*(B_mu[i]*mu_syn[i]/tau_mu_2[i] + B_mu[i]*dmu_ext_dt[i]-2.*z_mu[i]/tau_mu_2[i]
                                          -(1./tau_mu_2[i]**2 + (2.*np.pi*f0_mu[i])**2)*y_mu[i])
                mu_f[i+1] = x_mu[i+1] + y_mu[i+1]
                mu_a[i+1] = mu_a[i] + dt*(mu_syn[i]-mu_a[i])/dt #dt #tau_mua[i]
                sigma_f[i+1] = sigma_f[i] + dt*(sigma_syn[i]-sigma_f[i])/tau_sigma[i]
                wm[i+1] = wm[i] + dt*(a[i]*(Vm[i] - EW) - wm[i] + tauw * b[i] * r[i])/tauw




            #additional heun step #TODO: adjust according to euler above
            if j == 1:
                x_mu[i+1] = x_mu[i] + dt/2.*((mu_syn[i] - x_mu[i]/tau_mu_1[i])
                                             +(mu_syn[i+1] - x_mu[i+1]/tau_mu_1[i+1]))
                y_mu[i+1] = y_mu[i] + dt/2.*(z_mu[i]+z_mu[i+1])
                z_mu[i+1] = z_mu[i] + dt/2.*((mu_syn[i]/tau_mu_2[i] + dmu_ext_dt[i]-2.*z_mu[i]/tau_mu_2[i]
                                              -(1./tau_mu_2[i]**2 + (2.*np.pi*f0_mu[i])**2)*y_mu[i])
                                            +(mu_syn[i+1]/tau_mu_2[i+1] + dmu_ext_dt[i+1]-2.*z_mu[i+1]/tau_mu_2[i+1]
                                              -(1./tau_mu_2[i+1]**2 + (2.*np.pi*f0_mu[i+1])**2)*y_mu[i+1]))

                # no heun scheme for that step
                mu_f[i+1] = A_mu[i]*x_mu[i+1] + B_mu[i]*y_mu[i+1]
                mu_a[i+1] = mu_a[i] + dt/2.*((mu_syn[i]-mu_a[i])/tau_mua[i]
                                             +(mu_syn[i+1]-mu_a[i+1])/tau_mua[i+1])
                sigma_f[i+1] = sigma_f[i] + dt/2.*((sigma_syn[i]-sigma_f[i])/tau_sigma[i]
                                                   +(sigma_syn[i+1]-sigma_f[i+1])/tau_sigma[i+1])
                wm[i+1] = wm[i] + dt/2.*((a[i]*(Vm[i] - EW) - wm[i] + tauw * b[i] * r[i])/tauw
                                         +(a[i+1]*(Vm[i+1] - EW) - wm[i+1] + tauw * b[i+1] * r[i+1])/tauw)
    return (r*1000., wm, Vm)


def run_ln_bexdos(ext_signal, params, filename,
                       rec_vars = ['wm'],
                       rec = False, FS = False):

    if FS:
        raise NotImplementedError('FS-effects not implemented for LNexp model!')

    # print('running lndosc')

    # runtime parameters
    runtime = params['runtime']
    dt = params['uni_dt']
    steps = int(runtime/dt)
    t = np.linspace(0., runtime, steps+1)

    # external inputs
    mu_ext = ext_signal[0]
    sigma_ext = ext_signal[1]

    # adaptation parameters
    a = params['a']
    b = params['b']
    # convert to array if adapt params are scalar values
    if type(a) in [int,float]:
        a = np.ones(steps+1)*a
    if type(b) in [int,float]:
        b = np.ones(steps+1)*b
    tauw = params['tauw']
    Ew = params['Ew']
    have_apad = True if (a.any() or b.any()) else False

    # coupling parameters
    K  = params['K']
    J = params['J']
    taud = params['taud']
    ndt = int(params['const_delay']/dt)
    delay_type = params['delay_type']
    # time integration method
    uni_int_order = params['uni_int_order']

    # membrane capacitance
    C = params['C']



    h5file = tables.open_file(filename, mode='r')
    mu_vals = np.array(h5file.root.mu_vals)
    sigma_vals = np.array(h5file.root.sigma_vals)
    f0_mu = np.array(h5file.root.f0_mu_bedosc)
    tau_mu_1 = np.array(h5file.root.tau1_mu_bedosc)
    tau_mu_1[tau_mu_1<dt] = dt
    tau_mu_2 = np.array(h5file.root.tau2_mu_bedosc)
    tau_mu_2[tau_mu_2<dt] = dt
    B_mu = np.array(h5file.root.B_mu_bedosc)
    tau_mu_f = np.array(h5file.root.tau_mu_exp)
    tau_mu_f[tau_mu_f < dt] = dt
    tau_sigma = np.array(h5file.root.tau_sigma_exp)
    tau_sigma[tau_sigma < dt] = dt
    Vm = np.array(h5file.root.V_mean_ss)
    r_ss = np.array(h5file.root.r_ss)
    h5file.close()


    # initial value != 0 if we have adaptation
    wm0 = params['wm_init'] if have_apad else False

    # compute derivatives
    dmu_ext_dt = np.diff(mu_ext)/dt
    dmu_ext_dt = np.append(dmu_ext_dt,dmu_ext_dt[-1])


    res_sim = sim_ln_bexdos(mu_ext, sigma_ext, mu_vals, sigma_vals,
                            B_mu, tau_mu_1,tau_mu_2,f0_mu,tau_mu_f,
                            tau_sigma,Vm, r_ss, dmu_ext_dt, C, K,
                            J, ndt, delay_type,rec, taud, a,b,
                            tauw, Ew,t, dt, steps,uni_int_order)

    # todo: also save mu_total
    #store results in dictionary which is returned
    results_dict = dict()
    results_dict['r'] = res_sim[0]
    results_dict['t'] = t
    results_dict['mu_total'] = res_sim[2]
    # take all elements except last one so that len(time)==len(wm)
    if 'wm' in rec_vars: results_dict['wm'] = res_sim[1]
    if 'Vm' in rec_vars: results_dict['Vm'] = res_sim[2]
    return results_dict