# imports
import numpy as np
import tables
from misc.utils import interpolate_xy, lookup_xy, get_mu_syn, get_sigma_syn, outside_grid_warning
from matplotlib.pyplot import *
# try to import numba
# or define dummy decorator
try:
    import numba
    from numba import njit
    use_numba = True
except:
    def njit(func):
        return func
    use_numba = False


@njit
def sim_ln_dos(mu_ext, sig_ext, dmu_ext_dt, t, dt, steps,
               mu_range, sig_range, omega_grid,tau_grid,
               tau_mu_f_grid, tau_sigma_f_grid, Vmean_grid,
               r_grid,a,b,C,wm0, EW, tauw, rec, K, J, delay_type,
               const_delay,taud,uni_int_order, grid_warn = True):
    
    # small optimization(s)
    b_tauW = b * tauw
    
    # initialize arrays
    r     = np.zeros_like(t)
    r_d   = np.zeros_like(t)
    mu_f  = np.zeros_like(t)
    mu_total = np.zeros_like(t)
    dmu_f = np.zeros_like(t)      #aux. var for integration
    dmu_syn_dt = np.zeros_like(t)
    sig_f     = np.zeros_like(t)
    wm  = np.zeros_like(t)
    Vm = np.zeros_like(t)
    mu_syn = np.zeros_like(t)
    sigma_syn = np.zeros_like(t)
    omega = np.zeros_like(t)
    tau = np.zeros_like(t)
    tau_mu_f = np.zeros_like(t)
    tau_sigma_f = np.zeros_like(t)
    A = np.zeros_like(t)

    # set initial value of wm
    wm[0] = wm0

    for i in xrange(steps):
        for j in range(int(uni_int_order)):



            # outside grid warning
            if grid_warn and j == 0:
                outside_grid_warning(mu_f[i+j]-wm[i+j]/C, sig_f[i+j],
                                     mu_range, sig_range, dt*i)



            # weights for looking up mean membrane voltage (Vm) 
            # and the firing rate (r)
            weights_2 = interpolate_xy(mu_f[i+j]-wm[i+j]/C, sig_f[i+j],
                                       mu_range, sig_range)
            Vm[i+j] = lookup_xy(Vmean_grid,weights_2)
            r[i+j] = lookup_xy(r_grid,weights_2)
            
            # with recurrency compute the synaptic mean and sigma
            # by generative functions 
            if rec:
                mu_syn[i+j] = get_mu_syn(K,J,mu_ext,delay_type,i+j,
                                    r,r_d,const_delay,taud,dt)
                sigma_syn[i+j] = get_sigma_syn(K,J,sig_ext,delay_type,i+j,
                                          r,r_d,const_delay,taud,dt)

                # exponentially distributed delays -> compute dmu_syn_dt analytically
                if delay_type == 2:
                    dmu_syn_dt[i+j] = dmu_ext_dt[i+j] + (r[i+j]-r_d[i+j])/taud
                # in all other rec cases compute dmu_syn by finite differences
                else:
                    dmu_syn_dt[i+j] = (mu_syn[i+j]-mu_syn[i+j-1])/dt

            # w/o recurrency (i.e. in the case of an upcoupled populatoin of neurons, K=0)
            # -> mu_syn, sigma_syn and dmu_syn_dt equal the external quantities
            else:
                mu_syn[i+j] = mu_ext[i+j]
                sigma_syn[i+j] = sig_ext[i+j]
                dmu_syn_dt[i+j] = dmu_ext_dt[i+j]

            # save total mu
            mu_total[i+j] = mu_syn[i+j]-wm[i+j]/C

            # weights for looking up omega, tau, tau_mu_f, tau_sigma_f                    
            weights_1 = interpolate_xy(mu_total[i+j],sigma_syn[i+j],
                                       mu_range, sig_range)   
                                       
            omega[i+j]  =      lookup_xy(omega_grid, weights_1)
            tau[i+j] =         lookup_xy(tau_grid, weights_1)
            tau_mu_f[i+j] =    lookup_xy(tau_mu_f_grid, weights_1)
            tau_sigma_f[i+j] = lookup_xy(tau_sigma_f_grid, weights_1)

            A[i+j] = (tau[i+j]**2*omega[i+j]**2+1)/tau[i+j]

            # j == 0: corresponds to a simple Euler method. 
            # If j reaches 1 (-> j!=0) the ELSE block is executed 
            # which updates using the HEUN scheme.
            if j == 0:
                mu_f[i+1] = mu_f[i] + dt*dmu_f[i]
                dmu_f[i+1] = dmu_f[i] + dt*( (-2./tau[i])*dmu_f[i]
                                        -(1./tau[i]**2 + omega[i]**2)*mu_f[i]
                                        +A[i]*(mu_syn[i]/tau[i]+dmu_syn_dt[i]))
                
                sig_f[i+1] = sig_f[i] + dt*(sigma_syn[i]-sig_f[i])/tau_sigma_f[i]
                wm[i+1] = wm[i] + dt*(a[i]*(Vm[i] - EW) - wm[i] + b_tauW[i] * r[i])/tauw
            # only perform Heun integration step if 'uni_int_order' == 2
            else:
                mu_f[i+1] = mu_f[i] + dt/2.*(dmu_f[i]+dmu_f[i+1])
                dmu_f[i+1] = dmu_f[i] + dt/2. * (( (-2./tau[i])*dmu_f[i]
                                        -(1./tau[i]**2 + omega[i]**2)*mu_f[i]
                                        +A[i]*(mu_syn[i]/tau[i]+dmu_syn_dt[i]))
                                        +( (-2./tau[i+1])*dmu_f[i+1]
                                        -(1./tau[i+1]**2 + omega[i+1]**2)*mu_f[i+1]
                                        +A[i+1]*(mu_syn[i+1]/tau[i+1]+dmu_syn_dt[i+1])))

                sig_f[i+1] = sig_f[i] + dt/2.*((sigma_syn[i]-sig_f[i])/tau_sigma_f[i]
                                            +(sigma_syn[i+1]-sig_f[i+1])/tau_sigma_f[i+1])
                
                wm[i+1] = wm[i] + dt/2.*((a[i]*(Vm[i] - EW) - wm[i] + b_tauW[i] * r[i])/tauw
                                      +(a[i+1]*(Vm[i+1] - EW) - wm[i+1] + b_tauW[i+1] * r[i+1])/tauw)

    # return results tuple
    return (r*1000., wm, mu_total, Vm)


# wrapper functions for the simulation
def run_ln_dos(ext_signal, params, filename,
                       rec_vars = ['wm'],
                       rec = False, FS = False):

    if FS:
        raise NotImplementedError('FS-effects not implemented for LNexp model!')

    print('==================== integrating LNdos-model ====================')

    # runtime parameters
    runtime = params['runtime']
    dt = params['uni_dt']
    steps = int(runtime/dt)
    t = np.linspace(0., runtime, steps+1)

    # external input moments
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

    # time integration method (switch between Euler & Heun method)
    uni_int_order = params['uni_int_order']

    # outside grid warning
    grid_warn = params['grid_warn']

    # membrane capacitance
    C = params['C']

    # extract quantities from hdf5-file 
    h5file = tables.open_file(filename, mode='r')
    mu_vals = np.array(h5file.root.mu_vals)
    sigma_vals = np.array(h5file.root.sigma_vals)
    f0_mu_dosc = np.array(h5file.root.f0_mu_dosc)
    omega = f0_mu_dosc*2.*np.pi
    tau = np.array(h5file.root.tau_mu_dosc)
    tau[tau < dt] = dt
    tau_mu_f = np.array(h5file.root.tau_mu_exp)
    tau_mu_f[tau_mu_f < dt] = dt
    tau_sigma_f = np.array(h5file.root.tau_sigma_exp)
    tau_sigma_f[tau_sigma_f < dt] = dt
    V_mean_ss = np.array(h5file.root.V_mean_ss)
    r_ss = np.array(h5file.root.r_ss)
    h5file.close()


    # initial value != 0 if we have adaptation
    wm0 = params['wm_init'] if have_apad else False

    # time derivative of input
    dmu_ext_dt = np.diff(mu_ext)/dt
    dmu_ext_dt = np.append(dmu_ext_dt,dmu_ext_dt[-1])

    # run simulation and save everything in the results dict
    res_sim = sim_ln_dos(mu_ext,sigma_ext,dmu_ext_dt,t,dt,steps,
                         mu_vals,sigma_vals,omega,tau,tau_mu_f,
                         tau_sigma_f,V_mean_ss,r_ss,a,b,C,wm0,
                         Ew,tauw,rec,K,J,delay_type,ndt,taud,
                         uni_int_order, grid_warn)

    #store results in dictionary which is returned
    results_dict = dict()
    results_dict['r'] = res_sim[0]
    results_dict['t'] = t
    results_dict['mu_total'] = res_sim[2]
    # take all elements except last one so that len(time)==len(wm)
    if 'wm' in rec_vars: results_dict['wm'] = res_sim[1]
    if 'Vm' in rec_vars: results_dict['Vm'] = res_sim[3]
    return results_dict

