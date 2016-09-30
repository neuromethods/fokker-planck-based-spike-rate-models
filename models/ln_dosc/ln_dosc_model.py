# imports
import numpy as np
import tables
from misc.utils import interpolate_xy, lookup_xy, get_mu_syn, get_sigma_syn
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


# TODO: correct derivative of external mean input

# hallo josef :)
# current ln_dosc model version with HEUN time intergation method
#################################################################
@njit
def sim_ln_dosc(mu_ext, sig_ext, dmu_ext_dt, t, dt, steps,
                       mu_range, sig_range, omega_grid,
                       tau_grid, tau_mu_f_grid,
                       tau_sigma_f_grid, Vmean_grid,
                       r_grid,a,b,C,wm0, EW, tauw, rec, K, J,
                       delay_type, const_delay,taud,uni_int_order):

    # initialize arrays
    r     = np.zeros_like(t)
    r_d   = np.zeros_like(t)
    mu_f  = np.zeros_like(t)
#    mu_a  = np.zeros_like(t)  # mu_a only used for "old" version, see JL dissertation
    mu_total = np.zeros_like(t)
    dmu_f = np.zeros_like(t)      #aux. var for integration
    dmu_syn_dt = np.zeros_like(t)
    sig_f     = np.zeros_like(t)
    wm  = np.zeros_like(t)
    # new arrays for HEUN method
    Vm = np.zeros_like(t)
    mu_syn = np.zeros_like(t)
    sigma_syn = np.zeros_like(t)
    omega = np.zeros_like(t)
    tau = np.zeros_like(t)
    tau_mu_f = np.zeros_like(t)
    tau_sigma_f = np.zeros_like(t)
    A = np.zeros_like(t)


    # set initial values
    wm[0] = wm0
    # mu_f[0]  = mu_f0
    # sig_f[0] = sig_f0

    for i in xrange(steps):
        for j in range(int(uni_int_order)):



            # mu_ext filtered by damped oscillatory function for looking up Vm and r
            weights_2 = interpolate_xy(mu_f[i+j]-wm[i+j]/C, sig_f[i+j],
                                       mu_range, sig_range)
            Vm[i+j] = lookup_xy(Vmean_grid,weights_2)
            r[i+j] = lookup_xy(r_grid,weights_2)

            if rec:
                mu_syn[i+j] = get_mu_syn(K,J,mu_ext,delay_type,i+j,
                                    r,r_d,const_delay,taud,dt)
                sigma_syn[i+j] = get_sigma_syn(K,J,sig_ext,delay_type,i+j,
                                          r,r_d,const_delay,taud,dt)

                # in the case of exponentially distributed
                # delays; compute dmu_syn_dt analytically
                if delay_type == 2:
                    dmu_syn_dt[i+j] = dmu_ext_dt[i+j] + (r[i+j]-r_d[i+j])/taud
                # else finite differences mu_syn
                else:
                    dmu_syn_dt[i+j] = (mu_syn[i+j]-mu_syn[i+j-1])/dt

            # this case corresponds to an UNCOUPLED POPULATION of neurons, i.e. K=0.
            else:
                mu_syn[i+j] = mu_ext[i+j]
                sigma_syn[i+j] = sig_ext[i+j]
                dmu_syn_dt[i+j] = dmu_ext_dt[i+j]


            # save mu_total for param exploration
            # todo: remove this again
            mu_total[i+j] = mu_syn[i+j]-wm[i+j]/C


#            # for "old" version, see JL dissertation:
#            weights_1 = interpolate_xy(mu_a[i+j]-wm[i+j]/C,sig_f[i+j],
#                                       mu_range, sig_range)
                                       
            weights_1 = interpolate_xy(mu_total[i+j],sigma_syn[i+j],
                                       mu_range, sig_range)   
                                       
            omega[i+j]  =      lookup_xy(omega_grid, weights_1)
            tau[i+j] =         lookup_xy(tau_grid, weights_1)
            tau_mu_f[i+j] =    lookup_xy(tau_mu_f_grid, weights_1)
            tau_sigma_f[i+j] = lookup_xy(tau_sigma_f_grid, weights_1)
            # 2nd interpolation and lookup with



            # euler step
            A[i+j] = (tau[i+j]**2*omega[i+j]**2+1)/tau[i+j]

            if j == 0:
                mu_f[i+1] = mu_f[i] + dt*dmu_f[i]
                dmu_f[i+1] = dmu_f[i] + dt*( (-2./tau[i])*dmu_f[i]
                                        -(1./tau[i]**2 + omega[i]**2)*mu_f[i]
                                        +A[i]*(mu_syn[i]/tau[i]+dmu_syn_dt[i]))

#                # mu_a only used for "old" version, see JL dissertation
#                mu_a[i+1] = mu_a[i] + dt*(mu_syn[i]-mu_a[i])/tau_mu_f[i]
                
                sig_f[i+1] = sig_f[i] + dt*(sigma_syn[i]-sig_f[i])/tau_sigma_f[i]
                wm[i+1] = wm[i] + dt*(a[i]*(Vm[i] - EW) - wm[i] + tauw * b[i] * r[i])/tauw
            if j == 1:
                mu_f[i+1] = mu_f[i] + dt/2.*(dmu_f[i]+dmu_f[i+1])
                dmu_f[i+1] = dmu_f[i] + dt/2. * (( (-2./tau[i])*dmu_f[i]
                                        -(1./tau[i]**2 + omega[i]**2)*mu_f[i]
                                        +A[i]*(mu_syn[i]/tau[i]+dmu_syn_dt[i]))
                                        +( (-2./tau[i+1])*dmu_f[i+1]
                                        -(1./tau[i+1]**2 + omega[i+1]**2)*mu_f[i+1]
                                        +A[i+1]*(mu_syn[i+1]/tau[i+1]+dmu_syn_dt[i+1])))

#                # mu_a only used for "old" version, see JL dissertation
#                mu_a[i+1] = mu_a[i] + dt/2.*((mu_syn[i]-mu_a[i])/tau_mu_f[i]
#                                                       +(mu_syn[i+1]-mu_a[i+1])/tau_mu_f[i+1])

                sig_f[i+1] = sig_f[i] + dt/2.*((sigma_syn[i]-sig_f[i])/tau_sigma_f[i]
                                            +(sigma_syn[i+1]-sig_f[i+1])/tau_sigma_f[i+1])
                wm[i+1] = wm[i] + dt/2.*((a[i]*(Vm[i] - EW) - wm[i] + tauw * b[i] * r[i])/tauw
                                      +(a[i+1]*(Vm[i+1] - EW) - wm[i+1] + tauw * b[i+1] * r[i+1])/tauw)
    return (r*1000., wm, mu_total, Vm)

# new version of the ln dosc model. In contrast the the funtion run_ln_dosc
# this loop filters the external mu input by intgrating a second order differential
# equation (real) and not a complex first order ODE. Yielded improvements
# with respect to changes in the sigma input: no strange undershoots anymore!

def run_ln_dosc(ext_signal, params, filename,
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


    # lookup tables for quantities
    # data = loadmat(filename)
    # mu_range = data['presimdata'][0, 0]['Irange'][0]/(params['taum'])
    # sig_range = data['presimdata'][0, 0]['sigmarange'][0]
    # f0_grid = data['presimdata'][0, 0]['mu_dosc_f0']
    # omega_grid = f0_grid*2.*np.pi
    # #dr_grid = data['presimdata'][0, 0]['drdmu_raw']
    # r_grid = data['presimdata'][0, 0]['r_raw']
    # Vmean_grid = data['presimdata'][0, 0]['Vmean_raw']
    # tau_grid = data['presimdata'][0, 0]['mu_dosc_tau']
    # tau_grid[np.where(tau_grid < dt)] = dt
    # tau_mu_f_grid    = data['presimdata'][0, 0]['mu_exp_tau']
    # tau_mu_f_grid[np.where(tau_mu_f_grid < dt)] = dt # dt = 0.05 ms
    # tau_sigma_f_grid = data['presimdata'][0, 0]['sigma_exp_tau_deltacorrection']
    # tau_sigma_f_grid[np.where(tau_sigma_f_grid < dt)] = dt # dt = 0.05 ms

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

    res_sim = sim_ln_dosc(mu_ext,sigma_ext,dmu_ext_dt,t,dt,
                                 steps,mu_vals,sigma_vals,omega,
                                 tau,tau_mu_f,tau_sigma_f,
                                 V_mean_ss,r_ss,a,b,C,wm0, Ew,tauw,rec,K,J,
                                 delay_type,ndt,taud, uni_int_order)

    #store results in dictionary which is returned
    results_dict = dict()
    results_dict['r'] = res_sim[0]
    results_dict['t'] = t
    results_dict['mu_total'] = res_sim[2]
    # take all elements except last one so that len(time)==len(wm)
    if 'wm' in rec_vars: results_dict['wm'] = res_sim[1]
    if 'Vm' in rec_vars: results_dict['Vm'] = res_sim[3]
    return results_dict
###################################################################################################

# old model verions
##################
@njit
def sim_ln_dosc_complex_model_OLD(mu_ext, sigma_ext,L,mu_exp,sigma_exp,
                w_m_orig,dt,EW,taum,gL,tauw,a,b,mu_range,
                sig_range,f0_grid,tau_grid,tau_mu_f_grid,
                tau_sigma_f_grid,Vmean_grid,r_grid,rec,
                K,J,delay_type,n_d,taud):

    mu_f = np.zeros(L+1, dtype = numba.complex128) \
        if use_numba else np.zeros(L+1, dtype=np.complex128)
    mu_a = np.zeros(L+1)
    r = np.zeros(L+1)
    r_d = np.zeros(L+1)
    wm = np.zeros(L+1)
    sigma_f = np.zeros(L+1)
    #initial_conditions
    mu_a[0] = mu_exp
    wm[0] = w_m_orig
    sigma_f[0] = sigma_exp
    weights_a = interpolate_xy(mu_a[0]-wm[0]/(taum*gL), sigma_f[0],
                               mu_range, sig_range)
    f0  =         lookup_xy(f0_grid, weights_a)
    tau =         lookup_xy(tau_grid, weights_a)
    mu_f[0] = (mu_exp*(1+1j*2*np.pi*f0*tau))
    for i in xrange(L):
        #interpolates real numbers
        weights_a = interpolate_xy(mu_a[i]-wm[i]/(taum*gL), sigma_f[i], mu_range, sig_range)
        f0  =         lookup_xy(f0_grid, weights_a)
        tau =         lookup_xy(tau_grid, weights_a)
        tau_mu_f =    lookup_xy(tau_mu_f_grid, weights_a)
        tau_sigma_f = lookup_xy(tau_sigma_f_grid, weights_a)

        weights_f = interpolate_xy(mu_f[i].real-wm[i]/(taum*gL), sigma_f[i],mu_range, sig_range)
        Vmean_dosc = lookup_xy(Vmean_grid,weights_f)
        r_dosc = lookup_xy(r_grid,weights_f)
        r[i] = r_dosc

        if rec:
            mu_syn = get_mu_syn(K,J,mu_ext,delay_type,i,r,r_d,n_d,taud,dt)
            sigma_syn = get_sigma_syn(K,J,sigma_ext,delay_type,i,r,r_d,n_d,taud,dt)
        else:
            mu_syn = mu_ext[i]
            sigma_syn = sigma_ext[i]

        mu_f_rhs = ((2 * np.pi * f0)**2 * tau + 1./tau) * mu_syn + (1j * 2 * np.pi * f0 - 1./tau) * mu_f[i]
        sigma_f_rhs = (sigma_syn - sigma_f[i])/tau_sigma_f
        w_rhs = (a[i]*(Vmean_dosc - EW) - wm[i] + tauw * b[i] * r_dosc)/tauw
        mu_a_rhs = (mu_syn - mu_a[i])/tau_mu_f


        #EULER STEP
        mu_f[i+1] = mu_f[i] + dt * mu_f_rhs
        mu_a[i+1]    = mu_a[i] + dt * mu_a_rhs
        sigma_f[i+1] = sigma_f[i] + dt*sigma_f_rhs
        wm[i+1] = wm[i] + dt*w_rhs


    return (r*1000.,wm)

def run_ln_dosc_complex_model_OLD(ext_signal, params,
                rec_vars = ['wm'],
                rec = False, FS = False):

    if FS:
        NotImplementedError()

    print('running lndosc')

    #runtime parameters
    runtime = params['runtime']
    dt = params['uni_dt']
    steps = int(runtime/dt)
    t = np.linspace(0., runtime, steps+1)

    #external inputs
    mu_ext = ext_signal[0]    #[mV/ms]
    sigma_ext = ext_signal[1] #[mv/sqrt(ms)]

    #adaptation parameters
    a = params['a']
    b = params['b']
    tauw = params['tauw']
    EW = params['Ew']
    have_adap = True if (a.any() or b.any()) else False

    # coupling parameters
    K = params['K']
    J = params['J']
    taud = params['taud']
    ndt = int(params['const_delay']/dt)
    delay_type = params['delay_type']



    N = params['N_total']

    taum = params['C']/params['gL']
    gL = params['gL']
    #for rec



    # parameters are needed if initial values are computed
    # Vl = params['ALN_v_lower_bound']/mV # lower bound for voltage
    Vs = -40.0 #mV
    dVs = 0.5
    Vr = params['Vr']
    VT = params['VT']
    EL = params['EL']
    DeltaT = params['deltaT']
    C = params['C']

    filename_matlab = 'precalc05_int.mat'
    data = loadmat(filename_matlab)
    mu_range = data['presimdata'][0, 0]['Irange'][0]/(params['taum'])
    sig_range = data['presimdata'][0, 0]['sigmarange'][0]
    f0_grid = data['presimdata'][0, 0]['mu_dosc_f0']
    #dr_grid = data['presimdata'][0, 0]['drdmu_raw']
    r_grid = data['presimdata'][0, 0]['r_raw']
    Vmean_grid = data['presimdata'][0, 0]['Vmean_raw']
    tau_grid = data['presimdata'][0, 0]['mu_dosc_tau']
    tau_grid[np.where(tau_grid < dt)] = dt # dt = 0.05 ms
    #manipulation of data that minimum value of tau_mu, tau_sigma is dt!
    tau_mu_f_grid    = data['presimdata'][0, 0]['mu_exp_tau']
    tau_mu_f_grid[np.where(tau_mu_f_grid < dt)] = dt # dt = 0.05 ms
    tau_sigma_f_grid = data['presimdata'][0, 0]['sigma_exp_tau_deltacorrection']#deltacorrection
    tau_sigma_f_grid[np.where(tau_sigma_f_grid < dt)] = dt # dt = 0.05 ms

    #get initial values
    mu_exp = 0.#0.5
    sigma_exp = 0.#. 1.6
    #mu_exp, sigma_exp = get_initial_values(N, Vs, Vl, dVs, taum, EL, Vr, VT, DeltaT, C)
    mu_a_dosc = mu_exp
    sigma_f_dosc = sigma_exp
    w_orig_m = 0.#100.0
    w_m_dosc = w_orig_m


    # initial value of w_m still needs to be defined
    w_m0 = 0
    w_m= w_m0


    res_sim = sim_ln_dosc_complex_model_OLD(mu_ext,sigma_ext,steps,mu_exp,sigma_exp,
                          w_orig_m,dt,EW,taum,gL,tauw,a,b,mu_range,
                          sig_range,f0_grid,tau_grid,tau_mu_f_grid,
                          tau_sigma_f_grid,Vmean_grid,r_grid,rec,K,J,
                          delay_type,ndt,taud)

    #store results in dictionary which is returned
    results_dict = dict()
    results_dict['r'] = res_sim[0][:-1]
    results_dict['t'] = t[:-1]
    # take all elements except last one so that len(time)==len(wm)
    if 'wm' in rec_vars: results_dict['wm'] = res_sim[1][:-1]
    #put more
    # if 'Vm' in record_variables:
    # if 'XX' in record_variables:
    # if 'XX' in record_variables:
    return results_dict


@njit
def sim_ln_dosc_UPDATE_without_HEUN(mu_ext, sig_ext, dmu_ext_dt, t, dt, steps,
                       mu_range, sig_range, omega_grid,
                       tau_grid, tau_mu_f_grid,
                       tau_sigma_f_grid, Vmean_grid,
                       r_grid,a,b,C,wm0, EW, tauw, rec, K, J,
                       delay_type, const_delay,taud):

    # initialize arrays
    r     = np.zeros_like(t)
    r_d   = np.zeros_like(t)
    mu_f  = np.zeros_like(t)
    dmu_f = np.zeros_like(t)  #aux. var for integration
    mu_lookup = np.zeros_like(t)
    sig_f     = np.zeros_like(t)
    wm  = np.zeros_like(t)

    # set initial values
    wm[0] = wm0
    # mu_f[0]  = mu_f0
    # sig_f[0] = sig_f0

    for i in xrange(steps):

        # 1st interpolation and lookup with
        # exponentially filtered mu_ext for looking up omega,tau ..
        weights_1 = interpolate_xy(mu_lookup[i]-wm[i]/C,sig_f[i],mu_range, sig_range)
        omega  =      lookup_xy(omega_grid, weights_1)
        tau =         lookup_xy(tau_grid, weights_1)
        tau_mu_f =    lookup_xy(tau_mu_f_grid, weights_1)
        tau_sigma_f = lookup_xy(tau_sigma_f_grid, weights_1)
        # 2nd interpolation and lookup with
        # mu_ext filtered by damped oscillatory function for looking up Vm and r
        weights_2 = interpolate_xy(mu_f[i]-wm[i]/C, sig_f[i], mu_range, sig_range)
        Vm = lookup_xy(Vmean_grid,weights_2)
        r[i] = lookup_xy(r_grid,weights_2)

        if rec:
            mu_syn = get_mu_syn(K,J,mu_ext,delay_type,i,r,r_d,const_delay,taud,dt)
            sigma_syn = get_sigma_syn(K,J,sig_ext,delay_type,i,r,r_d,const_delay,taud,dt)
        else:
            mu_syn = mu_ext[i]
            sigma_syn = sig_ext[i]

        # rhs of the equations
        mu_lookup_rhs = (mu_syn-mu_lookup[i])/tau_mu_f
        sig_f_rhs = (sigma_syn-sig_f[i])/tau_sigma_f
        wm_rhs = (a[i]*(Vm - EW) - wm[i] + tauw * b[i] * r[i])/tauw

        # euler step
        A = (tau**2*omega**2+1)/tau
        mu_f[i+1] = mu_f[i] + dt*dmu_f[i]
        dmu_f[i+1] = dmu_f[i] + dt*( (-2./tau)*dmu_f[i]
                                    -(1./tau**2 + omega**2)*mu_f[i]
                                    +A*(mu_syn/tau+dmu_ext_dt[i]))
        # indentical to the old model
        mu_lookup[i+1] = mu_lookup[i] + dt*mu_lookup_rhs
        sig_f[i+1] = sig_f[i] + dt*sig_f_rhs
        wm[i+1] = wm[i] + dt*wm_rhs

    return (r*1000., wm)

# new version of the ln dosc model. In contrast the the funtion run_ln_dosc
# this loop filters the external mu input by intgrating a second order differential
# equation (real) and not a complex first order ODE. Yielded improvements
# with respect to changes in the sigma input: no strange undershoots anymore!

def run_ln_dosc_UPDATE_without_HEUN(ext_signal, params, filename_mat,
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

    # membrane capacitance
    C = params['C']


    # lookup tables for quantities
    data = loadmat(filename_mat)
    mu_range = data['presimdata'][0, 0]['Irange'][0]/(params['taum'])
    sig_range = data['presimdata'][0, 0]['sigmarange'][0]
    f0_grid = data['presimdata'][0, 0]['mu_dosc_f0']
    omega_grid = f0_grid*2.*np.pi
    #dr_grid = data['presimdata'][0, 0]['drdmu_raw']
    r_grid = data['presimdata'][0, 0]['r_raw']
    Vmean_grid = data['presimdata'][0, 0]['Vmean_raw']
    tau_grid = data['presimdata'][0, 0]['mu_dosc_tau']
    tau_grid[np.where(tau_grid < dt)] = dt
    tau_mu_f_grid    = data['presimdata'][0, 0]['mu_exp_tau']
    tau_mu_f_grid[np.where(tau_mu_f_grid < dt)] = dt # dt = 0.05 ms
    tau_sigma_f_grid = data['presimdata'][0, 0]['sigma_exp_tau_deltacorrection']
    tau_sigma_f_grid[np.where(tau_sigma_f_grid < dt)] = dt # dt = 0.05 ms

    # todo, find out:
    # should there be initial values for mu_f/sigma_f or just zero?

    # initial value != 0 if we have adaptation
    wm0 = params['wm_init'] if have_apad else False

    # time derivative of input
    dmu_ext_dt = np.diff(mu_ext)/dt
    dmu_ext_dt = np.append(dmu_ext_dt,dmu_ext_dt[-1])

    res_sim = sim_ln_dosc_UPDATE_without_HEUN(mu_ext,sigma_ext,dmu_ext_dt,t,dt,
                                 steps,mu_range,sig_range,omega_grid,
                                 tau_grid,tau_mu_f_grid,tau_sigma_f_grid,
                                 Vmean_grid,r_grid,a,b,C,wm0, Ew,tauw,rec,K,J,
                                 delay_type,ndt,taud)

    #store results in dictionary which is returned
    results_dict = dict()
    results_dict['r'] = res_sim[0]
    results_dict['t'] = t
    # take all elements except last one so that len(time)==len(wm)
    if 'wm' in rec_vars: results_dict['wm'] = res_sim[1]
    return results_dict




# imports
import numpy as np
import tables
from misc.utils import interpolate_xy, lookup_xy, get_mu_syn, get_sigma_syn
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


# TODO: correct derivative of external mean input


# current ln_dosc model version with HEUN time intergation method
#################################################################
@njit
def sim_ln_dosc(mu_ext, sig_ext, dmu_ext_dt, t, dt, steps,
                       mu_range, sig_range, omega_grid,
                       tau_grid, tau_mu_f_grid,
                       tau_sigma_f_grid, Vmean_grid,
                       r_grid,a,b,C,wm0, EW, tauw, rec, K, J,
                       delay_type, const_delay,taud,uni_int_order):

    # initialize arrays
    r     = np.zeros_like(t)
    r_d   = np.zeros_like(t)
    mu_f  = np.zeros_like(t)
    mu_total = np.zeros_like(t)
    dmu_f = np.zeros_like(t)      #aux. var for integration
    dmu_syn_dt = np.zeros_like(t)
    sig_f     = np.zeros_like(t)
    wm  = np.zeros_like(t)
    # new arrays for HEUN method
    Vm = np.zeros_like(t)
    mu_syn = np.zeros_like(t)
    sigma_syn = np.zeros_like(t)
    omega = np.zeros_like(t)
    tau = np.zeros_like(t)
    tau_mu_f = np.zeros_like(t)
    tau_sigma_f = np.zeros_like(t)
    A = np.zeros_like(t)


    # set initial values
    wm[0] = wm0
    # mu_f[0]  = mu_f0
    # sig_f[0] = sig_f0

    for i in xrange(steps):
        for j in range(int(uni_int_order)):



            # mu_ext filtered by damped oscillatory function for looking up Vm and r
            weights_2 = interpolate_xy(mu_f[i+j]-wm[i+j]/C, sig_f[i+j],
                                       mu_range, sig_range)
            Vm[i+j] = lookup_xy(Vmean_grid,weights_2)
            r[i+j] = lookup_xy(r_grid,weights_2)

            if rec:
                mu_syn[i+j] = get_mu_syn(K,J,mu_ext,delay_type,i+j,
                                    r,r_d,const_delay,taud,dt)
                sigma_syn[i+j] = get_sigma_syn(K,J,sig_ext,delay_type,i+j,
                                          r,r_d,const_delay,taud,dt)

                # in the case of exponentially distributed
                # delays; compute dmu_syn_dt analytically
                if delay_type == 2:
                    dmu_syn_dt[i+j] = dmu_ext_dt[i+j] + K*J*(r[i+j]-r_d[i+j])/taud
                # else finite differences mu_syn
                else:
                    dmu_syn_dt[i+j] = (mu_syn[i+j]-mu_syn[i+j-1])/dt

            # this case corresponds to an UNCOUPLED POPULATION of neurons, i.e. K=0.
            else:
                mu_syn[i+j] = mu_ext[i+j]
                sigma_syn[i+j] = sig_ext[i+j]
                dmu_syn_dt[i+j] = dmu_ext_dt[i+j]


            # save mu_total for param exploration
            # todo: remove this again
            mu_total[i+j] = mu_syn[i+j]-wm[i+j]/C


            # todo
            weights_1 = interpolate_xy(mu_total[i+j],sig_f[i+j],
                                       mu_range, sig_range)
            omega[i+j]  =      lookup_xy(omega_grid, weights_1)
            tau[i+j] =         lookup_xy(tau_grid, weights_1)
            tau_mu_f[i+j] =    lookup_xy(tau_mu_f_grid, weights_1)
            tau_sigma_f[i+j] = lookup_xy(tau_sigma_f_grid, weights_1)
            # 2nd interpolation and lookup with



            # euler step
            A[i+j] = (tau[i+j]**2*omega[i+j]**2+1)/tau[i+j]

            if j == 0:
                mu_f[i+1] = mu_f[i] + dt*dmu_f[i]
                dmu_f[i+1] = dmu_f[i] + dt*( (-2./tau[i])*dmu_f[i]
                                        -(1./tau[i]**2 + omega[i]**2)*mu_f[i]
                                        +A[i]*(mu_syn[i]/tau[i]+dmu_syn_dt[i]))

                # indentical to the old model
                # mu_lookup[i+1] = mu_lookup[i] + dt*(mu_syn[i]-mu_lookup[i])/tau_mu_f[i]
                sig_f[i+1] = sig_f[i] + dt*(sigma_syn[i]-sig_f[i])/tau_sigma_f[i]
                wm[i+1] = wm[i] + dt*(a[i]*(Vm[i] - EW) - wm[i] + tauw * b[i] * r[i])/tauw
            if j == 1:
                mu_f[i+1] = mu_f[i] + dt/2.*(dmu_f[i]+dmu_f[i+1])
                dmu_f[i+1] = dmu_f[i] + dt/2. * (( (-2./tau[i])*dmu_f[i]
                                        -(1./tau[i]**2 + omega[i]**2)*mu_f[i]
                                        +A[i]*(mu_syn[i]/tau[i]+dmu_syn_dt[i]))
                                        +( (-2./tau[i+1])*dmu_f[i+1]
                                        -(1./tau[i+1]**2 + omega[i+1]**2)*mu_f[i+1]
                                        +A[i+1]*(mu_syn[i+1]/tau[i+1]+dmu_syn_dt[i+1])))

                # indentical to the old model
                # mu_lookup[i+1] = mu_lookup[i] + dt/2.*((mu_syn[i]-mu_lookup[i])/tau_mu_f[i]
                #                                       +(mu_syn[i+1]-mu_lookup[i+1])/tau_mu_f[i+1])

                sig_f[i+1] = sig_f[i] + dt/2.*((sigma_syn[i]-sig_f[i])/tau_sigma_f[i]
                                            +(sigma_syn[i+1]-sig_f[i+1])/tau_sigma_f[i+1])
                wm[i+1] = wm[i] + dt/2.*((a[i]*(Vm[i] - EW) - wm[i] + tauw * b[i] * r[i])/tauw
                                      +(a[i+1]*(Vm[i+1] - EW) - wm[i+1] + tauw * b[i+1] * r[i+1])/tauw)
    return (r*1000., wm, mu_total, Vm)

# new version of the ln dosc model. In contrast the the funtion run_ln_dosc
# this loop filters the external mu input by intgrating a second order differential
# equation (real) and not a complex first order ODE. Yielded improvements
# with respect to changes in the sigma input: no strange undershoots anymore!

def run_ln_dosc(ext_signal, params, filename,
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


    # lookup tables for quantities
    # data = loadmat(filename)
    # mu_range = data['presimdata'][0, 0]['Irange'][0]/(params['taum'])
    # sig_range = data['presimdata'][0, 0]['sigmarange'][0]
    # f0_grid = data['presimdata'][0, 0]['mu_dosc_f0']
    # omega_grid = f0_grid*2.*np.pi
    # #dr_grid = data['presimdata'][0, 0]['drdmu_raw']
    # r_grid = data['presimdata'][0, 0]['r_raw']
    # Vmean_grid = data['presimdata'][0, 0]['Vmean_raw']
    # tau_grid = data['presimdata'][0, 0]['mu_dosc_tau']
    # tau_grid[np.where(tau_grid < dt)] = dt
    # tau_mu_f_grid    = data['presimdata'][0, 0]['mu_exp_tau']
    # tau_mu_f_grid[np.where(tau_mu_f_grid < dt)] = dt # dt = 0.05 ms
    # tau_sigma_f_grid = data['presimdata'][0, 0]['sigma_exp_tau_deltacorrection']
    # tau_sigma_f_grid[np.where(tau_sigma_f_grid < dt)] = dt # dt = 0.05 ms

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

    res_sim = sim_ln_dosc(mu_ext,sigma_ext,dmu_ext_dt,t,dt,
                                 steps,mu_vals,sigma_vals,omega,
                                 tau,tau_mu_f,tau_sigma_f,
                                 V_mean_ss,r_ss,a,b,C,wm0, Ew,tauw,rec,K,J,
                                 delay_type,ndt,taud, uni_int_order)

    #store results in dictionary which is returned
    results_dict = dict()
    results_dict['r'] = res_sim[0]
    results_dict['t'] = t
    results_dict['mu_total'] = res_sim[2]
    # take all elements except last one so that len(time)==len(wm)
    if 'wm' in rec_vars: results_dict['wm'] = res_sim[1]
    if 'Vm' in rec_vars: results_dict['Vm'] = res_sim[3]
    return results_dict
###################################################################################################

# old model verions
##################
@njit
def sim_ln_dosc_complex_model_OLD(mu_ext, sigma_ext,L,mu_exp,sigma_exp,
                w_m_orig,dt,EW,taum,gL,tauw,a,b,mu_range,
                sig_range,f0_grid,tau_grid,tau_mu_f_grid,
                tau_sigma_f_grid,Vmean_grid,r_grid,rec,
                K,J,delay_type,n_d,taud):

    mu_f = np.zeros(L+1, dtype = numba.complex128) \
        if use_numba else np.zeros(L+1, dtype=np.complex128)
    mu_a = np.zeros(L+1)
    r = np.zeros(L+1)
    r_d = np.zeros(L+1)
    wm = np.zeros(L+1)
    sigma_f = np.zeros(L+1)
    #initial_conditions
    mu_a[0] = mu_exp
    wm[0] = w_m_orig
    sigma_f[0] = sigma_exp
    weights_a = interpolate_xy(mu_a[0]-wm[0]/(taum*gL), sigma_f[0],
                               mu_range, sig_range)
    f0  =         lookup_xy(f0_grid, weights_a)
    tau =         lookup_xy(tau_grid, weights_a)
    mu_f[0] = (mu_exp*(1+1j*2*np.pi*f0*tau))
    for i in xrange(L):
        #interpolates real numbers
        weights_a = interpolate_xy(mu_a[i]-wm[i]/(taum*gL), sigma_f[i], mu_range, sig_range)
        f0  =         lookup_xy(f0_grid, weights_a)
        tau =         lookup_xy(tau_grid, weights_a)
        tau_mu_f =    lookup_xy(tau_mu_f_grid, weights_a)
        tau_sigma_f = lookup_xy(tau_sigma_f_grid, weights_a)

        weights_f = interpolate_xy(mu_f[i].real-wm[i]/(taum*gL), sigma_f[i],mu_range, sig_range)
        Vmean_dosc = lookup_xy(Vmean_grid,weights_f)
        r_dosc = lookup_xy(r_grid,weights_f)
        r[i] = r_dosc

        if rec:
            mu_syn = get_mu_syn(K,J,mu_ext,delay_type,i,r,r_d,n_d,taud,dt)
            sigma_syn = get_sigma_syn(K,J,sigma_ext,delay_type,i,r,r_d,n_d,taud,dt)
        else:
            mu_syn = mu_ext[i]
            sigma_syn = sigma_ext[i]

        mu_f_rhs = ((2 * np.pi * f0)**2 * tau + 1./tau) * mu_syn + (1j * 2 * np.pi * f0 - 1./tau) * mu_f[i]
        sigma_f_rhs = (sigma_syn - sigma_f[i])/tau_sigma_f
        w_rhs = (a[i]*(Vmean_dosc - EW) - wm[i] + tauw * b[i] * r_dosc)/tauw
        mu_a_rhs = (mu_syn - mu_a[i])/tau_mu_f


        #EULER STEP
        mu_f[i+1] = mu_f[i] + dt * mu_f_rhs
        mu_a[i+1]    = mu_a[i] + dt * mu_a_rhs
        sigma_f[i+1] = sigma_f[i] + dt*sigma_f_rhs
        wm[i+1] = wm[i] + dt*w_rhs


    return (r*1000.,wm)

def run_ln_dosc_complex_model_OLD(ext_signal, params,
                rec_vars = ['wm'],
                rec = False, FS = False):

    if FS:
        NotImplementedError()

    print('running lndosc')

    #runtime parameters
    runtime = params['runtime']
    dt = params['uni_dt']
    steps = int(runtime/dt)
    t = np.linspace(0., runtime, steps+1)

    #external inputs
    mu_ext = ext_signal[0]    #[mV/ms]
    sigma_ext = ext_signal[1] #[mv/sqrt(ms)]

    #adaptation parameters
    a = params['a']
    b = params['b']
    tauw = params['tauw']
    EW = params['Ew']
    have_adap = True if (a.any() or b.any()) else False

    # coupling parameters
    K = params['K']
    J = params['J']
    taud = params['taud']
    ndt = int(params['const_delay']/dt)
    delay_type = params['delay_type']



    N = params['N_total']

    taum = params['C']/params['gL']
    gL = params['gL']
    #for rec



    # parameters are needed if initial values are computed
    # Vl = params['ALN_v_lower_bound']/mV # lower bound for voltage
    Vs = -40.0 #mV
    dVs = 0.5
    Vr = params['Vr']
    VT = params['VT']
    EL = params['EL']
    DeltaT = params['deltaT']
    C = params['C']

    filename_matlab = 'precalc05_int.mat'
    data = loadmat(filename_matlab)
    mu_range = data['presimdata'][0, 0]['Irange'][0]/(params['taum'])
    sig_range = data['presimdata'][0, 0]['sigmarange'][0]
    f0_grid = data['presimdata'][0, 0]['mu_dosc_f0']
    #dr_grid = data['presimdata'][0, 0]['drdmu_raw']
    r_grid = data['presimdata'][0, 0]['r_raw']
    Vmean_grid = data['presimdata'][0, 0]['Vmean_raw']
    tau_grid = data['presimdata'][0, 0]['mu_dosc_tau']
    tau_grid[np.where(tau_grid < dt)] = dt # dt = 0.05 ms
    #manipulation of data that minimum value of tau_mu, tau_sigma is dt!
    tau_mu_f_grid    = data['presimdata'][0, 0]['mu_exp_tau']
    tau_mu_f_grid[np.where(tau_mu_f_grid < dt)] = dt # dt = 0.05 ms
    tau_sigma_f_grid = data['presimdata'][0, 0]['sigma_exp_tau_deltacorrection']#deltacorrection
    tau_sigma_f_grid[np.where(tau_sigma_f_grid < dt)] = dt # dt = 0.05 ms

    #get initial values
    mu_exp = 0.#0.5
    sigma_exp = 0.#. 1.6
    #mu_exp, sigma_exp = get_initial_values(N, Vs, Vl, dVs, taum, EL, Vr, VT, DeltaT, C)
    mu_a_dosc = mu_exp
    sigma_f_dosc = sigma_exp
    w_orig_m = 0.#100.0
    w_m_dosc = w_orig_m


    # initial value of w_m still needs to be defined
    w_m0 = 0
    w_m= w_m0


    res_sim = sim_ln_dosc_complex_model_OLD(mu_ext,sigma_ext,steps,mu_exp,sigma_exp,
                          w_orig_m,dt,EW,taum,gL,tauw,a,b,mu_range,
                          sig_range,f0_grid,tau_grid,tau_mu_f_grid,
                          tau_sigma_f_grid,Vmean_grid,r_grid,rec,K,J,
                          delay_type,ndt,taud)

    #store results in dictionary which is returned
    results_dict = dict()
    results_dict['r'] = res_sim[0][:-1]
    results_dict['t'] = t[:-1]
    # take all elements except last one so that len(time)==len(wm)
    if 'wm' in rec_vars: results_dict['wm'] = res_sim[1][:-1]
    #put more
    # if 'Vm' in record_variables:
    # if 'XX' in record_variables:
    # if 'XX' in record_variables:
    return results_dict


@njit
def sim_ln_dosc_UPDATE_without_HEUN(mu_ext, sig_ext, dmu_ext_dt, t, dt, steps,
                       mu_range, sig_range, omega_grid,
                       tau_grid, tau_mu_f_grid,
                       tau_sigma_f_grid, Vmean_grid,
                       r_grid,a,b,C,wm0, EW, tauw, rec, K, J,
                       delay_type, const_delay,taud):

    # initialize arrays
    r     = np.zeros_like(t)
    r_d   = np.zeros_like(t)
    mu_f  = np.zeros_like(t)
    dmu_f = np.zeros_like(t)  #aux. var for integration
    mu_lookup = np.zeros_like(t)
    sig_f     = np.zeros_like(t)
    wm  = np.zeros_like(t)

    # set initial values
    wm[0] = wm0
    # mu_f[0]  = mu_f0
    # sig_f[0] = sig_f0

    for i in xrange(steps):

        # 1st interpolation and lookup with
        # exponentially filtered mu_ext for looking up omega,tau ..
        weights_1 = interpolate_xy(mu_lookup[i]-wm[i]/C,sig_f[i],mu_range, sig_range)
        omega  =      lookup_xy(omega_grid, weights_1)
        tau =         lookup_xy(tau_grid, weights_1)
        tau_mu_f =    lookup_xy(tau_mu_f_grid, weights_1)
        tau_sigma_f = lookup_xy(tau_sigma_f_grid, weights_1)
        # 2nd interpolation and lookup with
        # mu_ext filtered by damped oscillatory function for looking up Vm and r
        weights_2 = interpolate_xy(mu_f[i]-wm[i]/C, sig_f[i], mu_range, sig_range)
        Vm = lookup_xy(Vmean_grid,weights_2)
        r[i] = lookup_xy(r_grid,weights_2)

        if rec:
            mu_syn = get_mu_syn(K,J,mu_ext,delay_type,i,r,r_d,const_delay,taud,dt)
            sigma_syn = get_sigma_syn(K,J,sig_ext,delay_type,i,r,r_d,const_delay,taud,dt)
        else:
            mu_syn = mu_ext[i]
            sigma_syn = sig_ext[i]

        # rhs of the equations
        mu_lookup_rhs = (mu_syn-mu_lookup[i])/tau_mu_f
        sig_f_rhs = (sigma_syn-sig_f[i])/tau_sigma_f
        wm_rhs = (a[i]*(Vm - EW) - wm[i] + tauw * b[i] * r[i])/tauw

        # euler step
        A = (tau**2*omega**2+1)/tau
        mu_f[i+1] = mu_f[i] + dt*dmu_f[i]
        dmu_f[i+1] = dmu_f[i] + dt*( (-2./tau)*dmu_f[i]
                                    -(1./tau**2 + omega**2)*mu_f[i]
                                    +A*(mu_syn/tau+dmu_ext_dt[i]))
        # indentical to the old model
        mu_lookup[i+1] = mu_lookup[i] + dt*mu_lookup_rhs
        sig_f[i+1] = sig_f[i] + dt*sig_f_rhs
        wm[i+1] = wm[i] + dt*wm_rhs

    return (r*1000., wm)

# new version of the ln dosc model. In contrast the the funtion run_ln_dosc
# this loop filters the external mu input by intgrating a second order differential
# equation (real) and not a complex first order ODE. Yielded improvements
# with respect to changes in the sigma input: no strange undershoots anymore!

def run_ln_dosc_UPDATE_without_HEUN(ext_signal, params, filename_mat,
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

    # membrane capacitance
    C = params['C']


    # lookup tables for quantities
    data = loadmat(filename_mat)
    mu_range = data['presimdata'][0, 0]['Irange'][0]/(params['taum'])
    sig_range = data['presimdata'][0, 0]['sigmarange'][0]
    f0_grid = data['presimdata'][0, 0]['mu_dosc_f0']
    omega_grid = f0_grid*2.*np.pi
    #dr_grid = data['presimdata'][0, 0]['drdmu_raw']
    r_grid = data['presimdata'][0, 0]['r_raw']
    Vmean_grid = data['presimdata'][0, 0]['Vmean_raw']
    tau_grid = data['presimdata'][0, 0]['mu_dosc_tau']
    tau_grid[np.where(tau_grid < dt)] = dt
    tau_mu_f_grid    = data['presimdata'][0, 0]['mu_exp_tau']
    tau_mu_f_grid[np.where(tau_mu_f_grid < dt)] = dt # dt = 0.05 ms
    tau_sigma_f_grid = data['presimdata'][0, 0]['sigma_exp_tau_deltacorrection']
    tau_sigma_f_grid[np.where(tau_sigma_f_grid < dt)] = dt # dt = 0.05 ms

    # todo, find out:
    # should there be initial values for mu_f/sigma_f or just zero?

    # initial value != 0 if we have adaptation
    wm0 = params['wm_init'] if have_apad else False

    # time derivative of input
    dmu_ext_dt = np.diff(mu_ext)/dt
    dmu_ext_dt = np.append(dmu_ext_dt,dmu_ext_dt[-1])

    res_sim = sim_ln_dosc_UPDATE_without_HEUN(mu_ext,sigma_ext,dmu_ext_dt,t,dt,
                                 steps,mu_range,sig_range,omega_grid,
                                 tau_grid,tau_mu_f_grid,tau_sigma_f_grid,
                                 Vmean_grid,r_grid,a,b,C,wm0, Ew,tauw,rec,K,J,
                                 delay_type,ndt,taud)

    #store results in dictionary which is returned
    results_dict = dict()
    results_dict['r'] = res_sim[0]
    results_dict['t'] = t
    # take all elements except last one so that len(time)==len(wm)
    if 'wm' in rec_vars: results_dict['wm'] = res_sim[1]
    return results_dict




