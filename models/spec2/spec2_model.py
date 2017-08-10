from __future__ import print_function
import sys
from scipy.misc import derivative
from matplotlib.pyplot import *
import time
from misc.utils import interpolate_xy, lookup_xy, x_filter, outside_grid_warning
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


@njit
def sim_spec2(mu_ext, dmu_ext_dt, d2mu_ext_dt2, sig_ext,
               dsig_ext2_dt, d2sig_ext2_dt2, steps, t, dt, mu_range,
               sig_range, rec, delay_type, const_delay, K, J, taud,
               tauw, a, b, C,wm0, Ew, r_inf_map, Vm_inf_map, T_map,
               S_sig_map, D_map, M_map, Fmu_map, Fsig_map, dVm_dsig_map,
               dVm_dmu_map, dr_inf_dmu_map, dr_inf_dsig_map, fC_mu_map,
               fC_sig_map, uni_int_order, rectify, grid_warn = True):


    # state variables
    r  = np.zeros_like(t)
    r_d = np.zeros(len(t)+1)
    # r_rec = np.zeros_like(t)

    dr = np.zeros_like(t)
    wm = np.zeros_like(t)
    
    # set initial value(s)
    wm[0]=wm0

    # total moments
    mu_tot = np.zeros_like(t)
    sig_tot= np.zeros_like(t)
    mu_syn = np.zeros_like(t)

    # lumped quantities of the spec2 model
    D = np.zeros_like(t)
    T = np.zeros_like(t)
    M = np.zeros_like(t)
    Ssig2 = np.zeros_like(t)
    TM = np.zeros_like(t)
    DM = np.zeros_like(t)
    DSsig2 = np.zeros_like(t)
    TSsig2 = np.zeros_like(t)
    Fmu = np.zeros_like(t)
    DFmu = np.zeros_like(t)
    A = np.zeros_like(t)
    Fsig2 = np.zeros_like(t)
    DFsig2 = np.zeros_like(t)
    fCmu = np.zeros_like(t)
    fCsig2 = np.zeros_like(t)
    dr_dmu = np.zeros_like(t)
    dr_dsig2 = np.zeros_like(t)
    dVm_dmu = np.zeros_like(t)
    dVm_dsig2 = np.zeros_like(t)
    Vm_inf = np.zeros_like(t)
    r_inf = np.zeros_like(t)
    
    
    ddr = np.zeros_like(t)
    r_d_dot = np.zeros_like(t)
    r_d_dotdot = np.zeros_like(t)

    # some other abbreviations
    R = np.zeros_like(t)
    H_mu = np.zeros_like(t)
    H_sigma2 = np.zeros_like(t)
    # integration coefficients
    beta0 = np.zeros_like(t)
    beta1 = np.zeros_like(t)
    beta2 = np.zeros_like(t)
    betaC = np.zeros_like(t)

    betaC_tilde = 0.
    beta0_tilde = 0.
    beta1_tilde = 0.
    beta2_tilde = 0.
    
    
    dt_taud = dt/taud
    dt_tauw = dt/tauw
    # ensure at least one time step
    ndt = max(1, int(const_delay/dt))


    # integration loop
    for i in xrange(steps):
        for j in xrange(int(uni_int_order)):
            if rec:
                # no delay
                if delay_type == 0:
                    r_d[i+j] = r[i+j]
                elif delay_type == 1:
                    # r is initialized as
                    r_d[i+j] = r[i+j-ndt]
                # exp. distributed delays
                elif delay_type == 2:
                    r_d[i+j+1] = r_d[i+j]+dt_taud*(r[i+j]-r_d[i+j])
                # exp. distributed delays + const. delay
                elif delay_type == 3:
                    r_d[i+j+1] = r_d[i+j]+(dt/taud)*(r[i+j-ndt]-r_d[i+j])
                    r_d_dot[i+j] = (1./taud)*(r[i+j-ndt]-r_d[i+j])
                    r_d_dotdot[i+j] = (1./taud)*(dr[i+j]-(r[i+j-ndt] - r_d[i+j])/taud)
                # no other delay_type is implemented
                else:
                    raise NotImplementedError
                
                # compute the total synaptic inputs
                mu_syn[i+j] = mu_ext[i+j] + K*J*r_d[i+j]
                mu_tot[i+j] = mu_syn[i+j] - wm[i+j]/C
                sig_tot[i+j] = np.sqrt(sig_ext[i+j]**2 + K*J**2*r_d[i+j])

            # no recurrency: this case corresponds to an UNCOUPLED populations of neurons, i.e. K=0.
            else:
                mu_tot[i+j] = mu_ext[i+j] - wm[i+j]/C
                sig_tot[i+j] = sig_ext[i+j]



            # outside-grid warning
            if grid_warn and j == 0:
                outside_grid_warning(mu_tot[i+j], sig_tot[i+j], mu_range, sig_range, dt*i)
            # get weights for looking up the quantities for the respective 
            # total synaptic input moments
            weights = interpolate_xy(mu_tot[i+j], sig_tot[i+j], mu_range, sig_range)
            M[i+j] = lookup_xy(M_map, weights)
            S_sig = lookup_xy(S_sig_map, weights)  # first compute w.r.t. sig2 before saving
            D[i+j] = lookup_xy(D_map, weights)
            T[i+j] = lookup_xy(T_map, weights)
            dVm_dmu[i+j] = lookup_xy(dVm_dmu_map, weights)
            dVm_dsig = lookup_xy(dVm_dsig_map, weights)
            Fmu[i+j] = lookup_xy(Fmu_map, weights)
            Fsig = lookup_xy(Fsig_map, weights)    # first compute w.r.t. sig2 before saving
            Vm_inf[i+j] = lookup_xy(Vm_inf_map, weights)
            r_inf[i+j] = lookup_xy(r_inf_map, weights)

            # debug
            fCmu[i+j] = lookup_xy(fC_mu_map, weights)
            fCsig = lookup_xy(fC_sig_map, weights)  # first compute w.r.t. sig2 before saving
            dr_dmu[i+j] = lookup_xy(dr_inf_dmu_map, weights)
            dr_dsig = lookup_xy(dr_inf_dsig_map, weights) # first compute w.r.t. sig2 before saving



            # after the look up scale the quantities depending on sig by 2*sig_tot
            dVm_dsig2[i+j] = dVm_dsig/(2*sig_tot[i+j])
            Fsig2[i+j] = Fsig/(2*sig_tot[i+j])
            Ssig2[i+j] = S_sig/(2*sig_tot[i+j])



            # A is the same for all settings
            # todo: remove A from tex AND code
            # todo: make r_d array
            A[i+j] = (a[i+j]*(Vm_inf[i+j]-Ew)-wm[i+j])/(tauw*C)



            # save debug quantities in each time step
            TM[i+j]                   = T[i+j]*M[i+j]
            DM[i+j]                   = D[i+j]*M[i+j]
            DSsig2[i+j]               = D[i+j]*Ssig2[i+j]
            TSsig2[i+j]               = T[i+j]*Ssig2[i+j]
            DFmu[i+j]                 = D[i+j]*Fmu[i+j]
            DFsig2[i+j]               = D[i+j]*Fsig2[i+j]



            # some abbreviations
            a_tauwC = a[i+j]/(tauw*C)
            b_tauwC = b[i+j]/(tauw*C)
            b_C = b[i+j]/C
            inv_taud = 1./taud
            inv_taud2 = 1./taud**2

            H_mu[i+j] = T[i+j]*M[i+j]\
                        +D[i+j]*M[i+j]*a_tauwC*dVm_dmu[i+j]\
                        -D[i+j]*Fmu[i+j]

            # H_(sigma)**2
            H_sigma2[i+j] =  T[i+j]*Ssig2[i+j]\
                             +D[i+j]*M[i+j]*a_tauwC*dVm_dsig2[i+j]\
                             -D[i+j]*Fsig2[i+j]


            if rec:
                R[i+j] = D[i+j]*M[i+j]*K*J\
                         +D[i+j]*Ssig2[i+j]*K*J**2

                if delay_type == 0:
                    beta2_tilde = -R[i+j]
                    beta1_tilde = K*J*H_mu[i+j]\
                                  +K*J**2*H_sigma2[i+j]
                    beta0_tilde = 0.
                    betaC_tilde = 0.

                elif delay_type == 1:
                    beta2_tilde = 0.
                    beta1_tilde = 0.
                    beta0_tilde = 0.

                elif delay_type == 2:
                    beta2_tilde = 0.
                    beta1_tilde = -inv_taud*R[i+j]
                    beta0_tilde = inv_taud*(K*J*H_mu[i+j]
                                            +K*J**2*H_sigma2[i+j])\
                                            +inv_taud2*R[i+j]
                    betaC_tilde = -(inv_taud*(K*J*H_mu[i+j]+K*J**2*H_sigma2[i+j])
                                    +inv_taud2*R[i+j])*r_d[i+j]

                elif delay_type == 3:
                    beta2_tilde = 0.
                    beta1_tilde = 0.
                    beta0_tilde = 0.

                else:
                    raise NotImplementedError

                beta2[i+j] = beta2_tilde + D[i+j]
                beta1[i+j] = beta1_tilde - T[i+j] + D[i+j]*M[i+j]*b_C
                beta0[i+j] = beta0_tilde - D[i+j]*M[i+j]*b_tauwC-b_C*H_mu[i+j]
                betaC[i+j] = betaC_tilde - (d2mu_ext_dt2[i+j]
                                            + (a[i+j] * (Vm_inf[i+j]-Ew)-wm[i+j])/(tauw**2*C))*D[i+j]*M[i+j]\
                             -d2sig_ext2_dt2[i+j]*D[i+j]*Ssig2[i+j]\
                             +(dmu_ext_dt[i+j]-(a[i+j] * (Vm_inf[i+j]-Ew)-wm[i+j])/(tauw*C))*H_mu[i+j]\
                             +dsig_ext2_dt[i+j]*H_sigma2[i+j]

                # very hacky: finish delay_type == 1 [betaC] down here
                if delay_type == 1:
                    betaC[i+j] = - ddr[i+j-ndt]*R[i+j] + dr[i+j-ndt]*(K*J*H_mu[i+j]+K*J**2*H_sigma2[i+j])\
                                 -(d2mu_ext_dt2[i+j] +(a[i+j] * (Vm_inf[i+j]-Ew)-wm[i+j]) /(tauw**2*C))*D[i+j]*M[i+j]\
                                 -d2sig_ext2_dt2[i+j]*D[i+j]*Ssig2[i+j]\
                                 +(dmu_ext_dt[i+j]-(a[i+j] * (Vm_inf[i+j]-Ew)-wm[i+j])/(tauw*C))*H_mu[i+j]\
                                 +dsig_ext2_dt[i+j]*H_sigma2[i+j]

                if delay_type == 3:
                    betaC[i+j] = - r_d_dotdot[i+j-ndt]*R[i+j] + r_d_dot[i+j-ndt]*(K*J*H_mu[i+j]+K*J**2*H_sigma2[i+j])\
                                 -(d2mu_ext_dt2[i+j] +(a[i+j] * (Vm_inf[i+j]-Ew)-wm[i+j]) /(tauw**2*C))*D[i+j]*M[i+j]\
                                 -d2sig_ext2_dt2[i+j]*D[i+j]*Ssig2[i+j]\
                                 +(dmu_ext_dt[i+j]-(a[i+j] * (Vm_inf[i+j]-Ew)-wm[i+j])/(tauw*C))*H_mu[i+j]\
                                 +dsig_ext2_dt[i+j]*H_sigma2[i+j]

            # this corresponds to the case of an UNCOUPLED POPULATION of neurons, i.e. K=0.
            else:
                beta2[i+j] =  D[i+j]
                beta1[i+j] = -T[i+j] + D[i+j]*M[i+j]*b_C
                beta0[i+j] = -D[i+j]*M[i+j]*b_tauwC-b_C*H_mu[i+j]

                betaC[i+j] = -D[i+j]*M[i+j]*(d2mu_ext_dt2[i+j] + (a[i+j] * (Vm_inf[i+j]-Ew)-wm[i+j])/(tauw**2*C))\
                             -D[i+j]*Ssig2[i+j]*d2sig_ext2_dt2[i+j]\
                             +(dmu_ext_dt[i+j] - (a[i+j] * (Vm_inf[i+j]-Ew)-wm[i+j])/(tauw*C))*H_mu[i+j]\
                             +dsig_ext2_dt[i+j]*H_sigma2[i+j]

            
            # j == 0: corresponds to a simple Euler method. 
            # If j reaches 1 (-> j!=0) the ELSE block is executed 
            # which updates using the HEUN scheme.
            if j == 0:
                dr[i+1]= dr[i] + dt* (-beta1[i]*dr[i]-(beta0[i]+1.)*r[i]+r_inf[i]-betaC[i])/beta2[i] 
                r[i+1] = r[i]  + dt* dr[i]
                wm[i+1] = wm[i]+ dt_tauw * (a[i]*(Vm_inf[i]-Ew)- wm[i] + b[i] * tauw * r[i])
                ddr[i] = (dr[i] - dr[i-1])/dt 
            # only perform Heun integration step if 'uni_int_order' == 2
            else:
                dr[i+1]= dr[i] + dt/2.* (((-beta1[i]*dr[i]-(beta0[i]+1.)*r[i]
                                           +r_inf[i]-betaC[i])/beta2[i])
                                         +((-beta1[i+1]*dr[i+1]-(beta0[i+1]+1.)*r[i+1]
                                            +r_inf[i+1]-betaC[i+1])/beta2[i+1]))
                r[i+1] = r[i]  + dt/2.* (dr[i]+dr[i+1])
                
                wm[i+1] = wm[i]+ dt_tauw/2. * ((a[i]*(Vm_inf[i]-Ew)- wm[i] + b[i] * tauw * r[i])
                                               +(a[i+1]*(Vm_inf[i+1]-Ew)- wm[i+1] + b[i+1] * tauw * r[i+1]))
                                               
        # Rectification:
        # set variables to zero if they would be integrated below 0
        if rectify and r[i+1]<0:
            r[i+1] = 0.
            dr[i+1] = 0.
    
    # return results
    return (r*1000, wm)



def run_spec2(ext_signal, params, filename_h5,
               rec_vars=['wm'], rec=False,
               FS=False, filter_input=False):
    if FS:
        raise NotImplementedError('FS-effects not implemented for spectral 2m model!')

    print('==================== integrating spec2-model ====================')

    # runtime parameters
    dt = params['uni_dt']
    runtime = params['runtime']
    steps = int(runtime/dt)
    t = np.linspace(0., runtime, steps+1)

    # external signal
    mu_ext = ext_signal[0]
    sig_ext = ext_signal[1]

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
    have_adapt = True if (a.any() or b.any()) else False

    # coupling parameters
    K = params['K']
    J = params['J']
    taud = params['taud']
    delay_type=int(params['delay_type'])
    const_delay = params['const_delay']
    # time integration method
    uni_int_order = params['uni_int_order']
    # booleans for rectification & grid-warning
    rectify = params['rectify_spec_models']
    grid_warn = params['grid_warn']

    # membrane capacitance
    C = params['C']

    # load quantities from hdf5-file
    h5file = tables.open_file(filename_h5, mode='r')
    mu_tab          = h5file.root.mu.read()
    sig_tab         = h5file.root.sigma.read()
    lambda_1        = h5file.root.lambda_1.read()
    lambda_2        = h5file.root.lambda_2.read()
    f1              = h5file.root.f_1.read()
    f2              = h5file.root.f_2.read()
    c_mu_1          = h5file.root.c_mu_1.read()
    c_mu_2          = h5file.root.c_mu_2.read()
    c_sigma_1       = h5file.root.c_sigma_1.read()
    c_sigma_2       = h5file.root.c_sigma_2.read()
    psi_r_1         = h5file.root.psi_r_1.read()
    psi_r_2         = h5file.root.psi_r_2.read()
    dr_inf_dmu      = h5file.root.dr_inf_dmu.read()
    dr_inf_dsigma   = h5file.root.dr_inf_dsigma.read()
    dVm_dmu         = h5file.root.dV_mean_inf_dmu.read()
    dVm_dsig        = h5file.root.dV_mean_inf_dsigma.read()
    V_mean_inf      = h5file.root.V_mean_inf.read()
    r_inf           = h5file.root.r_inf.read()
    h5file.close()



    # filter the input (typically done before calling here, thus disabled)
    # NOTE: this should result in a smooth input which
    # should be differentiable twice; maybe this function
    # will be transfered out of this function
    filter_input = False # from now on assume calling with smooth input
    if filter_input:
        # filter input moments
        print('filtering inputs...'); start=time.time()
        mu_ext = x_filter(mu_ext,params)
        sig_ext = x_filter(sig_ext,params)
        print('filtering inputs done in {} s.'
              .format(time.time()-start))

    # compute 1st/2nd derivatives of input variance (not std!)
    sig_ext2 = sig_ext**2
    dsig_ext2_dt = np.diff(sig_ext2)/dt
    dsig_ext2_dt = np.insert(dsig_ext2_dt, 0, dsig_ext2_dt[-1]) # 1st order backwards differences
    d2sig_ext2_dt2 = np.diff(dsig_ext2_dt)/dt #
    d2sig_ext2_dt2 = np.append(d2sig_ext2_dt2, d2sig_ext2_dt2[-1]) # 2nd order central diffs

    # compute 1st/2nd derivatives of input mean
    dmu_ext_dt = np.diff(mu_ext)/dt
    dmu_ext_dt = np.insert(dmu_ext_dt, 0, dmu_ext_dt[-1]) # 1st order backwards differences
    d2mu_ext_dt2 = np.diff(dmu_ext_dt)/dt
    d2mu_ext_dt2 = np.append(d2mu_ext_dt2, d2mu_ext_dt2[-1]) # 2nd order central diffs


    # precompute some matrices which are used in the integration loop
    # Notation adapted from the manuscript!
    # --> complex parts vanish as quantities are multiplicaitons of 
    # complex conjugated pairs (-> cf. manuscript)
    T =                (lambda_1**-1 + lambda_2**-1).real
    D =                    ((lambda_1 * lambda_2)**-1).real
    M =       (dr_inf_dmu + (f1*c_mu_1 + f2*c_mu_2)).real
    F_mu =   (lambda_1*f1*c_mu_1+lambda_2*f2*c_mu_2).real

    # only for debugging
    fC_mu = (f1*c_mu_1 + f2*c_mu_2).real
    fC_sig = (f1*c_sigma_1 + f2*c_sigma_2).real

    # quantities as derivatives with respect to sig NOT sig2
    # later on the get scaled by a factor of 1/(2*sig_tot)
    #  wihtin the integration loop, compare derivation
    # of the equations
    # --> eliminate complex parts!
    F_sig = (lambda_1*f1*c_sigma_1 + lambda_2*f2*c_sigma_2).real
    S_sig = (dr_inf_dsigma + (f1*c_sigma_1 + f2*c_sigma_2)).real


    # initial values
    wm0 = params['wm_init'] if have_adapt else 0.
    
    if params['fp_v_init'] == 'delta':
        # assuming vanishing ext. input variation, recurrency and and adaptation in the beginning
        mu_init = mu_ext[0]
        sigma_init = sig_ext[0]
        psi_r_1_interp = scipy.interpolate.RectBivariateSpline(mu_tab, sig_tab, 
                                                               psi_r_1, kx=1, ky=1)
        psi_r_2_interp = scipy.interpolate.RectBivariateSpline(mu_tab, sig_tab, 
                                                               psi_r_2, kx=1, ky=1)
#        psi_r_1_init = psi_r_1_interp()...
                                                               
        # calculating projection coefficients
        a1_0 = psi_r_1
        
        # initial values
    else:
        r0 = 0.
        dr0 = 0.


    results = sim_spec2(mu_ext,dmu_ext_dt,d2mu_ext_dt2,
                        sig_ext, dsig_ext2_dt,d2sig_ext2_dt2,steps,
                        t,dt, mu_tab, sig_tab,rec,delay_type, const_delay,
                        K, J,taud,tauw,a,b,C,wm0, Ew,r_inf, V_mean_inf,T,
                        S_sig,D,M,F_mu,F_sig, dVm_dsig,dVm_dmu,dr_inf_dmu,
                        dr_inf_dsigma,fC_mu, fC_sig, uni_int_order, rectify,
                        grid_warn)

    results_dict = dict()
    results_dict['t'] = t
    results_dict['r'] = results[0]
    if 'wm' in rec_vars: results_dict['wm'] = results[1]    
    
    # return results 
    return results_dict


