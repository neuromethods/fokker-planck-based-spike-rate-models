from __future__ import print_function
import sys

# for computing the time derivatives of the external moments
from scipy.misc import derivative
from matplotlib.pyplot import *
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

import matplotlib.pyplot as plt

print('WARNING: njit seems to be off?')

# CURRENT VERSION OF s2m model (without STACKING) with possinility of HEUN time integraiton method
#################################################################################################
# @njit
def sim_spec2m(mu_ext, dmu_ext_dt, d2mu_ext_dt2, sig_ext,
               dsig_ext2_dt, d2sig_ext2_dt2, steps, t, dt, mu_range,
               sig_range, rec, delay_type, const_delay, K, J, taud, tauw, a, b, C,wm0, Ew,
               r_inf_map, Vm_inf_map, T_map, S_sig_map, D_map, M_map, Fmu_map,
               Fsig_map, dVm_dsig_map, dVm_dmu_map, dr_inf_dmu_map,
               dr_inf_dsig_map, fC_mu_map,fC_sig_map, uni_int_order, rectify):

    # TODO : Komentar sigma_ext --> sigma_ext2 CAUTION!!!!
    # NOTE; the external input sigma_ext is used as sig_ext**2 within the integration loop.
    # Keep that in mind when if tempted to set the second derivative of it to zero for debugging.


    feedforward = not rec

    # state variables
    r  = np.zeros_like(t)
    r_d = np.zeros(len(t)+1)
    # r_rec = np.zeros_like(t)

    dr = np.zeros_like(t)
    wm = np.zeros_like(t)
    # set initial value of the mean adaptation current
    wm[0]=wm0

    # total moments
    mu_tot = np.zeros_like(t)
    sig_tot= np.zeros_like(t)
    mu_syn = np.zeros_like(t)

    # debug quantities
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

    # for delay_type == 1
    ddr = np.zeros_like(t)
    # for delay_type == 3
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
    



    # if not recurrent r_d =0 --> gets overwritten in recurrent case
    # todo make
    # r_d = 0.
    dt_taud = dt/taud
    dt_tauw = dt/tauw
    # ensure at least one time step
    ndt = max(1, int(const_delay/dt))



    #todo| start with feedforward case and then extend the system
    #todo| to the recurrent settings!
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
                # exp distributed delays
                elif delay_type == 2:
                    r_d[i+j+1] = r_d[i+j]+dt_taud*(r[i+j]-r_d[i+j])
                # exp. distributed delays + const. delay
                elif delay_type == 3:
                    r_d[i+j+1] = r_d[i+j]+(dt/taud)*(r[i+j-ndt]-r_d[i+j])
                    r_d_dot[i+j] = (1./taud)*(r[i+j-ndt]-r_d[i+j])
                    r_d_dotdot[i+j] = (1./taud)*(dr[i+j]-(r[i+j-ndt] - r_d[i+j])/taud)
                # no other delay_type is supported
                else:
                    raise NotImplementedError

                mu_syn[i+j] = mu_ext[i+j] + K*J*r_d[i+j]
                mu_tot[i+j] = mu_syn[i+j] - wm[i+j]/C
                sig_tot[i+j] = np.sqrt(sig_ext[i+j]**2 + K*J**2*r_d[i+j])

            # this case corresponds to an UNCOUPLED POPULATION of neurons, i.e. K=0.
            if feedforward:
                # if feedforward setting
                mu_tot[i+j] = mu_ext[i+j] - wm[i+j]/C
                sig_tot[i+j] = sig_ext[i+j]




            # get weights
            weights = interpolate_xy(mu_tot[i+j], sig_tot[i+j], mu_range, sig_range)
            # all lookups
            ###
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

            # debug
            fCsig2[i+j] = fCsig/(2*sig_tot[i+j])
            dr_dsig2[i+j] = dr_dsig/(2*sig_tot[i+j])


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
                # define some abbreviations
                # todo check if some of them are general are can
                # therefore be computed also for feedforward case
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
                                  + inv_taud2*R[i+j]
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


                    # betaC[i+j] = 0.


                # no delays
                # if delay_type == 0:
                #     beta0 = -D*M*b_tauwC-b_C*(T*M + D*M*a_tauwC*dVm_dmu-D*Fmu) #check

                #     beta1 = -T+D*M*b_C+K*J*(T*M+D*M*a_tauwC*dVm_dmu-D*Fmu)\
                #             +K*J**2*(T*S_sig2+D*M*a_tauwC*dVm_dsig_2-D*Fsig2)

                #     beta2 = D - D*M*K*J -D*S_sig2*K*J**2
                #     betaC = -D*M*(d2mu_ext_dt2[i]+A/tauw) - D*S_sig2*d2sig_ext2_dt2[i] + \
                #             (dmu_ext_dt[i]-A)*(T*M+D*M*a_tauwC*dVm_dmu-D*Fmu)+dsig_ext2_dt[i] * \
                #             (T*S_sig2+D*M*a_tauwC*dVm_dsig_2-D*Fsig2)
                #     betaD = 0
                # # exp distributed delays
                # elif delay_type == 2:
                #     beta0 = -D*M*b_tauwC-b_C*(T*M+D*M*a_tauwC*dVm_dmu-D*Fmu)+\
                #             inv_taud*(K*J*(T*M+D*M*a_tauwC*dVm_dmu-D*Fmu)+
                #                       K*J**2*(T*S_sig2+D*M*a_tauwC*dVm_dsig_2-D*Fsig2))-\
                #             inv_taud2*(-D*M*K*J-D*S_sig2*K*J**2)
                #     beta1 = -T + D*M*b_C+inv_taud*(-D*M*K*J-D*S_sig2*K*J**2)
                #     beta2 = D
                #     betaC = -D*M*(d2mu_ext_dt2[i]+A/tauw) - D*S_sig2*d2sig_ext2_dt2[i] + \
                #             (dmu_ext_dt[i]-A)*(T*M+D*M*a_tauwC*dVm_dmu-D*Fmu)+dsig_ext2_dt[i] * \
                #             (T*S_sig2+D*M*a_tauwC*dVm_dsig_2-D*Fsig2)
                #     betaD = -inv_taud*(K*J*(T*M+D*M*a_tauwC*dVm_dmu-D*Fmu)+
                #                        K*J**2*(T*S_sig2+D*M*a_tauwC*dVm_dsig_2-D*Fsig2))-\
                #             inv_taud2*(-D*M*K*J-D*S_sig2*K*J**2)


            # this corresponds to the case of an UNCOUPLED POPULATION of neurons, i.e. K=0.
            else:
                beta2[i+j] =  D[i+j]
                beta1[i+j] = -T[i+j] + D[i+j]*M[i+j]*b_C
                beta0[i+j] = -D[i+j]*M[i+j]*b_tauwC-b_C*H_mu[i+j]

                betaC[i+j] = -D[i+j]*M[i+j]*(d2mu_ext_dt2[i+j] + (a[i+j] * (Vm_inf[i+j]-Ew)-wm[i+j])/(tauw**2*C))\
                             -D[i+j]*Ssig2[i+j]*d2sig_ext2_dt2[i+j]\
                             +(dmu_ext_dt[i+j] - (a[i+j] * (Vm_inf[i+j]-Ew)-wm[i+j])/(tauw*C))*H_mu[i+j]\
                             +dsig_ext2_dt[i+j]*H_sigma2[i+j]



                # beta0[i+j] = -D[i+j]*M[i+j]*(b[i+j]/(tauw*C))-(b[i+j]/C)*(T[i+j]*M[i+j] + D[i+j]*M[i+j]*
                #                                                         (a[i+j]/(tauw*C))
                #                                                         *dVm_dmu[i+j]-D[i+j]*Fmu[i+j]) #check
                # beta1[i+j] = -T[i+j] + D[i+j]*M[i+j]*b[i+j]/C #check
                # beta2[i+j] =  D[i+j] # check
                # betaC[i+j] = -D[i+j]*M[i+j]*(d2mu_ext_dt2[i+j]+A[i+j]/tauw) \
                #              - D[i+j]*Ssig2[i+j]*d2sig_ext2_dt2[i+j] + (dmu_ext_dt[i+j]-A[i+j])\
                #                                                        *(T[i+j]*M[i+j]+D[i+j]*M[i+j]
                #                                                          *(a[i+j]/(tauw*C))
                #                                                          *dVm_dmu[i+j]-D[i+j]*Fmu[i+j])\
                #            +dsig_ext2_dt[i+j] \
                #             *(T[i+j]*Ssig2[i+j]+D[i+j]*M[i+j]
                #               *(a[i+j]/(tauw*C))*dVm_dsig2[i+j]-D[i+j]*Fsig2[i+j])
                # betaD[i] = 0. # for the feedforward case



            # implemenet the different delay scenarios
            # integration step
            # def. dr:= dr/dt yields coupled 1st order system + adaptation
            if j == 0:
                # todo: re-include betaC
                dr[i+1]= dr[i] + dt* (-beta1[i]*dr[i]-(beta0[i]+1.)*r[i]+r_inf[i]-betaC[i])/beta2[i] #-betaC[i]
                r[i+1] = r[i]  + dt* dr[i]
                wm[i+1] = wm[i]+ dt_tauw * (a[i]*(Vm_inf[i]-Ew)- wm[i] + b[i] * tauw * r[i])
                # 2nd derivative of r
                ddr[i] = (dr[i] - dr[i-1])/dt # for Euler would be equivalent to next line but less redundant
#                ddr[i] = (-beta1[i]*dr[i]-(beta0[i]+1.)*r[i]+r_inf[i]-betaC[i])/beta2[i]

            elif j == 1:
                dr[i+1]= dr[i] + dt/2.* (((-beta1[i]*dr[i]-(beta0[i]+1.)*r[i]
                                           +r_inf[i]-betaC[i])/beta2[i])
                                         +((-beta1[i+1]*dr[i+1]-(beta0[i+1]+1.)*r[i+1]
                                            +r_inf[i+1]-betaC[i+1])/beta2[i+1]))
                r[i+1] = r[i]  + dt/2.* (dr[i]+dr[i+1])
                # TODO: add rectify of r via toggle (def: on)
                wm[i+1] = wm[i]+ dt_tauw/2. * ((a[i]*(Vm_inf[i]-Ew)- wm[i] + b[i] * tauw * r[i])
                                               +(a[i+1]*(Vm_inf[i+1]-Ew)- wm[i+1] + b[i+1] * tauw * r[i+1]))
                                               

        # set variables to zero if they would be integrated below 0
        if rectify and r[i+1]<0:
            r[i+1] = 0.
            dr[i+1] = 0.
#        print('after step i={}'.format(i))
#        print('mu_tot={}'.format(mu_tot[i]))
#        print('sig_tot={}'.format(sig_tot[i]))
#        print('beta2={}'.format(beta2))
#        print('r_inf={}'.format(r_inf))
#        print('Vm_inf={}'.format(Vm_inf))print('weights='.format(weights))
#        print('Dmin/max={},{}'.format(D_map.min(),D_map.max()))
#        print('D={}'.format(D))
#        print('wm[i+1]={}'.format(wm[i+1]))
#        print('r[i+1]={}'.format(r[i+1]))

#     plt.plot(r, label='r')
#     plt.plot(dr, label='dr')
#     plt.plot(ddr, label='ddr')
#     plt.plot(betaC, label='betaC')
#     plt.legend()
#     plt.figure()
#     plt.plot(dmu_ext_dt, label='dmu_ext')
#     plt.legend()
#     plt.show()


    return (r*1000, wm, D, T, M, Ssig2, TM, DM, DSsig2, TSsig2,
            Fmu, DFmu, A, Fsig2,DFsig2, fCmu, fCsig2, dr_dmu,
            dr_dsig2, mu_tot, Vm_inf, r_inf, betaC, sig_tot)





def run_spec2m(ext_signal, params, filename_h5,
               set_1st_derivs_0=False,
               set_2nd_derivs_0=False,C_terms_OFF=False,
               dr_inf_dmu_OFF=False,rec_vars=['wm'], rec=False,
               FS=False, filter_input=False):
    if FS:
        raise NotImplementedError('FS-effects not implemented for spectral 2m model!')

    print('running spec2m')

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
    # boolean for rectification
    rectify = params['rectify_spec_models']

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

#    plt.figure()
#    plt.subplot(3,2,1)
#    plt.title('mu_ext')
#    plt.plot(t, mu_ext)
#    plt.subplot(3,2,2)
#    plt.title('sig_ext2')
#    plt.plot(t, sig_ext2)
#    plt.subplot(3,2,3)
#    plt.title('dmu_ext_dt')
#    plt.plot(t, dmu_ext_dt)
#    plt.subplot(3,2,4)
#    plt.title('dsig_ext2_dt')
#    plt.plot(t, dsig_ext2_dt)
#    plt.subplot(3,2,5)
#    plt.title('d2mu_ext_dt2')
#    plt.plot(t, d2mu_ext_dt2)
#    plt.subplot(3,2,6)
#    plt.title('d2sig_ext2_dt2')
#    plt.plot(t, d2sig_ext2_dt2)


    # options to set certain input values to 0 for debugging
    if set_1st_derivs_0:
        #set first derivs to zero
        dmu_ext_dt = np.zeros_like(sig_ext)
        dsig_ext2_dt = np.zeros_like(sig_ext)
    if set_2nd_derivs_0:
        d2mu_ext_dt2 = np.zeros_like(mu_ext)
        d2sig_ext2_dt2=np.zeros_like(mu_ext)
    if C_terms_OFF:
        c_mu_1 = np.zeros_like(c_mu_1)
        c_mu_2 = np.zeros_like(c_mu_2)
        print('C-TERMS ARE SWITCHED OFF!')
    else:
        print('C-TERMS ENTER!')
    if dr_inf_dmu_OFF:
        dr_inf_dmu=np.zeros_like(dr_inf_dmu)

    # precompute some matrices which are used in the loop
    # Notation adapted from the manuscript!
    # --> eliminate complex parts as quantities are real (theory).
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


    results = sim_spec2m(mu_ext,dmu_ext_dt,d2mu_ext_dt2,
                         sig_ext, dsig_ext2_dt,d2sig_ext2_dt2,steps,t,dt,
                         mu_tab,sig_tab,rec,delay_type, const_delay,
                         K, J,taud,tauw,a,b,C,wm0,
                         Ew,r_inf, V_mean_inf,T,S_sig,D,M,F_mu,F_sig,
                         dVm_dsig,dVm_dmu,dr_inf_dmu,dr_inf_dsigma,fC_mu,
                         fC_sig, uni_int_order, rectify)

    results_dict = dict()
    results_dict['t'] = t
    results_dict['r'] = results[0]
    if 'wm' in rec_vars: results_dict['wm'] = results[1]
    results_dict['D']       =  results[2]
    results_dict['T']       =  results[3]
    results_dict['M']       =  results[4]
    results_dict['Ssig2']   =  results[5]
    results_dict['TM']      =  results[6]
    results_dict['DM']      =  results[7]
    results_dict['DSsig2']  =  results[8]
    results_dict['TSsig2']  =  results[9]
    results_dict['Fmu']     = results[10]
    results_dict['DFmu']    = results[11]
    results_dict['A']       = results[12]
    results_dict['Fsig2']   = results[13]
    results_dict['DFsig2']  = results[14]
    results_dict['fCmu']    = results[15]
    results_dict['fCsig2']  = results[16]
    results_dict['dr_dmu']  = results[17]
    results_dict['dr_dsig2']= results[18]
    results_dict['mu_total']= results[19]
    results_dict['Vm_inf']= results[20]
    results_dict['r_inf']= results[21]
    results_dict['betaC']= results[22]
    results_dict['sigma_total']= results[23]
    return results_dict


# new version of the s2m model. much faster then the old one, BUT wihtout HEUN integration method
#################################################################################################
@njit
def sim_spec2m_without_HEUN_OLD(mu_ext, dmu_ext_dt, d2mu_ext_dt2, sig_ext,
               dsig_ext2_dt, d2sig_ext2_dt2, steps, t, dt, mu_range,
               sig_range, rec, delay_type, K,J,taud, tauw, a, b, C,wm0, Ew,
               r_inf_map, Vm_inf_map, T_map, S_sig_map, D_map, M_map, Fmu_map,
               Fsig_map, dVm_dsig_map, dVm_dmu_map, dr_inf_dmu_map,
               dr_inf_dsig_map, fC_mu_map,fC_sig_map):

    # TODO : Komentar sigma_ext --> sigma_ext2 CAUTION!!!!
    # NOTE; the external input sigma_ext is used as sig_ext**2 within the integration loop.
    # Keep that in mind when if tempted to set the second derivative of it to zero for debugging.


    feedforward = not rec

    # for the moment assert feedforward case!
    assert feedforward

    # state variables
    r  = np.zeros_like(t)
    r_rec = np.zeros_like(t)
    dr = np.zeros_like(t)
    wm = np.zeros_like(t)
    # set initial value of the mean adaptation current
    wm[0]=wm0

    # total moments
    mu_tot = np.zeros_like(t)
    sig_tot= np.zeros_like(t)
    mu_syn = np.zeros_like(t)

    # debug quantities
    D_save = np.zeros_like(t)
    T_save = np.zeros_like(t)
    M_save = np.zeros_like(t)
    Ssig2_save = np.zeros_like(t)
    TM_save = np.zeros_like(t)
    DM_save = np.zeros_like(t)
    DSsig2_save = np.zeros_like(t)
    TSsig2_save = np.zeros_like(t)
    Fmu_save = np.zeros_like(t)
    DFmu_save = np.zeros_like(t)
    A_save = np.zeros_like(t)
    Fsig2_save = np.zeros_like(t)
    DFsig2_save = np.zeros_like(t)
    fCmu_save = np.zeros_like(t)
    fCsig2_save = np.zeros_like(t)
    dr_dmu_save = np.zeros_like(t)
    dr_dsig2_save = np.zeros_like(t)



    # if not recurrent r_d =0 --> gets overwritten in recurrent case
    r_d = 0.
    dt_taud = dt/taud



    #todo| start with feedforward case and then extend the system
    #todo| to the recurrent settings!
    # integration loop
    for i in xrange(steps):

        if feedforward:
            # if feedforward setting
            mu_tot[i] = mu_ext[i] - wm[i]/C
            sig_tot[i] = sig_ext[i]

        # else => rec
        else:
            # no delay
            if delay_type == 0:
                r_d = r[i]
            # exp distributed delays
            elif delay_type == 2:
                r_rec[i+1] = r_rec[i]+dt_taud*(r[i]-r_rec[i])
                r_d = r_rec[i]
            else:
                raise NotImplementedError

            mu_syn[i] = mu_ext[i] + K*J*r_d
            mu_tot[i] = mu_syn[i] - wm[i]/C
            sig_tot[i] = np.sqrt(sig_ext[i]**2 + K*J**2*r_d)

        # get weights
        weights = interpolate_xy(mu_tot[i], sig_tot[i], mu_range, sig_range)
        # all lookups
        ###
        M = lookup_xy(M_map, weights)
        S_sig = lookup_xy(S_sig_map, weights)
        D = lookup_xy(D_map, weights)
        T = lookup_xy(T_map, weights)
        dVm_dmu = lookup_xy(dVm_dmu_map, weights)
        dVm_dsig = lookup_xy(dVm_dsig_map, weights)
        Fmu = lookup_xy(Fmu_map, weights)
        Fsig = lookup_xy(Fsig_map, weights)
        Vm_inf = lookup_xy(Vm_inf_map, weights)
        r_inf = lookup_xy(r_inf_map, weights)

        # debug
        fC_mu = lookup_xy(fC_mu_map, weights)
        fC_sig = lookup_xy(fC_sig_map, weights)
        dr_dmu = lookup_xy(dr_inf_dmu_map, weights)
        dr_dsig = lookup_xy(dr_inf_dsig_map, weights)



        # after the look up scale the quantities depending on sig by 2*sig_tot
        dVm_dsig_2 = dVm_dsig/(2*sig_tot[i])
        Fsig2 = Fsig/(2*sig_tot[i])
        S_sig2 = S_sig/(2*sig_tot[i])

        # debug
        fC_sig2 = fC_sig/(2*sig_tot[i])
        dr_dsig2 = dr_dsig/(2*sig_tot[i])


        # A is the same for all settings
        A = (a[i]*(Vm_inf-Ew)-wm[i])/(tauw*C)



        # save debug quantities in each time step
        D_save[i]               = D
        T_save[i]               = T
        M_save[i]               = M
        Ssig2_save[i]           = S_sig2
        TM_save[i]              = T*M
        DM_save[i]              = D*M
        DSsig2_save[i]          = D*S_sig2
        TSsig2_save[i]          = T*S_sig2
        Fmu_save[i]             = Fmu
        DFmu_save[i]            = D*Fmu
        A_save[i]               = A
        Fsig2_save[i]           = Fsig2
        DFsig2_save[i]          = D*Fsig2
        fCmu_save[i]            = fC_mu
        fCsig2_save[i]          = fC_sig2
        dr_dmu_save[i]          = dr_dmu
        dr_dsig2_save[i]        = dr_dsig2



        a_tauwC = a[i]/(tauw*C)
        b_tauwC = b[i]/(tauw*C)
        b_C = b[i]/C
        inv_taud = 1./taud
        inv_taud2 = 1./taud**2
        # compute the scalar beta values after the lookup
        if feedforward:
            beta0 = -D*M*b_tauwC-b_C*(T*M + D*M*a_tauwC*dVm_dmu-D*Fmu)
            beta1 = -T + D*M*b_C
            beta2 =  D
            betaC = -D*M*(d2mu_ext_dt2[i]+A/tauw) - D*S_sig2*d2sig_ext2_dt2[i] + \
                    (dmu_ext_dt[i]-A)*(T*M+D*M*a_tauwC*dVm_dmu-D*Fmu)+dsig_ext2_dt[i] * \
                    (T*S_sig2+D*M*a_tauwC*dVm_dsig_2-D*Fsig2)
            betaD = 0. # for the feedforward case

        else:
            # no delays
            if delay_type == 0:
                beta0 = -D*M*b_tauwC-b_C*(T*M + D*M*a_tauwC*dVm_dmu-D*Fmu)
                beta1 = -T+D*M*b_C+K*J*(T*M+D*M*a_tauwC*dVm_dmu-D*Fmu)\
                        +K*J**2*(T*S_sig2+D*M*a_tauwC*dVm_dsig_2-D*Fsig2)
                beta2 = D - D*M*K*J -D*S_sig2*K*J**2
                betaC = -D*M*(d2mu_ext_dt2[i]+A/tauw) - D*S_sig2*d2sig_ext2_dt2[i] + \
                        (dmu_ext_dt[i]-A)*(T*M+D*M*a_tauwC*dVm_dmu-D*Fmu)+dsig_ext2_dt[i] * \
                        (T*S_sig2+D*M*a_tauwC*dVm_dsig_2-D*Fsig2)
                betaD = 0
            # exp distributed delays
            elif delay_type == 2:

                beta0 = -D*M*b_tauwC-b_C*(T*M+D*M*a_tauwC*dVm_dmu-D*Fmu)+\
                        inv_taud*(K*J*(T*M+D*M*a_tauwC*dVm_dmu-D*Fmu)+
                                  K*J**2*(T*S_sig2+D*M*a_tauwC*dVm_dsig_2-D*Fsig2))-\
                        inv_taud2*(-D*M*K*J-D*S_sig2*K*J**2)
                beta1 = -T + D*M*b_C+inv_taud*(-D*M*K*J-D*S_sig2*K*J**2)
                beta2 = D
                betaC = -D*M*(d2mu_ext_dt2[i]+A/tauw) - D*S_sig2*d2sig_ext2_dt2[i] + \
                        (dmu_ext_dt[i]-A)*(T*M+D*M*a_tauwC*dVm_dmu-D*Fmu)+dsig_ext2_dt[i] * \
                        (T*S_sig2+D*M*a_tauwC*dVm_dsig_2-D*Fsig2)
                betaD = -inv_taud*(K*J*(T*M+D*M*a_tauwC*dVm_dmu-D*Fmu)+
                                   K*J**2*(T*S_sig2+D*M*a_tauwC*dVm_dsig_2-D*Fsig2))-\
                        inv_taud2*(-D*M*K*J-D*S_sig2*K*J**2)

        # implemenet the different delay scenarios
        # integration step
        # def. dr:= dr/dt yields coupled 1st order system + adaptation        
        dr[i+1]= dr[i] + dt* (-beta1*dr[i]-(beta0+1.)*r[i]+r_inf-betaC+ betaD*r_d )/beta2
        r[i+1] = r[i]  + dt* dr[i]
        # TODO: add rectify of r via toggle (def: on)
        wm[i+1] = wm[i]+ (dt/tauw) * (a[i]*(Vm_inf-Ew)- wm[i] + b[i] * tauw * r[i])
        

#        print('after step i={}'.format(i))
#        print('mu_tot={}'.format(mu_tot[i]))
#        print('sig_tot={}'.format(sig_tot[i]))
#        print('beta2={}'.format(beta2))
#        print('r_inf={}'.format(r_inf))
#        print('Vm_inf={}'.format(Vm_inf))print('weights='.format(weights))
#        print('Dmin/max={},{}'.format(D_map.min(),D_map.max()))
#        print('D={}'.format(D))                
#        print('wm[i+1]={}'.format(wm[i+1]))
#        print('r[i+1]={}'.format(r[i+1]))

    return (r*1000, wm, D_save, T_save, M_save, Ssig2_save, TM_save, DM_save, DSsig2_save,
            TSsig2_save, Fmu_save, DFmu_save, A_save, Fsig2_save,DFsig2_save, fCmu_save,
            fCsig2_save, dr_dmu_save, dr_dsig2_save)





def run_spec2m_without_HEUN_OLD(ext_signal, params, filename_h5,
               set_1st_derivs_0=False,
               set_2nd_derivs_0=False,C_terms_OFF=False,
               dr_inf_dmu_OFF=False,rec_vars=['wm'], rec=False,
               FS=False, filter_input=False):
    if FS:
        raise NotImplementedError('FS-effects not implemented for spectral 2m model!')

    print('running spec2m')

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
    # currently only two different delay_types supported
    if rec: assert delay_type in (0,2)

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

#    plt.figure()
#    plt.subplot(3,2,1)
#    plt.title('mu_ext')
#    plt.plot(t, mu_ext)
#    plt.subplot(3,2,2)
#    plt.title('sig_ext2')
#    plt.plot(t, sig_ext2)
#    plt.subplot(3,2,3)
#    plt.title('dmu_ext_dt')
#    plt.plot(t, dmu_ext_dt)
#    plt.subplot(3,2,4)
#    plt.title('dsig_ext2_dt')
#    plt.plot(t, dsig_ext2_dt)
#    plt.subplot(3,2,5)
#    plt.title('d2mu_ext_dt2')
#    plt.plot(t, d2mu_ext_dt2)
#    plt.subplot(3,2,6)
#    plt.title('d2sig_ext2_dt2')
#    plt.plot(t, d2sig_ext2_dt2)


    # options to set certain input values to 0 for debugging
    if set_1st_derivs_0:
        #set first derivs to zero
        dmu_ext_dt = np.zeros_like(sig_ext)
        dsig_ext2_dt = np.zeros_like(sig_ext)
    if set_2nd_derivs_0:
        d2mu_ext_dt2 = np.zeros_like(mu_ext)
        d2sig_ext2_dt2=np.zeros_like(mu_ext)
    if C_terms_OFF:
        c_mu_1 = np.zeros_like(c_mu_1)
        c_mu_2 = np.zeros_like(c_mu_2)
        print('C-TERMS ARE SWITCHED OFF!')
    else:
        print('C-TERMS ENTER!')
    if dr_inf_dmu_OFF:
        dr_inf_dmu=np.zeros_like(dr_inf_dmu)

    # precompute some matrices which are used in the loop
    # Notation adapted from the manuscript!
    # --> eliminate complex parts as quantities are real (theory).
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

    results = sim_spec2m_without_HEUN_OLD(mu_ext,dmu_ext_dt,d2mu_ext_dt2,
                         sig_ext, dsig_ext2_dt,d2sig_ext2_dt2,steps,t,dt,
                         mu_tab,sig_tab,rec,delay_type, K, J,taud,tauw,a,b,C,wm0,
                         Ew,r_inf, V_mean_inf,T,S_sig,D,M,F_mu,F_sig,
                         dVm_dsig,dVm_dmu,dr_inf_dmu,dr_inf_dsigma,fC_mu,
                         fC_sig)

    results_dict = dict()
    results_dict['t'] = t
    results_dict['r'] = results[0]
    if 'wm' in rec_vars: results_dict['wm'] = results[1]
    results_dict['D']       =  results[2]
    results_dict['T']       =  results[3]
    results_dict['M']       =  results[4]
    results_dict['Ssig2']   =  results[5]
    results_dict['TM']      =  results[6]
    results_dict['DM']      =  results[7]
    results_dict['DSsig2']  =  results[8]
    results_dict['TSsig2']  =  results[9]
    results_dict['Fmu']     = results[10]
    results_dict['DFmu']    = results[11]
    results_dict['A']       = results[12]
    results_dict['Fsig2']   = results[13]
    results_dict['DFsig2']  = results[14]
    results_dict['fCmu']    = results[15]
    results_dict['fCsig2']  = results[16]
    results_dict['dr_dmu']  = results[17]
    results_dict['dr_dsig2']= results[18]
    return results_dict
#################################################################################################




# NOTE:
# old version of the s2m model. Do not use! very slow...
# was optimized for speed: use run_spec2m!
#################################################################################################
@njit
def sim_spec2m_OLD_STACKING(mu_ext, dmu_ext_dt, d2mu_ext_dt2, sig_ext,
               dsig_ext2_dt, d2sig_ext2_dt2, steps, t, dt,
               mu_range, sig_range, rec, delay_type,K,J,
               taud, tauw, a, b, C, Ew,beta0_map, beta1_map,
               beta2_map, betaD_map, r_inf_map,
               Vm_inf_map, T, S_sig, D, M,Fmu, F_sig,dVm_dsig,
               dVm_dmu, fC_mu,dr_inf_dmu,fC_sig,dr_inf_dsigma):

    # initialize arrays
    dr = np.zeros_like(t)
    r = np.zeros_like(t)
    wm = np.zeros_like(t)
    Vmean = np.zeros_like(t)
    r_rec = np.zeros_like(t)
    mu_syn = np.zeros_like(t)
    mu_tot = np.zeros_like(t)
    sig_tot = np.zeros_like(t)
    # record for debugging
    # both cases
    D_array=np.zeros_like(t)
    T_array=np.zeros_like(t)
    # if sig_ext=sig_ext(t)
    dr_inf_dsigma2_array=np.zeros_like(t)
    fC_sig2_array = np.zeros_like(t)
    DF_sig2_array=np.zeros_like(t)
    TS_array=np.zeros_like(t)
    DS_array = np.zeros_like(t)
    S_array = np.zeros_like(t)
    F_sig2_array=np.zeros_like(t)
    # if mu_ext=mu_ext(t)
    dr_inf_dmu_array = np.zeros_like(t)
    fC_mu_array = np.zeros_like(t)
    M_array=np.zeros_like(t)
    F_mu_array = np.zeros_like(t)
    TM_array = np.zeros_like(t)
    DM_array = np.zeros_like(t)
    DF_mu_array = np.zeros_like(t)


    # todo: optimize more after final debugg.
    # some optimizations
    inv_taud = 1./taud
    inv_taud2 = 1./taud**2
    dt_taud = dt/taud
    dt_tauw = dt/tauw
    b_tauw = b*tauw
    b_C = b/C
    tauwC = tauw*C
    a_tauwC = a/tauwC
    b_tauwC = b/tauwC
    TM = T*M
    DM = D*M
    KJ = K*J
    KJ2 =K*J**2
    DMKJ = DM*KJ
    DKJ2 = D*K*J**2
    # pass these arrays into the function
    # not possible in numba...
    #gradients
    # sig_ext2 = sig_ext**2
    # dmu_ext_dt = np.gradient(mu_ext, dt)
    # d2mu_ext_dt2 = np.gradient(dmu_ext_dt, dt)
    # dsig_ext2_dt = np.gradient(sig_ext2,dt)
    # d2sig_ext2_dt2 = np.gradient(dsig_ext2_dt, dt)

    # set r_d to zero, get overwritten if delayType is 1
    r_d = 0


    #inner loop
    for i in xrange(steps):
        # if feedforward setting
        mu_tot[i] = mu_ext[i] - wm[i]/C
        sig_tot[i] = sig_ext[i]
        # these quantities have to be scaled with sig_tot
        # which is sig_ext in the ff-case
        dVm_dsig_2 = dVm_dsig/(2*sig_tot[i])
        F_sig2 = F_sig/(2*sig_tot[i])
        S_sig2 = S_sig/(2*sig_tot[i])
        fC_sig2= fC_sig/(2*sig_tot[i])
        dr_inf_dsigma2 = dr_inf_dsigma/(2*sig_tot[i])
        beta0_map =  -D*M*(b[i]/(tauw*C))-(b[i]/C)*(T*M + D*M*(a[i]/(tauw*C))*dVm_dmu-D*Fmu)
        beta1_map =  -T + D*M*(b[i]/C)

        # if rec overwrite mu_tot, sig_tot
        if rec:
            # no delay
            if delay_type == 0:
                mu_syn[i] = mu_ext[i] + K*J*r[i]
                mu_tot[i] = mu_syn[i] - wm[i]/C
                sig_tot[i] = math.sqrt(sig_ext[i]**2 + K*J**2*r[i])

                # get derivative of quantities w.r.t. sig**2
                F_sig2 = F_sig/(2*sig_tot[i])
                S_sig2 = S_sig/(2*sig_tot[i])
                dVm_dsig_2 = dVm_dsig/(2*sig_tot[i])

                # calculate missing map/s: beta1_map
                beta1_map = -T+DM*b_C+KJ*(TM+DM*a_tauwC[i]*dVm_dmu-D*Fmu)+\
                             KJ2*(T*S_sig2+DM*a_tauwC[i]*dVm_dsig_2-D*F_sig2)
                beta2_map =  D-DMKJ-DKJ2*S_sig2
            # exp. distributed delay
            elif delay_type == 2:
                r_rec[i+1] = r[i]+dt_taud*(r[i]-r_rec[i])
                r_d = r_rec[i]
                mu_syn[i] = mu_ext[i] + K*J*r_d
                mu_tot[i] = mu_syn[i] - wm[i]/C
                sig_tot[i] = math.sqrt(sig_ext[i]**2 + K*J**2*r_d)

                # get derivative of quantities w.r.t. sig**2
                F_sig2 = F_sig/(2*sig_tot[i])
                S_sig2 = S_sig/(2*sig_tot[i])
                dVm_dsig_2 = dVm_dsig/(2*sig_tot[i])
                # calculate missing map/s: beta0_map, betaD_map
                beta0_map = -DM*b_tauwC-b_C*(TM+DM*a_tauwC[i]*dVm_dmu-D*Fmu)\
                            +inv_taud*(KJ*(TM+DM*a_tauwC[i]*dVm_dmu-D*Fmu)+KJ2*
                            (T*S_sig2+DM*a_tauwC[i]*dVm_dsig_2-D*F_sig2))-inv_taud2*\
                            (-DMKJ-DKJ2*S_sig2)
                beta1_map =  -T +DM*b_C+inv_taud*(-DMKJ-DKJ2*S_sig2)
                betaD_map = -inv_taud*(KJ*(TM+DM*a_tauwC[i]*dVm_dmu-D*Fmu)+
                            KJ2*(T*S_sig2+DM*a_tauwC[i]*dVm_dsig_2- D*F_sig2))\
                            +inv_taud2*(-DMKJ-DKJ2*S_sig2)
            else:
                raise NotImplementedError

        # compute the A map in every time step as
        # it depends on the mean adaptation current
        # todo check this here
        A = (a[i]*(Vm_inf_map-Ew)-wm[i])/(tauwC)
        betaC_map = -DM*(d2mu_ext_dt2[i]+A/tauw) - D*S_sig2*d2sig_ext2_dt2[i] + \
                    (dmu_ext_dt[i]-A)*(TM+DM*a_tauwC[i]*dVm_dmu-D*Fmu)+dsig_ext2_dt[i]*\
                    (T*S_sig2+DM*a_tauwC[i]*dVm_dsig_2-D*F_sig2)


        # get weights
        # todo|  implement scalar version of this such that
        # todo|  the beta maps are not stacked in each time step
        weights = interpolate_xy(mu_tot[i], sig_tot[i], mu_range, sig_range)
        # lookup
        ######################################
        # for debugging
        # debugging both cases
        D_array[i] = lookup_xy(D, weights)
        T_array[i] = lookup_xy(T, weights)
        # quantities for sig_ext(t)
        dr_inf_dsigma2_array[i] = lookup_xy(dr_inf_dsigma2, weights)
        fC_sig2_array[i]        = lookup_xy(fC_sig2,weights)
        DF_sig2_array[i]        = lookup_xy(D*F_sig2,weights)
        TS_array[i]             = lookup_xy(T*S_sig2,weights)
        DS_array[i]             = lookup_xy(D*S_sig2,weights)
        S_array[i]              = lookup_xy(S_sig2,weights)
        F_sig2_array[i]         = lookup_xy(F_sig2,weights)
        # quantities for mu_ext(t)
        dr_inf_dmu_array[i] = lookup_xy(dr_inf_dmu,weights)
        fC_mu_array[i]      = lookup_xy(fC_mu,weights)
        M_array[i]          = lookup_xy(M,weights)
        F_mu_array[i]       = lookup_xy(Fmu,weights)
        TM_array[i]         = lookup_xy(TM,weights)
        DM_array[i]         = lookup_xy(DM,weights)
        DF_mu_array[i]      = lookup_xy(D*Fmu,weights)

        # beta maps
        beta0 = lookup_xy(beta0_map, weights)
        beta1 = lookup_xy(beta1_map, weights)
        beta2 = lookup_xy(beta2_map, weights)
        beta_c = lookup_xy(betaC_map, weights)
        # in case of feedforward setting or delayType 0
        # betaD_map everywhere is zero; betaD is zero
        beta_d = lookup_xy(betaD_map, weights)
        r_inf_val = lookup_xy(r_inf_map, weights)
        Vmean_val = lookup_xy(Vm_inf_map, weights)


        # integration step
        # def. dr:= dr/dt yields coupled 1st order system + adaptation
#         dr[i+1]= dr[i] + dt* (-beta1*dr[i]-(beta0+1.)*r[i]+r_inf-betaC+ betaD*r_d )/beta2

        # r[i+1] = r[i]  + dt* dr[i]
#         r_prelim = r[i]  + dt* dr[i]
#        r[i+1] = r_prelim if r_prelim > 0. else 0.
#        wm[i+1] = wm[i]+ (dt/tauw) * (a[i]*(Vm_inf-Ew)- wm[i] + b[i] * tauw * r[i])

#    return (r*1000, wm, D_save, T_save, M_save, Ssig2_save, TM_save, DM_save, DSsig2_save,
#            TSsig2_save, Fmu_save, DFmu_save, A_save, Fsig2_save,DFsig2_save, fCmu_save,
#            fCsig2_save, dr_dmu_save, dr_dsig2_save)

        dr[i+1]= dr[i] + dt* (-beta1*dr[i]-(beta0+1.)*r[i]+r_inf_val-beta_c+ beta_d*r_d )/beta2
        r[i+1] = r[i]  + dt* dr[i]
        wm[i+1] = wm[i]+ dt_tauw * (a[i]*(Vmean_val-Ew)- wm[i] + b_tauw[i] * r[i])
    return (r*1000, wm, D_array, T_array, dr_inf_dsigma2_array,
            fC_sig2_array, DF_sig2_array, TS_array, DS_array,
            S_array, F_sig2_array, dr_inf_dmu_array,fC_mu_array,
            M_array, F_mu_array,TM_array, DM_array, DF_mu_array)



def run_spec2m_OLD_STACKING(ext_signal, params, set_1st_derivs_0=False,
               set_2nd_derivs_0=False,C_terms_OFF=False,
               dr_inf_dmu_OFF=False,rec_vars=['wm'], rec=False,
               FS=False, filter_input=False):
    if FS:
        raise NotImplementedError('FS-effects not implemented for spectral 2m model!')

    # print('running spec2m')
    print('running spec2m')

    mu_ext = ext_signal[0]
    sig_ext = ext_signal[1]


    # unpack params dict
    dt = params['uni_dt']
    runtime = params['runtime']
    J = params['J'] # for recurrency
    K = params['K'] # for recurrency
    taud = params['taud'] # for recurrency
    a = params['a']
    b = params['b']
    tauw = params['tauw']
    Ew = params['Ew']
    C = params['C']


    # how many integration steps
    steps = int(runtime/dt)

    # len(timearray) = steps+1
    t = np.linspace(0., runtime, steps+1)

    # rec
    delay_type = params['delay_type']

    if filter_input:
        # filter input moments
        # really slow...
        print('filtering inputs...'); start=time.time()
        mu_ext_filtered = x_filter(mu_ext,params)
        sigma_ext_filtered = x_filter(sig_ext,params)
        print('filtering inputs done in {} s.'
              .format(time.time()-start))

    # compute time derivatives of
    # the external input moments
    # which are then passed to the inner function

    # commented out version is old
    # sig_ext_filtered2 = sigma_ext_filtered**2 if filter_input else sig_ext**2
    # dmu_ext_dt = np.gradient(mu_ext_filtered, dt) if filter_input else np.gradient(mu_ext, dt)
    # d2mu_ext_dt2 = np.gradient(dmu_ext_dt, dt)
    # dsig_ext2_dt = np.gradient(sig_ext_filtered2,dt)
    # d2sig_ext2_dt2 = np.gradient(dsig_ext2_dt, dt)

    # new way for calculating the derivative with numpy diff !!
    sig_ext_filtered2 = sigma_ext_filtered**2 if filter_input else sig_ext**2
    dsig_ext2_dt = np.diff(sig_ext_filtered2)/dt
    dsig_ext2_dt = np.append(dsig_ext2_dt, dsig_ext2_dt[-1])
    d2sig_ext2_dt2 = np.diff(dsig_ext2_dt)/dt
    d2sig_ext2_dt2 = np.append(d2sig_ext2_dt2, d2sig_ext2_dt2[-1])
    # todo think about the 2nd derivative


    dmu_ext_dt = np.diff(mu_ext_filtered)/dt if filter_input else np.diff(mu_ext)/dt
    dmu_ext_dt = np.append(dmu_ext_dt,dmu_ext_dt[-1])
    d2mu_ext_dt2 = np.diff(dmu_ext_dt)/dt
    d2mu_ext_dt2 = np.append(d2mu_ext_dt2, d2mu_ext_dt2[-1])


    # print('remove neg_idxs=np.where(d2mu_ext_dt2<0)!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
    # neg_idxs=np.where(d2mu_ext_dt2<0)
    # d2mu_ext_dt2[neg_idxs]=0.

    # todo: also in new spec2m model file
    if set_2nd_derivs_0:
        d2mu_ext_dt2 = np.zeros_like(mu_ext)
        d2sig_ext2_dt2=np.zeros_like(mu_ext)

    if set_1st_derivs_0:
        #set first derivs to zero
        dmu_ext_dt = np.zeros_like(mu_ext)
        dsig_ext2_dt = np.zeros_like(mu_ext)

    if not set_1st_derivs_0:
        print('INCLUDE LINEAR TERMS, i.e. FIRST DERIVATIVES ENTER!')
    # assert dmu_ext_dt.all()==0. and dsig_ext2_dt.all()==0.
    # load quantities from hdf5-file
    file = tables.open_file('quantities_spectral_prefinal.h5')
    mu_tab          = file.root.mu.read()
    sig_tab         = file.root.sigma.read()
    lambda_1        = file.root.lambda_1.read()
    lambda_2        = file.root.lambda_2.read()
    f1              = file.root.f_1.read()
    f2              = file.root.f_2.read()
    c_mu_1          = file.root.c_mu_1.read()
    c_mu_2          = file.root.c_mu_2.read()
    c_sigma_1       = file.root.c_sigma_1.read()
    c_sigma_2       = file.root.c_sigma_2.read()
    dr_inf_dmu      = file.root.dr_inf_dmu.read()
    dr_inf_dsigma   = file.root.dr_inf_dsigma.read()
    dVm_dmu         = file.root.dV_mean_inf_dmu.read()
    dVm_dsig        = file.root.dV_mean_inf_dsigma.read()
    V_mean_inf      = file.root.V_mean_inf.read()
    r_inf           = file.root.r_inf.read()
    file.close()


    # switch off CTERMS for testing
    if C_terms_OFF:
        c_mu_1 = np.zeros_like(c_mu_1)
        c_mu_2 = np.zeros_like(c_mu_2)
        print('C-TERMS ARE SWITCHED OFF!')
    else:
        print('C-TERMS ENTER!')


    if dr_inf_dmu_OFF:
        dr_inf_dmu=np.zeros_like(dr_inf_dmu)

    # build meta data, which has to be real from theory
    # --> eliminate complex parts!
    T =                (lambda_1**-1 + lambda_2**-1).real             # 1/lambda_1+1/lambda_2
    D =                    ((lambda_1 * lambda_2)**-1).real             # 1/(lambda_1*lambda_2)
    M =       (dr_inf_dmu + (f1*c_mu_1 + f2*c_mu_2)).real
    F_mu =   (lambda_1*f1*c_mu_1+lambda_2*f2*c_mu_2).real
    fC_mu = (f1*c_mu_1 + f2*c_mu_2).real


    # quantities as derivatives with respect to sig NOT sig2
    # later on the get scaled by a factor of 1/(2*sig_tot)
    #  wihtin the integration loop, compare derivation
    # of the equations
    # --> eliminate complex parts!
    F_sig = (lambda_1*f1*c_sigma_1 + lambda_2*f2*c_sigma_2).real
    S_sig = (dr_inf_dsigma + (f1*c_sigma_1 + f2*c_sigma_2)).real
    fC_sig = (f1*c_sigma_1 + f2*c_sigma_2).real

    #####################################################################
    # prebuild maps that are passed into the inner loop
    # NOTE: b0,b1,b2 (and also bD for delay_type 1) are const, in time
    # for each model version and get therefore precalculated. bC is time
    # dependent as mu_ext, sigma_ext vary and cannot be precalculated
    # this way.
    #####################################################################
    # recurrent case
    if rec:
        # no delay
        if delay_type == 0:
            beta0_map = -D*M*(b/(tauw*C))-(b/C)*\
                        (T*M + D*M*(a/(tauw*C))*dVm_dmu-D*F_mu)
            # only bet0_map can be precomputed. all other maps
            # depend either on S (which has to be scaled by /(2*sig_tot))
            # or the mean adaptation current
            beta1_map = None
            beta2_map = None
            # make betaD a table with only 0 entries
            betaD_map = np.zeros_like(r_inf)

        # exponential delay distribution
        elif delay_type == 2:
            # make tables real
            beta0_map = None
            beta1_map = None
            beta2_map =  D
            betaD_map = None

        else:
            mes = 'Delay Type {} has not been ' \
                  'implemented for this model!'.format(params['delay_type'])
            raise NotImplementedError(mes)

    # feedforward case, no coupling
    else:
        beta0_map =  None
        beta1_map =  None
        beta2_map =   D
        betaD_map = np.zeros_like(r_inf)

    # initial values ???
    # put this HERE
    results = sim_spec2m_OLD_STACKING(mu_ext, dmu_ext_dt, d2mu_ext_dt2, sig_ext,
                         dsig_ext2_dt, d2sig_ext2_dt2, steps,t,dt,
                         mu_tab,sig_tab,rec, delay_type,K,J,taud,
                         tauw,a,b,C,Ew,beta0_map, beta1_map, beta2_map,
                         betaD_map,r_inf, V_mean_inf,T,S_sig,
                         D, M, F_mu, F_sig, dVm_dsig, dVm_dmu,
                         fC_mu,dr_inf_dmu,fC_sig,dr_inf_dsigma)
    results_dict = dict()
    results_dict['t'] = t
    results_dict['r']              = results[0]
    # only for debugging
    results_dict['D']              = results[2]
    results_dict['T']              = results[3]
    results_dict['dr_inf_dsigma2'] = results[4]
    results_dict['fC_sig2']        = results[5]
    results_dict['DF_sig2']        = results[6]
    results_dict['TS']             = results[7]
    results_dict['DS']             = results[8]
    results_dict['S']              = results[9]
    results_dict['F_sig2']         = results[10]
    results_dict['dr_inf_dmu']     = results[11]
    results_dict['fC_mu']          = results[12]
    results_dict['M']              = results[13]
    results_dict['F_mu']           = results[14]
    results_dict['TM']             = results[15]
    results_dict['DM']             = results[16]
    results_dict['DF_mu']          = results[17]

    if 'wm' in rec_vars: results_dict['wm'] = results[1]
    # results_dict[X] = Y
    # results_dict[X] = Y
    # results_dict[X] = Y
    return results_dict

#################################################################################################








































