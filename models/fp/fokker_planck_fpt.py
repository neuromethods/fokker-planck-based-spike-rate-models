'''neccessary funcitons and classes to run the scharfetter gummel discretization
of the fokker planck equation. Different modes are possible: feedforward, recurrent
and also finite size effects are included'''

from __future__ import print_function
import time
import numpy as np
import scipy.sparse
import scipy.sparse.linalg
from math import log, sqrt, exp
from scipy.linalg import solve_banded
# try to import numba
# or define dummy decorator
try:
    from numba import njit
except:
    def njit(func):
        return func


class Grid(object):
    '''
    This class implements the V discretization, which is used for both the
    Scharfetter-Gummel method and the upwind implementation
    '''

    def __init__(self, V_0=-200., V_1=-40., V_r=-70., N_V=100):
        self.V_0 = V_0
        self.V_1 = V_1
        self.V_r = V_r
        self.N_V = int(N_V)
        # construct the grid object
        self.construct()

    def construct(self):
        self.V_centers = np.linspace(self.V_0, self.V_1, self.N_V)
        # shift V_centers by half of the grid spacing to the left
        # such that the last interface lies exactly on V_l
        self.V_centers -= (self.V_centers[-1] - self.V_centers[-2]) / 2.
        self.dV_centers = np.diff(self.V_centers)
        self.V_interfaces = np.zeros(self.N_V + 1)
        self.V_interfaces[1:-1] = self.V_centers[:-1] + 0.5 * self.dV_centers
        self.V_interfaces[0] = self.V_centers[0] - 0.5 * self.dV_centers[0]
        self.V_interfaces[-1] = self.V_centers[-1] + 0.5 * self.dV_centers[-1]
        self.dV_interfaces = np.diff(self.V_interfaces)
        self.dV = self.V_interfaces[2] - self.V_interfaces[1]
        self.ib = np.argmin(np.abs(self.V_centers - self.V_r))



# DRIFT COEFFICIENTS
@njit
def get_v_numba(L, Vi, DT, EL, VT, taum, mu, EIF = True):
    '''drift coeffs for EIF/PIF model, depending on EIF = True/False'''
    # EIF model
    drift = np.empty(L)
    if EIF:
        for i in xrange(L):
            drift[i] = ((EL - Vi[i]) + DT * exp((Vi[i] - VT) / DT)) / taum + mu
    # LIF model
    else:
        for i in xrange(L):
            drift[i] = (EL - Vi[i]) / taum + mu
    drift[np.where(drift == 0.)] = 10e-15
    return drift

def get_v(grid, mu, params):
        '''returns the coefficients for the drift part of the flux for different neuron
         models. At the moment only for exponential-integrate-and-fire (EIF)
          and perfect-integrate-and-fire (PIF) '''
        Vi = grid.V_interfaces
        if params['neuron_model'] == 'EIF':
            EL = params['EL']
            taum = params['taum']
            DT = params['deltaT']
            VT = params['VT']
            #THIS TERM WILL LATER EXPLICITLY CONTAIN AN ADAPTATION CURRENT
            drift = ((EL - Vi) + DT * np.exp((Vi - np.ones_like(Vi) * VT) / DT)) / taum + mu
            # idea for Problem v --> 0: set all zero values of the drift array to mininmum value
            drift[np.where(drift == 0.)] = 10e-15
        elif params['neuron_model'] == 'PIF':
            drift = np.ones_like(Vi) * mu
            drift[np.where(drift == 0.)] = 10e-15
        else:
            err_mes = 'The model "{}" has not been implemented yet. For options see params dict.'.format(params['neuron_model'])
            raise NotImplementedError(err_mes)
        return drift
@njit
def get_diagonals_sg(dim_p, v, dV, dt, D):
    diag = np.empty(dim_p)
    lower = np.empty(dim_p - 1)
    upper = np.empty(dim_p - 1)
    for i in xrange(1, dim_p - 1):
        diag[i] = dV/dt + ((v[i + 1])) * (1. / (1. - exp((-v[i + 1] * dV) / D))) + ((v[i])) * (
        exp((-v[i] * dV) / D) / (1. - exp((-v[i] * dV) / D)))
        lower[i - 1] = -((v[i])) * (1. / (1. - exp((-v[i] * dV) / D)))
        upper[i] = -((v[i + 1])) * (exp((-v[i + 1] * dV) / D) / (1. - exp((-v[i + 1] * dV) / D)))

    diag[0] = dV/dt + ((v[1])) * (1. / (1. - exp((-v[1] * dV) / D)))
    upper[0] = -((v[1])) * ((exp((-v[1] * dV) / D)) / (1. - exp((-v[1] * dV) / D)))

    diag[-1] = dV/dt + ((v[-2])) * ((exp((-v[-2] * dV) / D)) / (1. - exp((-v[-2] * dV) / D))) + ((v[-1])) * ((1. + exp((-v[-1] * dV) / D)) / (1. - exp((-v[-1] * dV) / D)))
    lower[0] = -((v[1])) * (1. / (1. - exp((-v[1] * dV) / D)))
    lower[-1] = -((v[-2])) * (1. / (1. - exp((-v[-2] * dV) / D)))
    return (upper, diag, lower)




import math
#####################
# helper function for diags_A
@njit
def exp_vdV_D(v,dV,D):
    return math.exp(-v*dV/D)


# this is a new function which fills the three diagonals of a matrix A
# this matrix can be used in both versions of this solver.
# 1) banded solve: (1I - A*dt)p(t+dt)=p(t)
# 2) exponential integrator: p(t+dt)=exp(Adt)p(t) + inhom....

# len(p) = N
# len(v) = N+1


def diags_A(u,d,l,N,v,D,dV):
    # todo check this
    dV_inv = 1./dV

    # Note even though in Matrix A diag,sub,sup
    # do not have the same shape, here we use
    # arrays of equal length for all of them
    # because later we stack a NX3 matrix with them!

    #fill diag
    # element first and last elemet get overwritten for boundary conditions
    for i in xrange(1,N-1):
        # diag[i] = -dV_inv*(v[i]*exp_vdV_D(v[i],dV,D)/(1.-exp_vdV_D(v[i],dV,D))
        #              +v[i+1]/exp_vdV_D(v[i+1],dV,D))
        d[i] = -dV_inv*(v[i]*exp_vdV_D(v[i],dV,D)/(1.-exp_vdV_D(v[i],dV,D))
                           +v[i+1]/(1.-exp_vdV_D(v[i+1],dV,D)))
        # print(i)

    # fill sub diag
    for i in xrange(1,N-1):
        # first element of sub corresponds to p1
        # --> v_{-} is in this case v[1]
        l[i-1] = dV_inv*v[i]/(1.-exp_vdV_D(v[i],dV,D))
        # print(i)

    # fill sup
    for i in xrange(1,N-1):
        # sup[i+1] = v[i+1]*exp_vdV_D(v[i+1],dV,D)/(1.-exp_vdV_D(v[i+1],dV,D))
        u[i+1] = dV_inv*v[i+1]*exp_vdV_D(v[i+1],dV,D)/(1.-exp_vdV_D(v[i+1],dV,D))
        # print(i)


    # boundary conditions
    # reflecting
    # d[0] = -dV_inv*(v[1]/(1.-exp_vdV_D(v[1],dV,D)))
    d[0] = -dV_inv*(v[1]/(1.-exp_vdV_D(v[1],dV,D)))
    u[1] = dV_inv*(v[1]*exp_vdV_D(v[1],dV,D)/(1.-exp_vdV_D(v[1],dV,D)))

    # u[1] = dV_inv*v[1]*exp_vdV_D(v[1],dV,D)/(1.-exp_vdV_D(v[1],dV,D))

    # absorbing
    # d[-1] = -dV_inv*( v[-2]*exp_vdV_D(v[-2],dV,D)/(1.-exp_vdV_D(v[-2],dV,D))
    #              + v[-1]*(1.+exp_vdV_D(v[-1],dV,D))/(1.-exp_vdV_D(v[-1],dV,D)))
    # l[-2] = dV_inv*v[-2]/(1.-exp_vdV_D(v[-2],dV,D))
    ##################################################################
    l[-2] = dV_inv*v[-2]/(1.-exp_vdV_D(v[-2],dV,D))
    d[-1] = -dV_inv*(v[-2]*exp_vdV_D(v[-2],dV,D)/(1.-exp_vdV_D(v[-2],dV,D))
                     +v[-1]*(1.+exp_vdV_D(v[-1],dV,D))/(1.-exp_vdV_D(v[-1],dV,D)))

@njit
def matAdt(mat,N,v,D,dV,dt):
    # todo check this
    dt_dV = dt/dV

    # Note even though in Matrix A diag,sub,sup
    # do not have the same shape, here we use
    # arrays of equal length for all of them
    # because later we stack a NX3 matrix with them!

    #fill diag
    # element first and last elemet get overwritten for boundary conditions
    for i in xrange(1,N-1):
        # diag[i] = -dV_inv*(v[i]*exp_vdV_D(v[i],dV,D)/(1.-exp_vdV_D(v[i],dV,D))
        #              +v[i+1]/exp_vdV_D(v[i+1],dV,D))
        mat[1,i] = -dt_dV*(v[i]*exp_vdV_D(v[i],dV,D)/(1.-exp_vdV_D(v[i],dV,D))
                           +v[i+1]/(1.-exp_vdV_D(v[i+1],dV,D)))
        # print(i)

    # fill sub diag
    for i in xrange(1,N-1):
        # first element of sub corresponds to p1
        # --> v_{-} is in this case v[1]
        mat[2,i-1] = dt_dV*v[i]/(1.-exp_vdV_D(v[i],dV,D))
        # print(i)

    # fill sup
    for i in xrange(1,N-1):
        # sup[i+1] = v[i+1]*exp_vdV_D(v[i+1],dV,D)/(1.-exp_vdV_D(v[i+1],dV,D))
        mat[0,i+1] = dt_dV*v[i+1]*exp_vdV_D(v[i+1],dV,D)/(1.-exp_vdV_D(v[i+1],dV,D))
        # print(i)


    # boundary conditions
    # reflecting
    # d[0] = -dV_inv*(v[1]/(1.-exp_vdV_D(v[1],dV,D)))
    mat[1,0] = -dt_dV*(v[1]/(1.-exp_vdV_D(v[1],dV,D)))
    mat[0,1] = dt_dV*(v[1]*exp_vdV_D(v[1],dV,D)/(1.-exp_vdV_D(v[1],dV,D)))

    # u[1] = dV_inv*v[1]*exp_vdV_D(v[1],dV,D)/(1.-exp_vdV_D(v[1],dV,D))

    # absorbing
    # d[-1] = -dV_inv*( v[-2]*exp_vdV_D(v[-2],dV,D)/(1.-exp_vdV_D(v[-2],dV,D))
    #              + v[-1]*(1.+exp_vdV_D(v[-1],dV,D))/(1.-exp_vdV_D(v[-1],dV,D)))
    # l[-2] = dV_inv*v[-2]/(1.-exp_vdV_D(v[-2],dV,D))
    ##################################################################
    mat[2,-2] = dt_dV*v[-2]/(1.-exp_vdV_D(v[-2],dV,D))
    mat[1,-1] = -dt_dV*(v[-2]*exp_vdV_D(v[-2],dV,D)/(1.-exp_vdV_D(v[-2],dV,D))
                     +v[-1]*(1.+exp_vdV_D(v[-1],dV,D))/(1.-exp_vdV_D(v[-1],dV,D)))

@njit
def matAdt_opt(mat,N,v,D,dV,dt):
    # todo check this
    dt_dV = dt/dV

    # Note even though in Matrix A diag,sub,sup
    # do not have the same shape, here we use
    # arrays of equal length for all of them
    # because later we stack a NX3 matrix with them!
    # todo| I have already taken care that v is never 0,
    # todo| but could be implemented less hacky
    for i in xrange(1,N-1):
        # diagonal
        mat[1,i] = -dt_dV*(v[i]*exp_vdV_D(v[i],dV,D)/(1.-exp_vdV_D(v[i],dV,D))
                           +v[i+1]/(1.-exp_vdV_D(v[i+1],dV,D)))
        # lower diagonal
        mat[2,i-1] = dt_dV*v[i]/(1.-exp_vdV_D(v[i],dV,D))
        # upper diagonal
        mat[0,i+1] = dt_dV*v[i+1]*exp_vdV_D(v[i+1],dV,D)/(1.-exp_vdV_D(v[i+1],dV,D))

    # boundary conditions
    # first diagonal
    mat[1,0] = -dt_dV*(v[1]/(1.-exp_vdV_D(v[1],dV,D))) # check
    # first upper
    mat[0,1] = dt_dV * v[1]*exp_vdV_D(v[1],dV,D)/(1.-exp_vdV_D(v[1],dV,D)) #check
    # last lower
    mat[2,-2] = dt_dV*v[-2]/(1.-exp_vdV_D(v[-2],dV,D)) #check
    # last diagonal
    mat[1,-1] = -dt_dV*(v[-2]*exp_vdV_D(v[-2],dV,D)/(1.-exp_vdV_D(v[-2],dV,D))
                     +v[-1]*(1.+exp_vdV_D(v[-1],dV,D))/(1.-exp_vdV_D(v[-1],dV,D)))

# initial probability distribution
def initial_p_distribution(grid,params):
    if params['fp_v_init'] == 'normal':
        mean_gauss = params['fp_normal_mean']
        sigma_gauss = params['fp_normal_sigma']
        p_init = np.exp(-np.power((grid.V_centers - mean_gauss), 2) / (2 * sigma_gauss ** 2))
    elif params['fp_v_init'] == 'delta':
        delta_peak_index = np.argmin(np.abs(grid.V_centers - params['fp_delta_peak']))
        p_init = np.zeros_like(grid.V_centers)
        p_init[delta_peak_index] = 1.
    elif params['fp_v_init'] == 'uniform':
        # uniform dist on [Vr, Vcut]
        p_init = np.zeros_like(grid.V_centers)
        p_init[grid.ib:] = 1.
    else:
        err_mes = ('Initial condition "{}" is not implemented! See params dict for options.').format(params['fp_v_init'])
        raise NotImplementedError(err_mes)
    # normalization with respect to the cell widths
    p_init =p_init/np.sum(p_init*grid.dV_interfaces)
    return p_init

@njit
def get_r_numba(v_end, dV, D, p_end):
    '''rate calculation'''
    r = v_end*((1.+exp((-v_end*dV)/D))/(1.-exp((-v_end*dV)/D)))*p_end
    return r


# current version for model comparison
# currently without FS effects until everything is tested
def sim_fp_sg_fpt(mu_ext, sigma_ext, params, fpt=True, rt=list()):#, timing=None):
    '''solving the fp equation (first passage time)
     using the scharfetter gummel method'''

    dt = params['fp_dt']

    # external input
    #assert len(mu_ext) == len(sigma_ext)


    T_ref = params['T_ref']
    DT = params['DeltaT']
    EL = params['EL']
    VT = params['VT']
    taum = params['taum']

    EIF_model = True if params['neuron_model'] == 'EIF' else False



    # instance of the spatial grid class
    grid = Grid(V_0=params['Vlb'], V_1=params['Vcut'], V_r=params['Vr'],
                N_V=params['N_centers_fp'])

    # initial density
    #p0 = initial_p_distribution(grid, params)


    # initialize arrays
    r = np.zeros_like(mu_ext)
#    mu_tot    = np.zeros_like(mu_ext)
#    sigma_tot = np.zeros_like(mu_ext)
    #int_P     = np.zeros_like(mu_ext)
    #int_ref   = np.zeros_like(mu_ext)
    #mass_timecourse = np.zeros_like(mu_ext)
    #Vmean = np.zeros_like(mu_ext)


    # optimizations
    dV = grid.dV
    #p = p0
    Adt = np.zeros((3,grid.N_V))
    #reinject_ib = np.zeros(grid.N_V); reinject_ib[grid.ib] = 1.

    # used for resetting p to Vr
    rc=0
    n_rt = len(rt)
    
    ones_mat = np.ones(grid.N_V)
    
    # drift coefficients
    v = get_v_numba(grid.N_V+1, grid.V_interfaces, DT, EL, VT,
                    taum, mu_ext[0], EIF = EIF_model)  
    # Diffusion coefficient
    D = (sigma_ext[0] ** 2) * 0.5  
    # create banded matrix A 
    matAdt_opt(Adt,grid.N_V,v,D,dV,dt)

    for n in xrange(len(mu_ext)):
        # print('---------------------')
        # print(reset_times[reset_counter])
        # print(n*dt)
        # print(reset_times[reset_counter]+dt)

        if rc<n_rt and rt[rc]<= n*dt < rt[rc]+dt:
            p = initial_p_distribution(grid, params)
            rc += 1
            
        if rc-1<n_rt and rt[rc-1]<= n*dt < rt[rc-1]+T_ref+dt:
            r[n] = 0
        else:
            
        # optional:
        # calculate mass inside and outside the comp. domain
        #int_P[n] = np.sum(p*dV)
        #int_ref[n] = np.sum(r[n-n_ref:n]*dt)
        #mass_timecourse[n] = int_P[n] + int_ref[n]
        
        # normalize the density in the inner domain
        #p_marg = p/int_P[n]
        # calculate the mean membrane voltage
        #Vmean[n] = np.sum(p_marg*grid.V_centers*dV)
        

        # print out some information
        # if n%inform==0:
        #      print('simulated {}%, total mass: {}'.
        #            format((float(n)*100/steps), mass_timecourse[n]))


#        # mu_ext --> mu_syn
#        mu_tot[n] = mu_ext[n]
#        sigma_tot[n] = sigma_ext[n]

            if n>0:
                toggle = False
                if mu_ext[n]!=mu_ext[n-1]:
                    # drift coefficients
                    v = get_v_numba(grid.N_V+1, grid.V_interfaces, DT, EL, VT,
                                    taum, mu_ext[n], EIF = EIF_model)
                    toggle = True
                if sigma_ext[n]!=sigma_ext[n-1]:
                    # Diffusion coefficient
                    D = (sigma_ext[n] ** 2) * 0.5
                    toggle = True
                if toggle:
                    # create banded matrix A in each time step
                    matAdt_opt(Adt,grid.N_V,v,D,dV,dt)
            

            rhs = p.copy()   
            # toggle first passage time computation     
#            if not fpt: 
#                reinjection = r[n-n_ref]*(dt/dV)
#                rhs[grid.ib] += reinjection
            Adt *= -1.
            Adt[1,:] += ones_mat
            # solve the linear system
            p_new = solve_banded((1, 1), Adt, rhs)

            # compute rate / likelihood
            r[n] = get_r_numba(v[-1], dV, D, p_new[-1])
            # = (v[-1]*((1.+exp((-v[-1]*dV)/D))/(1.-exp((-v[-1]*dV)/D)))*p_new[-1])
    
            # overwrite P with updated P_new
            p = p_new

    # return time, rates and adaptation arrays
    results = {'r':r}
    return results


