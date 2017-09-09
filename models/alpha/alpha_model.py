# simulates the dynamics of the alpha dynamics specified by the equations B1-B3
# in the paper.
# -- todo: include adaptation and recurrency
import tables
import numpy as np
from misc.utils import interpolate_xy, lookup_xy


def lookup_helper(tensor, q, q_string, N_eigvals, weights):

    if q_string in ['C_mu', 'C_sigma']:
        # do stuff for matrix of matrices
        for i in range(N_eigvals):
            for j in range(N_eigvals):
                tensor[i, j] = lookup_xy(q[i, j, :,:], weights)
    elif q_string == 'vander':
        # fill diagonal vandermonde matrix
        for i in range(N_eigvals):
            tensor[i, i] = lookup_xy(q[i, :, :], weights)
    else:
        # do stuff for vector of matrices
        for i in range(N_eigvals):
            tensor[i] = lookup_xy(q[i, :], weights)



def sim_alpha(mu_ext, sigma_ext, dmu_dt, dsigma2_dt,
              mu_range, sigma_range, lambda_all,
              r_inf, f, c_mu, c_sigma, C_mu, C_sigma,
              N_eigvals, a, b, Ew, tauw, t, steps, dt):

    # how to initialize the alphas
    alpha  = np.zeros(N_eigvals) + 0j
    vander_mat = np.zeros((N_eigvals, N_eigvals)) + 0j
    C_mu_mat = np.zeros((N_eigvals, N_eigvals)) + 0j
    C_sigma_mat = np.zeros((N_eigvals, N_eigvals)) + 0j
    c_mu_vec = np.zeros(N_eigvals) + 0j
    c_sigma_vec = np.zeros(N_eigvals) + 0j
    f_vec = np.zeros(N_eigvals) + 0j

    r = np.zeros(steps+1)
    r_without_f_alpha = np.zeros(steps+1)
    f_alpha_vec = np.zeros(steps+1)

    for i in range(steps):

        mu_tot = mu_ext[i]
        sigma_tot = sigma_ext[i]

        # interpolate
        weights = interpolate_xy(mu_tot, sigma_tot, mu_range, sigma_range)
        # fill the matrices/vectors
        # build vander
        lookup_helper(vander_mat, lambda_all, 'vander', N_eigvals, weights)
        # build C_mu
        lookup_helper(C_mu_mat, C_mu, 'C_mu', N_eigvals, weights)
        # build C_sigma
        lookup_helper(C_sigma_mat, C_sigma, 'C_sigma', N_eigvals, weights)
        # build c_mu
        lookup_helper(c_mu_vec, c_mu, None, N_eigvals, weights)
        # build c_sigma
        lookup_helper(c_sigma_vec, c_sigma, None, N_eigvals, weights)
        # build f
        lookup_helper(f_vec, f, None, N_eigvals, weights)
        # get r_inf
        r_inf_val = lookup_xy(r_inf, weights)
        # construct matrix M
        M = vander_mat + C_mu_mat * dmu_dt[i] + C_sigma_mat * dsigma2_dt[i] # matrix of complex numbers
        c = c_mu_vec * dmu_dt[i] + c_sigma_vec * dsigma2_dt[i] # vector of complex numbers

        # Euler step: get alpha(i+1)
        alpha = alpha + dt*(M.dot(alpha)+c)
        f_alpha = f_vec.dot(alpha.T)
        # get new r
        r[i+1] = r_inf_val  + f_alpha

        # return results
    return [r*1000, r_without_f_alpha*1000, f_alpha_vec]




# wrapper function to extract all the quantities
# sim_alpha is called from wihtin this function
def run_alpha(ext_signal, params, filename):


    # decide how many eigenvalues are considered
    N_eigvals = 2

    # external sigma
    mu_ext = ext_signal[0]
    sigma_ext = ext_signal[1]

    # extract quantities
    h5file = tables.open_file(filename, mode = 'r')
    mu_range = np.array(h5file.root.mu)
    sigma_range = np.array(h5file.root.sigma)
    lambda_all = np.array(h5file.root.lambda_all)
    r_inf = np.array(h5file.root.r_inf)
    f = np.array(h5file.root.f)
    c_mu = np.array(h5file.root.c_mu)
    c_sigma = np.array(h5file.root.c_sigma)
    C_mu = np.array(h5file.root.C_mu)
    C_sigma = np.array(h5file.root.C_sigma)
    h5file.close()
    # extract params from dictionary
    runtime = params['runtime']
    dt = params['uni_dt']
    steps = int(runtime/dt)
    t = np.linspace(0., runtime, steps+1)
    a = params['a']
    b = params['b']
    Ew = params['Ew']
    tauw = params['tauw']



    # todo check derivative of input with respect to time again
    dmu_dt = np.diff(mu_ext)/dt
    dsigma2_dt = np.diff(sigma_ext**2)/dt


    sim_results = sim_alpha(mu_ext, sigma_ext, dmu_dt, dsigma2_dt, mu_range,
                        sigma_range, lambda_all, r_inf, f, c_mu, c_sigma,
                        C_mu, C_sigma, N_eigvals, a, b, Ew, tauw, t, steps, dt)




    return {'r':sim_results[0], 't':t,'r_without':sim_results[1], 'f_alpha':sim_results[2]}





