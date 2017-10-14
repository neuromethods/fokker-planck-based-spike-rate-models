# spectral 3 model using in total three eigenvalues
# -> two diffusive and one regular mode





def sim_spec3():
    pass




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
    mu_tab = h5file.root.mu.read()
    sig_tab = h5file.root.sigma.read()


    lambda_reg_diff = h5file.root.lambda_reg_diff.read()
    lambda_1 = lambda_reg_diff[0, :, :]
    lambda_2 = lambda_reg_diff[1, :, :]
    lambda_3 = lambda_reg_diff[1, :, :]
    f = h5file.root.f.read()
    f1 = f[0, :, :]
    f2 = f[1, :, :]
    f3 = f[1, :, :]
    c_mu = h5file.root.c_mu.read()
    c_mu_1 = c_mu[0, :, :]
    c_mu_2 = c_mu[1, :, :]
    c_mu_3 = c_mu[2, :, :]
    c_sigma = h5file.root.c_sigma.read()
    c_sigma_1 = c_sigma[0, :, :]
    c_sigma_2 = c_sigma[1, :, :]
    c_sigma_3 = c_sigma[2, :, :]
    # psi_r = h5file.root.psi_r.read()
    # psi_r_1 = psi_r[0, :, :]
    # psi_r_2 = psi_r[1, :, :]


    dr_inf_dmu = h5file.root.dr_inf_dmu.read()
    dr_inf_dsigma = h5file.root.dr_inf_dsigma.read()
    dVm_dmu = h5file.root.dV_mean_inf_dmu.read()
    dVm_dsig = h5file.root.dV_mean_inf_dsigma.read()
    V_mean_inf = h5file.root.V_mean_inf.read()
    r_inf = h5file.root.r_inf.read()
    h5file.close()

    # idx = -1
    # plt.plot(lambda_1_old[:, idx], label = 'old')
    # plt.plot(lambda_2_old[:, idx], label = 'old')
    # plt.plot(lambda_1[:, idx])
    # plt.plot(lambda_2[:, idx])
    # plt.show()
    # plt.legend()
    # exit()

    # filter the input (typically done before calling here, thus disabled)
    # NOTE: this should result in a smooth input which
    # should be differentiable twice; maybe this function
    # will be transfered out of this function
    filter_input = False  # from now on assume calling with smooth input
    if filter_input:
        # filter input moments
        print('filtering inputs...');
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

    # compute 1st/2nd derivatives of input mean
    dmu_ext_dt = np.diff(mu_ext) / dt
    dmu_ext_dt = np.insert(dmu_ext_dt, 0, dmu_ext_dt[-1])  # 1st order backwards differences
    d2mu_ext_dt2 = np.diff(dmu_ext_dt) / dt
    d2mu_ext_dt2 = np.append(d2mu_ext_dt2, d2mu_ext_dt2[-1])  # 2nd order central diffs

    # precompute some matrices which are used in the integration loop
    # Notation adapted from the manuscript!
    # --> complex parts vanish as quantities are multiplicaitons of
    # complex conjugated pairs (-> cf. manuscript)
    T = (lambda_1 ** -1 + lambda_2 ** -1 + lambda_3 ** -1).real
    D = ((lambda_1 * lambda_2 * lambda_3) ** -1).real
    # is p real
    P = ((lambda_1 * lambda_2) ** -1 + (lambda_1 * lambda_2) ** -1 + (lambda_1 * lambda_2) ** -1).real
    F_mu = (lambda_1 * f1 * c_mu_1 + lambda_2 * f2 * c_mu_2).real

    # only for debugging
    fC_mu = (f1 * c_mu_1 + f2 * c_mu_2).real
    fC_sig = (f1 * c_sigma_1 + f2 * c_sigma_2).real

    # quantities as derivatives with respect to sig NOT sig2
    # later on the get scaled by a factor of 1/(2*sig_tot)
    #  wihtin the integration loop, compare derivation
    # of the equations
    # --> eliminate complex parts!
    F_sig = (lambda_1 * f1 * c_sigma_1 + lambda_2 * f2 * c_sigma_2).real
    S_sig = (dr_inf_dsigma + (f1 * c_sigma_1 + f2 * c_sigma_2)).real

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

    results = sim_spec2(mu_ext, dmu_ext_dt, d2mu_ext_dt2,
                        sig_ext, dsig_ext2_dt, d2sig_ext2_dt2, steps, t, dt,
                        mu_tab, sig_tab, rec, delay_type, const_delay,
                        K, J, taud, tauw, a, b, C, wm0,
                        Ew, r_inf, V_mean_inf, T, S_sig, D, M, F_mu, F_sig,
                        dVm_dsig, dVm_dmu, dr_inf_dmu, dr_inf_dsigma, fC_mu,
                        fC_sig, uni_int_order, rectify)

    results_dict = dict()
    results_dict['t'] = t
    results_dict['r'] = results[0]
    if 'wm' in rec_vars: results_dict['wm'] = results[1]

    # return results
    return results_dict





















