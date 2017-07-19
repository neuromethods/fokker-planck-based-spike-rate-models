 

# backed up from caluclate_quantities_spectral.py (validation via maurizio mat files)
        
#
#    # use precalc05_int.mat data for validation (of mean(V; p_inf(ref)) and r_inf as well as dr_dmu, dr_dsigma, ) first
#
#
#    # compare vs. maurizio matfiles (todo: compare with matfile from josef r_inf modelxomp)
#    assert params['conjugation_first_imag_negative']
#    
#    
#    
#    load_maurizio = lambda filename: scipy.io.loadmat(folder_maurizio+'/'+filename)['Expression1'].T
#    
#    quantities_validation = {'mu': load_maurizio('F1_X.mat').flatten() / 1000.,
#                             'sigma': load_maurizio('F1_Y.mat').flatten() / math.sqrt(1000.),
#                             'lambda_1': load_maurizio('Lambda1_Z.mat') / 1000.,
#                             'lambda_2': load_maurizio('Lambda-1_Z.mat') / 1000.,                             
#                             'r_inf': load_maurizio('Phin_Z.mat') / 1000.,
#                             'dr_inf_dmu': load_maurizio('DPhiDMun_Z.mat'),
#                             'dr_inf_dsigma': load_maurizio('DPhiDSigman_Z.mat') / math.sqrt(1000.), # * sigma?
#                             # V_mean and its derivatives w.r.t mu/sigma not yet available from Maurizio
#                             'f_1': load_maurizio('F1_Z.mat') / 1000.,
#                             'f_2': load_maurizio('F-1_Z.mat') / 1000.,
#                             'psi_r_1': load_maurizio('Psi1_Z.mat'),
#                             'psi_r_2': load_maurizio('Psi-1_Z.mat'),
#                             'c_mu_1': load_maurizio('cMu1_Z.mat') * 1000.,
#                             'c_mu_2': load_maurizio('cMu-1_Z.mat') * 1000.,
#                             'c_sigma_1': load_maurizio('DcDsigma1_Z.mat') * math.sqrt(1000.),
#                             'c_sigma_2': load_maurizio('DcDsigma-1_Z.mat') * math.sqrt(1000.)
#                            }
#    # phi_inf can be computed analytically due to Mattia02 PRE
#    mu_ana = quantities_validation['mu'] * 1000. # here we need seconds units
#    sigma_ana = quantities_validation['sigma'] * math.sqrt(1000.)
#    mu_ana = mu_ana.reshape(-1, 1)
#    sigma_ana = sigma_ana.reshape(1, -1)
#    xi = mu_ana/sigma_ana**2
#    c = 1./(sigma_ana**2/(2*mu_ana**2) * (2*mu_ana/sigma_ana**2 - 1 + np.exp(-2*mu_ana/sigma_ana**2)))
#    V_ana = np.linspace(0, 1, params['grid_V_points'])
#    V_mean_inf_ana = np.zeros_like(quantities_validation['r_inf'])    
#    for i in range(len(mu_ana)):
#        for j in range(len(sigma_ana)):
#            phi_ana_ij = c[i,j]/mu_ana[i] * (1-np.exp(-2*xi[i,j]*(1-V_ana)))
#            V_mean_inf_ana[i,j] = np.sum(np.diff(V_ana)*0.5*((V_ana*phi_ana_ij)[:-1]+(V_ana*phi_ana_ij)[1:]))
#    quantities_validation['V_mean_inf'] = V_mean_inf_ana
    # the mu/sigma derivatives of V_mean_inf could be added here too using, e.g., central finite differences
    
# the following multiplied quantities are generated in plot_quantities_complex
#    quantities_validation['f_1*psi_r_1'] = quantities_validation['f_1'] * quantities_validation['psi_r_1']
#    quantities_validation['f_2*psi_r_2'] = quantities_validation['f_2'] * quantities_validation['psi_r_2']
#    quantities_validation['f_1*c_mu_1'] = quantities_validation['f_1'] * quantities_validation['c_mu_1']
#    quantities_validation['f_2*c_mu_2'] = quantities_validation['f_2'] * quantities_validation['c_mu_2']
#    quantities_validation['f_1*c_sigma_1'] = quantities_validation['f_1'] * quantities_validation['c_sigma_1']
#    quantities_validation['f_2*c_sigma_2'] = quantities_validation['f_2'] * quantities_validation['c_sigma_2']
                            
