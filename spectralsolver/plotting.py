# -*- coding: utf-8 -*-
"""
Created on Wed Mar  9 11:25:42 2016

@author: augustin
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx
import misc.utils

def get_sigma_inds_validation(inds_sigma, sigma, sigma_validation):
    inds_sigma_validation = []
    for j in inds_sigma:
        sigma_j_validation = np.argmin(np.abs(sigma[j] - sigma_validation))
        inds_sigma_validation.append(sigma_j_validation)
    return inds_sigma_validation

def plot_quantities_eigvals(quantities, inds_sigma_plot, colormap_sigma='winter',
                            plot_validation=False, quantities_validation={}, 
                            marker_validation='o', ms_validation=2, 
                            color_validation='black', linestyle_validation='None'):

    mu = quantities['mu']
    sigma = quantities['sigma']
    lambda_1 = quantities['lambda_1']
    lambda_2 = quantities['lambda_2']

    N_sigma = sigma.shape[0]

   # final eigenvalue quantities
    plt.figure()
    plt.suptitle('final eigenvalues $\lambda_1, \lambda_2$')
    
    lambda_1_2 = np.concatenate([lambda_1, lambda_2])
    ylim_real = [np.amin(lambda_1_2.real), 0]
    
    if plot_validation:
        inds_sigma_validation = get_sigma_inds_validation(inds_sigma_plot, 
                                                          sigma, 
                                                          quantities_validation['sigma'])
                                    
    
    ax_real = plt.subplot(3, 2, 1)
    ax_imag = plt.subplot(3, 2, 2)
    for k_j, j in enumerate(inds_sigma_plot):
            
        # color
        cm = plt.get_cmap(colormap_sigma) 
        cNorm  = colors.Normalize(vmin=sigma[0], vmax=sigma[-1])
        scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cm)
        linecolor = scalarMap.to_rgba(sigma[j])
        # for poster cns prague 2015
#            rgb = [0, float(k_j)/(len(inds_sigma_plot)-1), 0]
#            linecolor = rgb
        
        if plot_validation:
            j_validation = inds_sigma_validation[k_j]
        
        # lambda_1 (real part)
        plt.subplot(3, 2, 1)
        # labels
        if j in [0, N_sigma//2, N_sigma-1]:
            siglabel = '$\sigma={0:.3}$ [mV/$\sqrt{{\mathrm{{ms}}}}$]'.format(sigma[j])
            if plot_validation:
                siglabel += ', $\sigma_\mathrm{{val}}=$ {0:.3}'.format(quantities_validation['sigma'][j_validation])
        else:
            siglabel = None

        if plot_validation and 'lambda_1' in quantities_validation:
            plt.plot(quantities_validation['mu'], 
                     quantities_validation['lambda_1'][:, j_validation].real,
                     color=color_validation, marker=marker_validation, markersize=ms_validation, 
                     linestyle=linestyle_validation)


        plt.plot(mu, lambda_1[:, j].real, label=siglabel, color=linecolor)

        if j==0:
            plt.title('real part')
            plt.ylim(ylim_real)
            plt.ylabel('$\Re(\lambda_1)$ [kHz]')
        if j==N_sigma-1:
            plt.legend(loc='best')
        
        
        # lambda_1 (imag part)
        plt.subplot(3, 2, 2, sharex=ax_real)

        if plot_validation and 'lambda_1' in quantities_validation:
            plt.plot(quantities_validation['mu'], 
                     quantities_validation['lambda_1'][:, j_validation].imag,
                     color=color_validation, marker=marker_validation, markersize=ms_validation,
                     linestyle=linestyle_validation)

        plt.plot(mu, lambda_1[:, j].imag, color=linecolor)
        
        if j==0:
            plt.title('imag. part')
            plt.ylim(np.amin(lambda_1.imag), np.amax(lambda_1.imag))
            plt.ylabel('$\Im(\lambda_1)$ [kHz]')
    
        
        # lambda_2 (real part)
        plt.subplot(3, 2, 3, sharex=ax_real, sharey=ax_real)
        
        if plot_validation and 'lambda_2' in quantities_validation:
            plt.plot(quantities_validation['mu'], 
                     quantities_validation['lambda_2'][:, j_validation].real,
                     color=color_validation, marker=marker_validation, markersize=ms_validation, 
                     linestyle=linestyle_validation)            
        
        plt.plot(mu, lambda_2[:, j].real, color=linecolor)
        
        if j==0:
            plt.ylim(ylim_real)
            plt.ylabel('$\Re(\lambda_2)$ [kHz]')
        
        plt.xlabel('$\mu$ [mV/ms]')
        
        
        # lambda_2 (imag part)
        plt.subplot(3, 2, 4, sharex=ax_real, sharey=ax_imag)
        
        if plot_validation and 'lambda_2' in quantities_validation:
            plt.plot(quantities_validation['mu'], 
                     quantities_validation['lambda_2'][:, j_validation].imag,
                     color=color_validation, marker=marker_validation, markersize=ms_validation, 
                     linestyle=linestyle_validation)            
        
        plt.plot(mu, lambda_2[:, j].imag, color=linecolor)
        
        if j==0:
            plt.ylim(np.amin(lambda_2.imag), np.amax(lambda_2.imag))
            plt.ylabel('$\Im(\lambda_2)$ [kHz]')
            
        plt.xlabel('$\mu$ [mV/ms]')


def plot_quantities_real(quantities, inds_sigma_plot, colormap_sigma='winter',
                         plot_validation=False, quantities_validation={}, 
                         marker_validation='o', ms_validation=2, 
                         color_validation='black', linestyle_validation='None'):
    
    mu = quantities['mu']
    sigma = quantities['sigma']
    r_inf = quantities['r_inf']
    dr_inf_dmu = quantities['dr_inf_dmu']
    dr_inf_dsigma = quantities['dr_inf_dsigma']
    V_mean_inf = quantities['V_mean_inf']
    dV_mean_inf_dmu = quantities['dV_mean_inf_dmu']
    dV_mean_inf_dsigma = quantities['dV_mean_inf_dsigma']
    

    N_sigma = sigma.shape[0]    
    
    plt.figure() 
    
    plt.suptitle('real quantities')
    
    if plot_validation:
        inds_sigma_validation = get_sigma_inds_validation(inds_sigma_plot, 
                                                      sigma, 
                                                      quantities_validation['sigma'])

    for k_j, j in enumerate(inds_sigma_plot):
        # color
        cm = plt.get_cmap(colormap_sigma) 
        cNorm  = colors.Normalize(vmin=sigma[0], vmax=sigma[-1])
        scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cm)
        linecolor = scalarMap.to_rgba(sigma[j])
        
        if plot_validation:
            j_validation = inds_sigma_validation[k_j]        
        
        # r_inf
        ax1 = plt.subplot(2, 3, 1)
    
        # labels
        if j in [0, N_sigma//2, N_sigma-1]:
            siglabel = '$\sigma={0:.3}$ [mV/$\sqrt{{\mathrm{{ms}}}}$]'.format(sigma[j])
            if plot_validation:
                siglabel += ', $\sigma_\mathrm{{val}}=$ {0:.3}'.format(quantities_validation['sigma'][j_validation])
            
        else:
            siglabel = None

        if plot_validation and 'r_inf' in quantities_validation:

            plt.plot(quantities_validation['mu'], 
                     quantities_validation['r_inf'][:, j_validation],
                     color=color_validation, marker=marker_validation, markersize=ms_validation,
                     linestyle=linestyle_validation)

        plt.plot(mu, r_inf[:, j], label=siglabel, color=linecolor)

        if j==0:
#                plt.ylim(0, 30)
            plt.ylabel('$r_\infty$ [kHz]')
        if j==N_sigma-1:
            plt.legend(loc='best')
        
        # dr_inf_dmu
        plt.subplot(2, 3, 2, sharex=ax1)

        if plot_validation and 'dr_inf_dmu' in quantities_validation:
            plt.plot(quantities_validation['mu'], 
                     quantities_validation['dr_inf_dmu'][:, j_validation],
                     color=color_validation, marker=marker_validation, markersize=ms_validation, 
                     linestyle=linestyle_validation)
        
        
        plt.plot(mu, dr_inf_dmu[:, j], color=linecolor)
        if j==0:
#                 plt.ylim(0, 200)
            plt.ylabel('$\partial_\mu r_\infty$ [1/mV)]')
        
        # dr_inf_dsigma
        plt.subplot(2, 3, 3, sharex=ax1)

        if plot_validation and 'dr_inf_dsigma' in quantities_validation:
            plt.plot(quantities_validation['mu'], 
                     quantities_validation['dr_inf_dsigma'][:, j_validation],
                     color=color_validation, marker=marker_validation, markersize=ms_validation, 
                     linestyle=linestyle_validation)

        plt.plot(mu, dr_inf_dsigma[:, j], color=linecolor)
        if j==0:
#                 plt.ylim(-90, -50)
            plt.ylabel('$\partial_\sigma r_\infty$ [$1/(\mathrm{mV} \sqrt{{\mathrm{{ms}}}})$]')
        
        # V_mean_inf
        plt.subplot(2, 3, 4, sharex=ax1)
        
        if plot_validation and 'V_mean_inf' in quantities_validation:
            plt.plot(quantities_validation['mu'], 
                     quantities_validation['V_mean_inf'][:, j_validation],
                     color=color_validation, marker=marker_validation, markersize=ms_validation, 
                     linestyle=linestyle_validation)
        
        plt.plot(mu, V_mean_inf[:, j], color=linecolor)
        if j==0:
#                plt.ylim(0, 0.5)
            plt.ylabel('$\langle V \\rangle_\infty$ [mV]')
            plt.xlabel('$\mu$ [mV/s]')
        
        # dV_mean_inf_dmu
        plt.subplot(2, 3, 5, sharex=ax1)
        
        if plot_validation and 'dV_mean_inf_dmu' in quantities_validation:
            plt.plot(quantities_validation['mu'], 
                     quantities_validation['dV_mean_inf_dmu'][:, j_validation],
                     color=color_validation, marker=marker_validation, markersize=ms_validation, 
                     linestyle=linestyle_validation)
        
        plt.plot(mu, dV_mean_inf_dmu[:, j], color=linecolor)
        if j==0:
            plt.ylabel('$\partial_\mu \langle V \\rangle_\infty$ [ms]')
            plt.xlabel('$\mu$ [mV/ms]')
        
        # dV_mean_inf_dmu
        plt.subplot(2, 3, 6, sharex=ax1)        

        if plot_validation and 'dV_mean_inf_dsigma' in quantities_validation:
            plt.plot(quantities_validation['mu'], 
                     quantities_validation['dV_mean_inf_dsigma'][:, j_validation],
                     color=color_validation, marker=marker_validation, markersize=ms_validation, 
                     linestyle=linestyle_validation)
                
        plt.plot(mu, dV_mean_inf_dsigma[:, j], color=linecolor)
        if j==0:
            plt.ylabel(r'$\partial_\sigma \langle V \rangle_\infty [\sqrt{\mathrm{ms}}]$')
            plt.xlabel('$\mu$ [mV/s]')
        
def plot_quantities_complex(complex_quants_plot, quantities, inds_sigma_plot, 
                            colormap_sigma='winter',
                            plot_validation=False, quantities_validation={}, 
                            marker_validation='o', ms_validation=2, 
                            color_validation='black', linestyle_validation='None'):
    
    # generate summed and/or multiplied quantities    
    for q in complex_quants_plot:        
        q_mult_index = q.find('*')            
        if q_mult_index >= 0:
            q_left = q[:q_mult_index]
            q_right = q[q_mult_index+1:]
            quantities[q] = quantities[q_left] * quantities[q_right]
            # validation (currently manually computed in script)
            if plot_validation and q_left in quantities_validation and q_right in quantities_validation:
                quantities_validation[q] = quantities_validation[q_left] * quantities_validation[q_right]
        
    

    sigma = quantities['sigma']
    N_sigma = sigma.shape[0]
    
    
    plt.figure()
    
    plt.suptitle('complex quantities')    
    
    spr = misc.utils.SubplotRect(2, len(complex_quants_plot))    
    
    if plot_validation:
        inds_sigma_validation = get_sigma_inds_validation(inds_sigma_plot, 
                                                  sigma, 
                                                  quantities_validation['sigma'])

    for k_j, j in enumerate(inds_sigma_plot):
        
        
        # color
        cm = plt.get_cmap(colormap_sigma) 
        cNorm  = colors.Normalize(vmin=sigma[0], vmax=sigma[-1])
        scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cm)
        linecolor = scalarMap.to_rgba(sigma[j])
        
        if plot_validation:
            j_validation = inds_sigma_validation[k_j]   
        
        spr.first()
        ax1 = spr.current_axes()
        firstquant = True
        
        for q in complex_quants_plot:        

            if firstquant and j == 0:
                plt.ylabel('real part')
            
            if firstquant and j in  [0, N_sigma//2, N_sigma-1]:
                siglabel = '$\sigma={0:.3}$ [mV/$\sqrt{{\mathrm{{ms}}}}$]'.format(sigma[j])
                if plot_validation:
                    siglabel += ', $\sigma_\mathrm{{val}}=$ {0:.3}'.format(quantities_validation['sigma'][j_validation])
            else:
                siglabel = None

            # we are already in real part subplot row in the col of the current quantity
            #spr.current_axes()

            plt.title(q)

            
            if plot_validation and q in quantities_validation:
                plt.plot(quantities_validation['mu'], 
                         quantities_validation[q][:, j_validation].real,
                         color=color_validation, marker=marker_validation, 
                         markersize=ms_validation, linestyle=linestyle_validation)
                    
            plt.plot(quantities['mu'], quantities[q][:, j].real, 
                     color=linecolor, label=siglabel)            

            if firstquant and j == N_sigma-1:                
                plt.legend(loc='best')
            
            
            # move to imag. part subplot row again            
            spr.nextrow(sharex=ax1)   
            
            if firstquant and j == 0:
                plt.ylabel('imag. part')
                
            if j == 0:
                plt.xlabel('$\mu$')
                
            if plot_validation and q in quantities_validation:
                plt.plot(quantities_validation['mu'], 
                         quantities_validation[q][:, j_validation].imag,
                         color=color_validation, marker=marker_validation, 
                         markersize=ms_validation, linestyle=linestyle_validation)
                    
            
            plt.plot(quantities['mu'], quantities[q][:, j].imag, color=linecolor)
            
            
            # move to real part subplot row again
            spr.nextrow(sharex=ax1)
                  
            # move to next quantity
            spr.nextcol(sharex=ax1)    
        
            firstquant = False    


def plot_quantities_composed(composed_quantities, quantities, inds_sigma_plot, 
                            colormap_sigma='winter',
                            plot_validation=False, quantities_validation={}, 
                            comp_quants_validat={},
                            marker_validation='o', ms_validation=2, 
                            color_validation='black', linestyle_validation='None'):

    sigma = quantities['sigma']
    N_sigma = sigma.shape[0]
    
    
    plt.figure()
    
    plt.suptitle('composed quantities (e.g., dot products)')    
    
    spr = misc.utils.SubplotRect(2, len(composed_quantities))    
    
    if plot_validation:
        inds_sigma_validation = get_sigma_inds_validation(inds_sigma_plot, 
                                                  sigma, 
                                                  quantities_validation['sigma'])

    for k_j, j in enumerate(inds_sigma_plot):
        
        
        # color
        cm = plt.get_cmap(colormap_sigma) 
        cNorm  = colors.Normalize(vmin=sigma[0], vmax=sigma[-1])
        scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cm)
        linecolor = scalarMap.to_rgba(sigma[j])
        
        if plot_validation:
            j_validation = inds_sigma_validation[k_j]   
        
        spr.first()
        ax1 = spr.current_axes()
        firstquant = True
        
        for q in composed_quantities.keys():        

            if firstquant and j == 0:
                plt.ylabel('real part')
            
            if firstquant and j in  [0, N_sigma//2, N_sigma-1]:
                siglabel = '$\sigma={0:.3}$ [mV/$\sqrt{{\mathrm{{ms}}}}$]'.format(sigma[j])
                if plot_validation:
                    siglabel += ', $\sigma_\mathrm{{val}}=$ {0:.3}'.format(quantities_validation['sigma'][j_validation])
            else:
                siglabel = None

            # we are already in real part subplot row in the col of the current quantity
            #spr.current_axes()

            plt.title(q)

            
            if plot_validation and q in comp_quants_validat:
                plt.plot(quantities_validation['mu'], 
                         comp_quants_validat[q][:, j_validation].real,
                         color=color_validation, marker=marker_validation, 
                         markersize=ms_validation, linestyle=linestyle_validation)
                    
            plt.plot(quantities['mu'], composed_quantities[q][:, j].real, 
                     color=linecolor, label=siglabel)            

            if firstquant and j == N_sigma-1:                
                plt.legend(loc='best')
            
            
            # move to imag. part subplot row again            
            spr.nextrow(sharex=ax1)   
            
            if firstquant and j == 0:
                plt.ylabel('imag. part')
                
            if j == 0:
                plt.xlabel('$\mu$')
                
            if plot_validation and q in comp_quants_validat:
                plt.plot(quantities_validation['mu'], 
                         comp_quants_validat[q][:, j_validation].imag,
                         color=color_validation, marker=marker_validation, 
                         markersize=ms_validation, linestyle=linestyle_validation)
                    
            
            plt.plot(quantities['mu'], composed_quantities[q][:, j].imag, color=linecolor)
            
            
            # move to real part subplot row again
            spr.nextrow(sharex=ax1)
                  
            # move to next quantity
            spr.nextcol(sharex=ax1)    
        
            firstquant = False    


def plot_raw_spectrum_sigma(lambda_all, mu, sigma, sigma_inds, max_per_fig=6, colormap='jet'):
    
    N_eigvals = lambda_all.shape[0]
    N_sigma = sigma.shape[0]
    
    for sii in range(0, len(sigma_inds), max_per_fig):
        sigma_inds_fig = sigma_inds[sii:min(sii+max_per_fig, N_sigma)]
        # subplot variable: sigma 
        plt.figure()
        plt.suptitle('full spectrum in $\mu,\sigma$ space by $\sigma$')
        subplotid = 1 # left/right real/imag, rows: eigvals
        N_plotcols = len(sigma_inds_fig) # N_sigma//sigma_skip_inds+(1 if N_sigma % sigma_skip_inds > 0 else 0)
        # axis sharing
        ax_real = plt.subplot(2, N_plotcols, subplotid, sharex=None, sharey=None)
        ax_imag = plt.subplot(2, N_plotcols, subplotid+N_plotcols, sharex=ax_real, sharey=None)
        for j in sigma_inds_fig: #  range(0, N_sigma, sigma_skip_inds):
            
            
            for k in range(N_eigvals):
                # labels
                if k in range(N_eigvals) and j==0:
                    eiglabel = '$\lambda_{0}^\mathrm{{raw}}$'.format(k+1)
                else:
                    eiglabel = None
                    
                # color
                cm = plt.get_cmap(colormap) 
                cNorm  = colors.Normalize(vmin=0, vmax=N_eigvals-1)
                scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cm)
                linecolor = scalarMap.to_rgba(k)
                
                # eigenval: real part
                plt.subplot(2, N_plotcols, subplotid, 
                            sharex=ax_real if subplotid > 1 else None, 
                            sharey=ax_real if subplotid > 1 else None)
                plt.plot(mu, lambda_all[k, :, j].real, label=eiglabel, color=linecolor)
                if j==0:
                    if k==0:
                        plt.ylabel('$\Re(\lambda)$ [kHz]')
                if k==0:
                    plt.title('$\sigma={0:.3}$ [mV/$\sqrt{{\mathrm{{ms}}}}$]'.format(sigma[j]))
                    plt.ylim(np.amin(lambda_all.real), 0)
                
                # eigenval: imag part
                plt.subplot(2, N_plotcols, subplotid+N_plotcols, 
                            sharex=ax_real, 
                            sharey=ax_imag if subplotid > 1 else None)
                plt.plot(mu, lambda_all[k, :, j].imag, color=linecolor)
                if j==0:
                    if k==0:
                        plt.ylabel('$\Im(\lambda)$ [kHz]')
                if k==0: 
                    plt.xlabel('$\mu$ [mV/ms]')
                    plt.ylim(np.amin(lambda_all.imag), np.amax(lambda_all.imag))
            
            if j==0:
                plt.subplot(2, N_plotcols, subplotid, 
                            sharex=ax_real if subplotid > 1 else None, 
                            sharey=ax_real if subplotid > 1 else None)
                plt.legend(loc='best')
            
            subplotid += 1
        

def plot_raw_spectrum_eigvals(lambda_all, mu, sigma, colormap='winter'):
    
    N_eigvals = lambda_all.shape[0]
    N_sigma = lambda_all.shape[2]
    
    # subplot variable: eigenvalue
    plt.figure()
    plt.suptitle('full spectrum in $\mu,\sigma$ space by $\lambda$ index')
    subplotid = 1 # left/right real/imag, rows: eigvals
    ax_real = plt.subplot(N_eigvals, 2, subplotid, sharex=None, sharey=None)
    ax_imag = plt.subplot(N_eigvals, 2, subplotid+1, sharex=ax_real, sharey=None)
    for k in range(N_eigvals):
        
        for j in range(N_sigma):
            # labels
            if j in [0, N_sigma-1] and k==0:
                siglabel = '$\sigma={0:.3}$ [mV/$\sqrt{{\mathrm{{ms}}}}$]'.format(sigma[j])
            else:
                siglabel = None
                
            # color
            cm = plt.get_cmap(colormap) 
            cNorm  = colors.Normalize(vmin=sigma[0], vmax=sigma[-1])
            scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cm)
            linecolor = scalarMap.to_rgba(sigma[j])
            
            # eigenval: real part
            plt.subplot(N_eigvals, 2, subplotid, 
                        sharex=ax_real if subplotid > 1 else None, 
                        sharey=ax_real if subplotid > 1 else None)
            plt.plot(mu, lambda_all[k, :, j].real, label=siglabel, color=linecolor)
            if j==0:
                if k==0:
                    plt.title('real part')
                plt.ylim(np.amin(lambda_all.real), 0)
                plt.ylabel('$\Re(\lambda_{0}^\mathrm{{raw}})$ [kHz]'.format(k+1))
                if k==N_eigvals-1:
                    plt.xlabel('$\mu$ [mV/ms]')
            
            # eigenval: imag part
            plt.subplot(N_eigvals, 2, subplotid+1, 
                        sharex=ax_real, 
                        sharey=ax_imag if subplotid > 1 else None)
            plt.plot(mu, lambda_all[k, :, j].imag, color=linecolor)
            if j==0:
                if k==0:
                    plt.title('imag. part')
                plt.ylim(np.amin(lambda_all.imag), np.amax(lambda_all.imag))
                plt.ylabel('$\Im(\lambda_{0}^\mathrm{{raw}})$ [kHz]'.format(k+1))
                if k==N_eigvals-1:
                    plt.xlabel('$\mu$ [mV/s]')
    
        if k==0:
            plt.subplot(N_eigvals, 2, subplotid, 
                        sharex=ax_real if subplotid > 1 else None, 
                        sharey=ax_real if subplotid > 1 else None)
            plt.legend(loc='best')
        subplotid += 2
        

