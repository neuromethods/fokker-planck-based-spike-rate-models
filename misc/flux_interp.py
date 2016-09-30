# here we validate the numerical discretization of the flux at the threshold voltage

import numpy as np
import matplotlib.pyplot as plt
import math


#TODO remove this finally if the flux state variable is used instead of interpolation

# the following function was copied to spectralsolver after testing here
# ==> (V, phi) are arrays of the discrete voltage and corresponding eigenfunction 
# values at those V[i].
# ==> sigma is the input std dev
# ==> order (1, 2 or 3) is the max. exponent of the interpolatoin polynomial
# through the last m+1 points (V[i], phi(V[i])) 
# note we assume a uniform V grid here
def thresholdflux(V, phi, sigma, order):
    
    dV = V[-1]-V[-2] # assume uniform grid -- generalization straight but more coeffs.

    if order == 1:
        phi_deriv_thr = -phi[-2]/dV
        
    elif order == 2:
        phi_deriv_thr = (-2.*phi[-2] + phi[-3]/2. )/dV
        
    elif order == 3:
        phi_deriv_thr = (-3.*phi[-2] + 3./2.*phi[-3] - 1./3.*phi[-4])/dV
        
    else:
        print('order {} is not supported -- only 1, 2 or 3'.format(order))
        exit()
        
    flux = -sigma**2 * phi_deriv_thr
    return flux
    
    
    
if __name__ == '__main__':
    V_fine = np.linspace(0,1,5000)
    V_coarse = np.linspace(0,1,500)
    
    
    # example: phi_inf for PIF with lower bound from Mattia and Del Giudice 2002 PRE
    mu = 40. # here unit 1 per second
    sigma = 0.5 # here unit 1 per sqrt(sec)
    xi = mu/sigma**2
    c = 1./(sigma**2/(2*mu**2) * (2*mu/sigma**2 - 1 + math.exp(-2*mu/sigma**2)))
    phi = lambda V: c/mu * (1-np.exp(-2*xi*(1-V)))
    phi_fine = phi(V_fine)
    phi_coarse = phi(V_coarse)
    
    plt.figure()
    plt.title('Approximations of the threshold flux: \nderivative at $V_s$ of up to 3rd order interpolation polynomials')
    plt.plot(V_fine, phi_fine, color='gray', label='$p_\infty$ (PIF)')
    plt.plot(V_coarse[-4:], phi_coarse[-4:], 'ok', ms=7)
    
    flux_1 = thresholdflux(V_coarse, phi_coarse, sigma, order=1)
    flux_2 = thresholdflux(V_coarse, phi_coarse, sigma, order=2)
    flux_3 = thresholdflux(V_coarse, phi_coarse, sigma, order=3)
    phi_approx_thr = lambda fl: fl*(-1.0/sigma**2)*(V_fine-1) 
    plt.plot(V_fine, phi_approx_thr(flux_1), color='orange', label='linear')
    plt.plot(V_fine, phi_approx_thr(flux_2), color='green', label='quadratic')
    plt.plot(V_fine, phi_approx_thr(flux_3), color='blue', label='cubic')

    dV = V_coarse[-1]-V_coarse[-2]
    # the following expression is simply the quadratic interpolation polynomial 
    # in lagrangian formulation
    quadr_interp = phi_coarse[-2] * (V_fine-V_coarse[-1])/(-dV) * (V_fine-V_coarse[-3])/dV \
                 + phi_coarse[-3] * (V_fine-V_coarse[-1])/(-2*dV) * (V_fine-V_coarse[-2])/(-dV)
    plt.plot(V_fine, quadr_interp, '--', color='green')

    cubic_interp = (V_fine-V_coarse[-1])/dV**3 * \
        (-phi_coarse[-2]/2. * (V_fine-V_coarse[-3]) * (V_fine-V_coarse[-4]) \
         +phi_coarse[-3]/2. * (V_fine-V_coarse[-2]) * (V_fine-V_coarse[-4]) \
         -phi_coarse[-4]/6. * (V_fine-V_coarse[-2]) * (V_fine-V_coarse[-3]))
    plt.plot(V_fine, cubic_interp, '--', color='blue')

    plt.ylim(0, max(1.2, max(phi_fine)))
    plt.xlim(0.988,1.001)
    plt.xlabel('voltage')
    plt.ylabel('density')
    
    plt.legend(loc='best')
    plt.show()
    
