# fokker-planck-based-spike-rate-models

## Note: the code is currently in working state (beta version), a final release is expected in April 2017.

Implementations of spike rate models derived from networks of adaptive exponential integrate-and-fire models:

1. Numerical solution of the mean-field Fokker-Planck (FP) equation using a finite volume method with Scharfetter-Gummel flux approximation.
2. Low-dimensional ordinary differential equations (ODE) derived from the spectral decomposition of the FP operator. 
3. Low-dimensional ODE based on a cascade of two linear filters and a nonlinearity determined from the FP equation and semi-analytically fitted.

Furthermore: precalculation codes for the (look-up) quantities involved in (2) and (3).

References: Augustin, Ladenbauer, Baumann, Obermayer (under review)

_Code Usage_: change to folder `adex_comparison` and see the README.md file contained therein
