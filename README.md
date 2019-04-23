# fokker-planck-based-spike-rate-models

Implementations of spike rate models derived from networks of adaptive exponential integrate-and-fire models, as described in:
__Augustin*, Ladenbauer*, Baumann, Obermayer,__ ___Low-dimensional spike rate models derived from networks of adaptive integrate-and-fire neurons: comparison and implementation,___ [PLOS Computational Biology 2017](https://doi.org/10.1371/journal.pcbi.1005545) 


1. Numerical solution of the mean-field Fokker-Planck (FP) equation using a finite volume method with Scharfetter-Gummel flux approximation.
2. Low-dimensional ordinary differential equations (ODE) derived from the spectral decomposition of the FP operator. 
3. Low-dimensional ODE based on a cascade of linear filters and a nonlinearity determined from the FP equation.

Furthermore: precalculation codes for the (look-up) quantities involved in (2) and (3).

For questions please contact us: Moritz Augustin and Josef Ladenbauer

_Code Usage_: change to folder `adex_comparison` and see the README.md file contained therein
