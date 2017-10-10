import tables
import numpy as np
import matplotlib.pyplot as plt


filename_old = 'quantities_spectral.h5'
filename_new = 'quantities_spectral_2reg.h5'


h5file = tables.open_file(filename_old, mode='r')
lambda_1_old = h5file.root.lambda_1.read()
lambda_2_old = h5file.root.lambda_2.read()
f_1_old = h5file.root.f_1.read()
f_2_old = h5file.root.f_2.read()
mu = h5file.root.mu.read()
sigma = h5file.root.sigma.read()
h5file.close()

h5file = tables.open_file(filename_new, mode='r')
lambda_all = h5file.root.lambda_reg_diff.read()
f = h5file.root.f.read()
f_1_new = f[0,:,:]
f_2_new = f[1,:,:]
lambda_1_new = lambda_all[0,:,:]
lambda_2_new = lambda_all[1,:,:]
h5file.close()


sigma_val=5
idx = np.argmin(np.abs(sigma-sigma_val))
plt.plot(mu,lambda_1_old[:, idx], label = 'old')
plt.plot(mu,lambda_2_old[:, idx], label = 'old')
plt.plot(mu,lambda_1_new[:, idx])
plt.plot(mu,lambda_2_new[:, idx])
plt.legend()
plt.figure()
plt.plot(mu,f_1_new[:, idx], '*', label='new')
plt.plot(mu,f_2_new[:, idx], '*', label='new')
plt.plot(mu,f_1_old[:, idx], label='old')
plt.plot(mu,f_2_old[:, idx], label='old')
plt.legend()
plt.show()

