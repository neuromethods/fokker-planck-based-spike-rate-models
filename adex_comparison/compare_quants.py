# compare the quantities which have been previously computed

import tables
import matplotlib.pyplot as plt
import numpy as np



filename_devel = 'quantities_spectral_nightsession.h5'
filename_master = 'quantities_spectral_master.h5'


# define mu, sigma



# we have two dictionaries to compare
h5file = tables.open_file(filename_master, mode='r')
d1 = {}
# extract stuff
for q in h5file.root:
    try:
        d1[q.name] = np.array(q)
    except:
        pass

h5file.close()

h5file = tables.open_file(filename_devel, mode='r')
# extract stuff
d2 = {}
for q in h5file.root:
    try:
        d2[q.name] = np.array(q)
    except:
        pass

h5file.close()


print(d2.keys())
# exit()

idx = 12


fig = plt.figure()
fig.suptitle('psi_r')
plt.plot(d1['mu'], d1['psi_r_1'][:,idx], label = 'master')
# plt.plot(master['mu'], master['psi_r_2'][:, 21], label = 'master')
# plt.plot(d2['mu'], d2['psi_r'][1, :, idx], label = 'devel')
plt.plot(d2['mu'], d2['psi_r'][0, :, idx], label = 'devel')
plt.legend()
#
# #
fig = plt.figure()
plt.suptitle('f')
plt.plot(d1['mu'], d1['f_1'][:,idx], label = 'master')
# plt.plot(master['mu'], master['psi_r_2'][:, 21], label = 'master')
plt.plot(d2['mu'], d2['f'][0, :, idx], label = 'devel')
plt.legend()

# some stationary quantities
fig = plt.figure()
plt.suptitle('r_inf')
plt.plot(d1['mu'], d1['r_inf'][:,idx], label = 'master')
# plt.plot(master['mu'], master['psi_r_2'][:, 21], label = 'master')
plt.plot(d2['mu'], d2['r_inf'][:, idx], label = 'devel')
plt.legend()
#
# c-quantities
fig = plt.figure()
plt.suptitle('c_mu')
plt.plot(d1['mu'], d1['c_mu_1'][:,idx], label = 'master')
plt.plot(d2['mu'], d2['c_mu'][0,:,idx], label = 'devel')
plt.legend()
#
fig = plt.figure()
plt.suptitle('c_sigma')
plt.plot(d1['mu'], d1['c_sigma_2'][:,idx], label = 'master')
plt.plot(d2['mu'], d2['c_sigma'][1,:,idx], label = 'devel')
plt.legend()
plt.show()