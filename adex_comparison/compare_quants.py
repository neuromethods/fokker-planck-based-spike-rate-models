# compare the quantities which have been previously computed

import tables
import matplotlib.pyplot as plt
import numpy as np



filename_devel = 'quantities_spectral.h5'
filename_master = 'quantities_spectral_master.h5'


# define mu, sigma




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



plt.plot(d1['mu'], d1['f_1'][:, 20], label = 'master')
# plt.plot(master['mu'], master['psi_r_2'][:, 21], label = 'master')
plt.plot(d2['mu'], d2['f'][1, :, 0], label = 'devel')
# plt.plot(devel['mu'], devel['psi_r'][1, :, 1], label = 'devel')
plt.legend()
plt.show()
