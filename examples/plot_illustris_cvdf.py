import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import torrey_cmf

tc = torrey_cmf.number_density()

redshift_list = np.arange(7)
n_bin   = 100
l_min_vd  = 1.8
l_max_vd  =  2.7
r = l_max_vd - l_min_vd

vd_array = np.arange(l_min_vd, l_max_vd, r / n_bin) 

fontsize=14
cm        = plt.get_cmap('nipy_spectral')

fig = plt.figure(figsize=(5,5))
ax = fig.add_subplot(1,1,1)
fig.subplots_adjust(left=0.16, right=0.99, top=0.99, bottom=0.12)
ax.set_yscale('log')
ax.set_xlim([l_min_vd, l_max_vd])
ax.set_ylim([3e-6, 2e-1])
ax.set_xlabel(r'$\mathrm{Log(\sigma_*\;[km/s])}$', fontsize=14)
ax.set_ylabel(r'N(>M) (Mpc${}^{-3}$)', fontsize=14)
for tick in ax.xaxis.get_major_ticks():
    tick.label.set_fontsize(fontsize)
for tick in ax.yaxis.get_major_ticks():
    tick.label.set_fontsize(fontsize)
 
for index,z in enumerate(redshift_list):
    color = cm(1.*index/(1.0*redshift_list.shape[0]))
    nd = 10.0**tc.cvdf_fit(vd_array, z)
    print (vd_array)
    print (nd)
    print ("\n\n\n")
    ax.plot( vd_array, nd,
		lw=2, label='z={:.1f}'.format(z))

ax.legend(loc=0, prop={'size':fontsize-3})
fig.savefig('./illustris_cvdf.pdf')

