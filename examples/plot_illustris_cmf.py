import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import torrey_cmf

tc = torrey_cmf.number_density()

redshift_list = np.arange(7)
n_m_bin   = 100
l_min_m  = 7.0
l_max_m  = 12.0
r = l_max_m - l_min_m

mass_array = np.arange(l_min_m, l_max_m, r / n_m_bin) 

fontsize=14
cm        = plt.get_cmap('nipy_spectral')

fig = plt.figure(figsize=(5,5))
ax = fig.add_subplot(1,1,1)
fig.subplots_adjust(left=0.16, right=0.99, top=0.99, bottom=0.12)
ax.set_yscale('log')
ax.set_xscale('log')
ax.set_xlim([10.0**l_min_m, 10.0**l_max_m])
ax.set_ylim([3e-6, 2e-1])
ax.set_xlabel(r'M${}_*$ (M${}_\odot$)', fontsize=14)
ax.set_ylabel(r'N(>M) (Mpc${}^{-3}$)', fontsize=14)
for tick in ax.xaxis.get_major_ticks():
    tick.label.set_fontsize(fontsize)
for tick in ax.yaxis.get_major_ticks():
    tick.label.set_fontsize(fontsize)
 
for index,z in enumerate(redshift_list):
    color = cm(1.*index/(1.0*redshift_list.shape[0]))
    nd = 10.0**tc.cmf_fit(mass_array, z)
    valid_range = (mass_array > 7.0) & (mass_array < 12.0) & (nd > 3e-5)
    ax.plot( 10.0**mass_array[valid_range], nd[valid_range],
		lw=2, label='z={:.1f}'.format(z))

ax.legend(loc=0, prop={'size':fontsize-3})
fig.savefig('./illustris_cmf.pdf')

