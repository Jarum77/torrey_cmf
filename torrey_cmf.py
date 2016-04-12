#!/usr/bin/env python
""" routines for describing the number density evolution of galaxy populations in Illustris.
    # Paul Torrey (ptorrey@mit.edu)
    """

__author__ = "Paul Torrey with contributions from Ryan McKinnon"
__copyright__ = "Copyright 2015, The Authors"
__credits__ = ["Paul Torrey and Ryan McKinnon"]
__license__ = "GPL"
__version__ = "1.0"
__maintainer__ = "Paul Torrey"
__email__ = "ptorrey@mit.edu"
__status__ = "Public Release.  v1.0."


import numpy as np
from scipy.optimize import newton


class number_density:
    def __init__(self):
        self._cmf_vars = np.array(
	[ [-2.89381061e+00,   8.21989048e-02,  -1.23157487e-01],  
	  [-6.25597878e-01,   8.62157470e-02,  -4.90327135e-02],  
	  [-3.88949868e-02,   2.54185320e-02,  -7.12953126e-03],   
	  [ 1.15238521e+01,  -1.87102380e-01,   2.10223127e-02] ] )

        self._dmf_vars = np.array(
    [ [-4.58622836,        0.32510541,      -0.00729362],
      [-0.85569723,       -0.11641107,       0.00950263],
      [ 0.00190046,       -0.01466030,       0.00173920],
      [14.14366307,       -0.60796535,       0.02010243]  ] )

        self._cvdf_vars = np.array(
    [ [ 7.39149763,        5.72940031,      -1.12055245],
      [-6.86339338,       -5.27327109,       1.10411386],
      [ 2.85208259,        1.25569600,      -0.28663846],
      [ 0.06703215,       -0.04868317,       0.00764841]  ] )

        self._cdmf_vars = np.array(
    [ [-5.08189685e+00,   1.55357858e-01,  -2.18816909e-02],
      [-1.05830010e+00,  -1.66413452e-01,   3.95287210e-03],
      [-2.11658380e-02,  -2.13713414e-02,   6.82334998e-04],
      [1.41924033e+01,   -5.14655812e-01,   2.22309520e-02]])
            
        self._vars_nc = np.array(
    [ [-2.64396065,       -0.29957949,      -0.03786140],
      [-0.52684351,       -0.13813571,      -0.03224572],
      [-0.02648208,       -0.01600591,      -0.00564523],
      [11.33927770,        0.39102542,      -0.01573175]  ])

        self._vars_nc_rev3 = np.array(
    [ [-4.35301221,       -0.43893418,       0.22963251],
      [-0.79280392,       -0.19135083,       0.06622329],
      [-0.02908882,       -0.02000640,       0.00739485],
      [11.80107352,        0.19790054,      -0.14974796]  ])

        self._vars_nc_rev2 = np.array(
    [ [-3.99168548,        0.34870581,       0.05096602],
      [-0.79382527,        0.01932935,       0.03692425],
      [-0.03829736,       -0.00023232,       0.00694828],
      [11.82842896,       -0.28090410,      -0.03059818]  ])

        self._vars_nc_rev_z1 = np.array(
    [ [-3.64009890,        1.03601036,      -0.31161953],
      [-0.79721518,        0.29299124,      -0.07746618],
      [-0.04838342,        0.03363121,      -0.00486779],
      [11.82759108,       -0.66505107,       0.18483682]  ])
    
        self._single_nd_fit = np.array(
     [ [-2.345933,   4.018919,   -2.562873,   0.538731],
       [-3.726408,   6.589452,   -4.041344,   0.812977],
       [-1.904525,   3.294253,   -1.936011,   0.374844],
       [-0.307381,   0.515786,   -0.290909,   0.054201],
      
       [7.103476,   -12.762439,   6.245308,   -0.987736],
       [10.465877,  -19.101222,   9.438636,   -1.488629],
       [4.687786,   -8.744238,    4.358233,   -0.686034],
       [0.658602,   -1.260553,    0.634731,   -0.099878],
      
       [-4.291523,   7.968271,   -3.905838,   0.591144],
       [-6.246956,  11.671404,   -5.756311,   0.873291],
       [-2.760469,   5.224862,   -2.596056,   0.395104],
       [-0.377230,   0.728651,   -0.365878,   0.055983]  )

    def singel_nd_fit(self, z, z0, init_N_tilda, target=0, **kwargs):
        """ Evaluate the forward number density evolution tracks for log_mass, z, and z0 """
        result=0
        A = np.zeros(4); B = np.zeros(4); C = np.zeros(4)
        for i in range(4):
            A[i] = np.sum( [this_vars[i+0][j] * z0**j] for j in range(4) ] )
            B[i] = np.sum( [this_vars[i+4][j] * z0**j] for j in range(4) ] )
            C[i] = np.sum( [this_vars[i+8][j] * z0**j] for j in range(4) ] )
      
        A = np.sum( [A[i] * init_N_tilda ** i  for i in range(4)    ]   )
        B = np.sum( [B[i] * init_N_tilda ** i  for i in range(4)    ]   )
        C = np.sum( [C[i] * init_N_tilda ** i  for i in range(4)    ]   )
    
        dz = (z0 - redshift)
        return init_N_tilda + A * dz + B * dz ** 2 + C * dz ** 3  - target
      
      
    def cmf_fit(self, log_mass, redshift, target=0, **kwargs):
        """ Evaluate the CMF at a given list of masses at some redshift"""
        if np.max(log_mass) > 20: warn_not_log_arg()
        return self._cmf_fit_func(log_mass, self._cmf_vars, redshift, target=target)

    def cvdf_fit(self, log_vd, redshift, target=0, **kwargs):
        """ Evaluate the CVDF at a given list of vd's at some redshift"""
        if np.max(log_vd) > 10: warn_not_log_arg()
        return self._cmf_fit_func(log_vd, self._cvdf_vars, redshift, target=target)

    def cdmf_fit(self, log_mass, redshift, target=0, **kwargs):
        """ Evaluate the dm CMF at a given list of masses at some redshift"""
        if np.max(log_mass) > 20: warn_not_log_arg()
        return self._cmf_fit_func(log_mass, self._cdmf_vars, redshift, target=target)

    def gsmf_fit(self, log_mass, redshift, target=0, **kwargs):
        """ Evaluate the GSMF at a given list of masses at some redshift"""
        if np.max(log_mass) > 20: warn_not_log_arg()
        return self._cmf_fit_func(log_mass, self._dmf_vars, redshift, target=target)

    def nc_cmf_fit(self, log_mass, redshift, target=0, z_init=0, **kwargs):
        """ If you feed this a **z_init*** stellar mass (log_mass), it will return the
            median number density of that galaxy population at some other redshift (redshift) """
        if np.max(log_mass) > 20:  warn_not_log_arg()
        if z_init == 0: this_vars = self._vars_nc
        if z_init == 1: this_vars = self._vars_nc_rev1
        if z_init == 2: this_vars = self._vars_nc_rev2
        if z_init == 3: this_vars = self._vars_nc_rev3
        return self._cmf_fit_func(log_mass, this_vars, redshift, target=target)
    
    def mass_from_density(self, cum_num_dens, redshift):
        """ Calculate the stellar mass from a cum num dens by inverting the CMF """
        args = (redshift, cum_num_dens)
        mass = newton(self.cmf_fit, 10.0, args=args)
        return mass

    def dm_mass_from_density(self, cum_num_dens, redshift):
        """ Calculate the dm mass from a cum num dens by inverting the dm CMF """
        args = (redshift, cum_num_dens)
        mass = newton(self.cdmf_fit, 10.0, args=args)
        return mass
    
    def vd_from_density(self, cum_num_dens, redshift):
        """ Calculate the vel disp from a cum num dens by inverting the CVDF """
        args = (redshift, cum_num_dens)
        try: val = newton(self.cvdf_fit, 2.0, args=args)
        except: val = -1
        return val

    def project_growth(self, arr, z1, z2, field='mass', nc=True):
        """ Project a galaxy (population's) growth from init value arr at z1 to z2 """
        arr = np.array([arr]).flatten()     # input masses or vds at z1
        res = np.zeros_like(arr)            # output masses or vds at z2
        
        if field == 'mass':
            from_dens = self.mass_from_density
            if nc:  arr = arr
            else:   to_dens = self.cmf_fit
        elif field == 'vd':
            from_dens = self.vd_from_density
            if nc:  arr = [self.mass_from_density(self.cvdf_fit(x, z1), z1) for x in arr]
            else:   to_dens = self.cvdf_fit
        elif field == 'dm':
            from_dens = self.dm_mass_from_density
            if nc:  arr = [self.mass_from_density(self.cdmf_fit(x, z1), z1) for x in arr]
            else:   to_dens = self.cdmf_fit

        if nc:
            to_dens = self.nc_cmf_fit

        for i, elem in enumerate(arr):
	    if nc:  proj_dens = to_dens(elem, z2, z_init=z1)
	    else:   proj_dens = to_dens(elem, z1)
            if proj_dens < np.log10(1e-5):  # you're below the fit limits
	 	print " "
                print " PROJECTING GROWTH BELOW FIT LIMITS! "
                proj_dens = np.log10(1e-5)
            res[i] = from_dens(proj_dens, z2)

        if res.shape[0] == 1:
                return res[0]
        else:
           return res

    def _cmf_fit_func(self, this_val, this_vars, redshift, target=0):
        """ Evaluate Equations 1 & 2-5 from Torrey+2015 """
        coeffs = [this_vars[i][0] + this_vars[i][1] * redshift +
              this_vars[i][2] * redshift**2 for i in range(4)]
        mstar = this_val - coeffs[3]
        return coeffs[0] + coeffs[1]*mstar + coeffs[2]*mstar**2 - np.exp(mstar) - target


def warn_not_log_arg():
    print " "
    print "   WARNING:  ARGUMENT DETECTED IN torrey_cmf.py THAT IS MUCH LARGER THAN EXPECTED"
    print "   WARNING:  VERIFY THAT YOU ARE USING LOG SCALE (AS REQUIRED) "
    print " "
