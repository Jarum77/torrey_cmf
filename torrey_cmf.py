#!/usr/bin/env python
""" routines describing the number density evolution of galaxy populations in Illustris. """

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

        self._dm_mf_vars = np.array(		# this was in github incorrectly as dmf -- should be dm mass function ...
        [ [-4.58622836,        0.32510541,      -0.00729362],
          [-0.85569723,       -0.11641107,       0.00950263],
          [ 0.00190046,       -0.01466030,       0.00173920],
          [14.14366307,       -0.60796535,       0.02010243]  ] )

#        self._dmf_vars = np.array(
#       [  [-3.44790213,        0.14230995,      -0.11932847],
#          [-0.90191429,        0.10964059,      -0.05106220],
#          [-0.08777999,        0.03218344,      -0.00801054],
#          [11.72126779,       -0.24683298,       0.02411400]  ] )

        self._dmf_vars = np.array(
        [  [-2.62477122,        0.08536494,      -0.11269835],
           [-0.51284616,        0.09404466,      -0.04949364],
           [-0.03864963,        0.03270023,      -0.00815488],
           [11.54279153,       -0.19945507,       0.01974203]  ] )

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

        self._mil_cdmf_vars = np.array(
        [ [     -4.44138264,        0.35711012,      -0.01111473],
          [     -0.72959257,       -0.09591461,       0.00593439],
          [      0.02103888,       -0.00868751,       0.00081672],
          [     14.32909807,       -0.61663087,       0.02347729]  ] )


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

        self._vars_nc_rev1 = np.array(
        [ [-3.64009890,        1.03601036,      -0.31161953],
          [-0.79721518,        0.29299124,      -0.07746618],
          [-0.04838342,        0.03363121,      -0.00486779],
          [11.82759108,       -0.66505107,       0.18483682]  ])
    
        self._single_nd_fit = np.array(
        [ [-6.616925,   4.772104,   -0.620776,   ],
          [-12.553859,   10.062380,   -1.593516, ],
          [-8.490627,   7.305744,   -1.297247,   ],
          [-2.393108,   2.170882,   -0.415974,   ],
          [-0.237254,   0.224544,   -0.045412,   ],
          [20.215976,   -19.237827,   3.786527,  ],
          [35.822739,   -34.933411,   7.113855,  ],
          [22.317927,   -22.302322,   4.677395,  ],
          [5.845230,   -5.964089,   1.282028,    ],
          [0.546387,   -0.566171,   0.124111,    ],
          [-12.516574,   11.750153,   -2.437514, ],
          [-21.967282,   20.806088,   -4.367866, ],
          [-13.612419,   13.014453,   -2.763500, ],
          [-3.562422,   3.431866,   -0.735922,   ],
          [-0.334570,   0.323606,   -0.069900,   ], ]
        )

        self._mil_single_nd_fit = np.array(
           [1.648593,   -3.732841,   0.559748, ],
           [3.922858,   -7.594945,   1.323168, ],
           [2.872717,   -5.333279,   1.028583, ],
           [0.875170,   -1.584895,   0.328197, ],
           [0.094326,   -0.167541,   0.036489, ],
           [0.288443,   3.046505,   -0.626692, ],
           [-1.102782,   6.973642,   -1.543784,],
           [-1.548751,   5.266007,   -1.215838,],
           [-0.624181,   1.636277,   -0.388549,],
           [-0.077724,   0.177261,   -0.042935,],
           [-0.553139,   -0.949717,   0.306548,],
           [0.010697,   -2.530025,   0.734459, ],
           [0.543777,   -2.073177,   0.572898, ],
           [0.285440,   -0.683686,   0.183176, ],
           [0.039872,   -0.077457,   0.020351, ],  )

        self._sigma_forward_fit = np.array( 
	   [ 0.30753811, -0.35706088,  0.07811929,
         0.11658773, -0.36310229,  0.0878064,
         0.05432019, -0.08567579,  0.01938756,
         0.3410485 , -0.26566291,  0.05867066,
         0.40404559, -0.27861255,  0.05734425,
         0.06319481, -0.04605944,  0.01022102] )
 
        self._surv_forward_fit = np.array(
        [-0.57610316,  0.25395473, -0.06909108,
         -0.32306881,  0.30592045, -0.0841637,
         -0.0588044 ,  0.07159256, -0.01935816,
          0.34438256, -0.21727332,  0.04128673,
          0.29859003, -0.21186113,  0.04228161,
          0.03889436, -0.03177222,  0.00690157] )

        self._nd_backward_fit = np.array(
        [-0.39274648, -0.47308899, -0.09720173,
          0.18875905,  0.16531384,  0.03451216] )

        self._sigma_backward_fit = np.array(
        [ 0.00961576, -0.06880463,  0.00405991,
         -0.01970638, -0.01135986, -0.00539835] )




    def single_nd_fit(self, z, z0, init_N_tilde, target=0, verbose=False, type='IllustrisCMF', **kwargs):
        """ Evaluate the forward number density evolution tracks for log_mass, z, and z0 """
        if (init_N_tilde>-1) or (init_N_tilde<-6.5):
            if verbose:  print "out of range"
            return init_N_tilde
    
        if type=='IllustrisCMF':
            this_vars=self._single_nd_fit
        elif type=='MillenniumCMF':
            this_vars=self._mil_single_nd_fit

        result=0
        n_z0_exp=3; n_N0_exp = 5
        A = np.zeros(n_N0_exp); B = np.zeros(n_N0_exp); C = np.zeros(n_N0_exp)
        for i in range(n_N0_exp):
            A[i] = np.sum( [this_vars[i+0*n_N0_exp][j] * z0**j for j in range(n_z0_exp) ] )
            B[i] = np.sum( [this_vars[i+1*n_N0_exp][j] * z0**j for j in range(n_z0_exp) ] )
            C[i] = np.sum( [this_vars[i+2*n_N0_exp][j] * z0**j for j in range(n_z0_exp) ] )

        A = np.sum( [A[i] * init_N_tilde ** i  for i in range(n_N0_exp)    ]   )
        B = np.sum( [B[i] * init_N_tilde ** i  for i in range(n_N0_exp)    ]   )
        C = np.sum( [C[i] * init_N_tilde ** i  for i in range(n_N0_exp)    ]   )
    
        dz = (z0 - z)
        return init_N_tilde + A * dz + B * dz ** 2 + C * dz ** 3  - target

    def sigma_forward_fit(self, z, z0, init_N_tilde, sigma_0=0.00, verbose=False):
        """ Evaluate scatter in forward ND evolution tracks """
        if (init_N_tilde>-1) or (init_N_tilde<-5.5):
            if verbose:  print "out of range"
            return sigma_0

        this_vars = self._sigma_forward_fit
	A_exp = [  np.sum(  [coeff * z0**iii for iii, coeff in enumerate(this_vars[ 0+3*jjj:0+3*(jjj+1) ]) ]  ) for jjj in range(3)  ]
        B_exp = [  np.sum(  [coeff * z0**iii for iii, coeff in enumerate(this_vars[ 9+3*jjj:9+3*(jjj+1) ]) ]  ) for jjj in range(3)  ]
	A = np.sum( [A_exp[iii] * init_N_tilde**iii for iii in range(3)  ] )
        B = np.sum( [B_exp[iii] * init_N_tilde**iii for iii in range(3)  ] )
	dz = (z0 - z)
 	return sigma_0 + A*dz + B*dz**2

    def surv_forward_fit(self, z, z0, init_N_tilde, sigma_0=0.00, verbose=False):
        """ Evaluate survival fraction of galaxies as a function of elapsed time """
        if (init_N_tilde>-1) or (init_N_tilde<-5.5):
            if verbose:  print "out of range"
            return 0.0
        this_vars = self._surv_forward_fit
        A_exp = [  np.sum(  [coeff * z0**iii for iii, coeff in enumerate(this_vars[ 0+3*jjj:0+3*(jjj+1) ]) ]  ) for jjj in range(3)  ]
        B_exp = [  np.sum(  [coeff * z0**iii for iii, coeff in enumerate(this_vars[ 9+3*jjj:9+3*(jjj+1) ]) ]  ) for jjj in range(3)  ]
        A = np.sum( [A_exp[iii] * init_N_tilde**iii for iii in range(3)  ] )
        B = np.sum( [B_exp[iii] * init_N_tilde**iii for iii in range(3)  ] )
        dz = (z0 - z)
        return 1.0 + A*dz + B*dz**2

    def nd_backward_fit( self, z, init_N_tilde ):
        this_vars = self._nd_backward_fit
        A_exp = np.sum( [this_vars[iii+0] * init_N_tilde **iii for iii in range(3) ] )
        B_exp = np.sum( [this_vars[iii+3] * init_N_tilde **iii for iii in range(3) ] )
        return init_N_tilde + A_exp * z + B_exp * z**2
     
    def sigma_backward_fit( self, z, init_N_tilde, sigma_0=0.0 ):
        this_vars = self._sigma_backward_fit
        A_exp = np.sum( [this_vars[iii+0] * init_N_tilde **iii for iii in range(3) ] )
        B_exp = np.sum( [this_vars[iii+3] * init_N_tilde **iii for iii in range(3) ] )
        return sigma_0 + A_exp * z + B_exp * z**2
 
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

    def mil_cdmf_fit(self, log_mass, redshift, target=0, **kwargs):
        """ Evaluate the dm CMF at a given list of masses at some redshift"""
        if np.max(log_mass) > 20: warn_not_log_arg()
        return self._cmf_fit_func(log_mass, self._mil_cdmf_vars, redshift, target=target)

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
    
    def mass_from_density(self, cum_num_dens, redshift, type='IllustrisCMF'):
        """ Calculate the stellar mass from a cum num dens by inverting the CMF """
        args = (redshift, cum_num_dens)
        if type=='IllustrisCMF':
            mass = newton(self.cmf_fit, 10.0, args=args)
        elif type=='MillenniumCDMF':
            mass = newton(self.mil_cdmf_fit, 10.0, args=args)

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
