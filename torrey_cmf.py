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
__email__ = "ptorrey@mit.harvard.edu"
__status__ = "Public Release.  v1.0."


import numpy as np
from scipy.optimize import newton


class number_density:
    def __init__(self):
        self._cmf_vars = np.array(
        [   [-4.02977287e+00,   1.76267393e-01,  -1.44941371e-01],
            [-1.05309169e+00,   1.17135834e-01,  -5.40660745e-02],
            [-8.58905816e-02,   2.52993131e-02,  -6.90611852e-03],
            [1.20106042e+01,   -2.34838939e-01,   2.52957319e-02]])
         

        self._cvdf_vars = np.array(
        [   [-2.34444132,  0.24249917, -0.10259731],
            [-1.47270662,  0.54513093, -0.22659028],
            [0.44307961,  1.21992189, -0.28089004],
            [2.3837376,  -0.04416525,  0.00729029]])

        self._cdmf_vars = np.array(
        [   [-5.08189685e+00,   1.55357858e-01,  -2.18816909e-02],
            [-1.05830010e+00,  -1.66413452e-01,   3.95287210e-03],
            [-2.11658380e-02,  -2.13713414e-02,   6.82334998e-04],
            [1.41924033e+01,  -5.14655812e-01,   2.22309520e-02]])
            
        self._vars_nc = np.array(
         [  [ -3.99510528e+00,   2.04772606e-01,  -1.43642170e-01],
            [-1.02669347e+00,    2.55361970e-02,  -6.16581730e-02],
            [-8.18861264e-02,   1.86254043e-03,   -8.24157788e-03],
            [1.20001548e+01,   3.51709332e-03,   6.84297177e-02]])




        self._vars_nc_rev3 = np.array(
        [ [ -4.40077515e+00,  -6.62145255e-01,   1.91921627e-01],
          [-8.82295647e-01,   -2.33066595e-01,   4.70923783e-02],
          [-3.97308012e-02,  -2.20131731e-02,    4.09743368e-03],
           [1.15256908e+01,   3.78107550e-01,    -1.37326740e-01]])

        self._vars_nc_rev2 = np.array(
        [ [ -4.19858527e+00,  -4.49363598e-01,   2.29778257e-01],
          [ -9.23353873e-01,   -1.83275908e-01,   6.85051354e-02],
          [ -5.38416102e-02,  -1.72440142e-02,    6.82696341e-03],
          [ 1.16729458e+01,   2.72053797e-01,  -1.56547865e-01]])

        self._vars_nc_rev_z1 = np.array(
        [ [-4.17225261e+00,   1.44894696e-01,   2.88689730e-02],
          [-1.02895757e+00,   3.94126650e-02,  -1.95460326e-03],
          [-7.53027534e-02,   9.59023782e-03,  -2.04448868e-03],
          [1.18825325e+01,  -2.53438989e-02,  -5.30882019e-02]])


    def cmf_fit(self, log_mass, redshift, target=0):
        """ Evaluate the CMF at a given list of masses at some redshift"""
        if np.max(log_mass) > 20: warn_not_log_arg()
        return self._cmf_fit_func(log_mass, self._cmf_vars, redshift, target=target)

    def cvdf_fit(self, log_vd, redshift, target=0):
        """ Evaluate the CVDF at a given list of vd's at some redshift"""
        if np.max(log_vd) > 10: warn_not_log_arg()
        return self._cmf_fit_func(log_vd, self._cvdf_vars, redshift, target=target)

    def cdmf_fit(self, log_mass, redshift, target=0):
        """ Evaluate the dm CMF at a given list of masses at some redshift"""
        if np.max(log_mass) > 20: warn_not_log_arg()
        return self._cmf_fit_func(log_mass, self._cdmf_vars, redshift, target=target)

    def nc_cmf_fit(self, log_mass, redshift, target=0, z_init=0):
        """ If you feed this a **z_init*** stellar mass, it will return the
            effective number density of that galaxy population at some other redshift """
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
            z1 = z2

        for i, elem in enumerate(arr):
            proj_dens = to_dens(elem, z1)
            if proj_dens < np.log10(3e-5):  # you're below the fit limits
                print " PROJECTING GROWTH BELOW FIT LIMITS! "
                proj_dens = np.log10(3e-5)
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
        return coeffs[0] + coeffs[1]*mstar + coeffs[2]*mstar**2 - 10.0**mstar - target




def warn_not_log_arg():
    print " "
    print "   WARNING:  ARGUMENT DETECTED IN torrey_cmf.py THAT IS MUCH LARGER THAN EXPECTED"
    print "   WARNING:  VERIFY THAT YOU ARE USING LOG SCALE (AS REQUIRED) "
    print " "
