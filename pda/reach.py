'''
reach.py
'''
from dataclasses import dataclass
import numpy as np
from scipy import linalg as sla
import astropy.units as u
import sys
from constants import *
from material import Material
import transfer_matrix as tm
import selfenergy as se
import rate as r
import couplings as coup


def reach(mass_list, q_XYZ_list, mat, coupling=None, snr_cut=3,
          exposure=1*KG_YR, pol_mixing=False):
    if coupling == None:
        # set default coupling, if none given as input
        coupling = coup.Scalar(q_XYZ_list)
    rate = r.rate(mass_list, q_XYZ_list, mat,
                  coupling=coupling, pol_mixing=pol_mixing)
    reach = np.sqrt(snr_cut / (rate * exposure))
    return (reach*(u.eV)**(-1)).to(1./u.GeV)
