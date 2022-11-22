'''
reach.py
'''
from dataclasses import dataclass
import numpy as np
from scipy import linalg as sla
import astropy.units as u
import new_physics as physics
import sys
from constants import *
from material import Material
import transfer_matrix as tm
import selfenergy as se
import rate as r

def reach(mass_list, q_XYZ_list, mat, n_pol_cut=3, exposure=1*KG_YR):
    rate = r.rate(mass_list, q_XYZ_list, mat)
    reach = np.sqrt(n_pol_cut / (rate * KG_YR))
    return (reach*(u.eV)**(-1)).to(1./u.GeV)
