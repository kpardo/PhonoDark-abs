'''
reach.py
'''
from dataclasses import dataclass
import numpy as np
from scipy import linalg as sla
import astropy.units as u
import sys
from pda.constants import *
from pda.material import Material
import pda.selfenergy as se
import pda.rate as r
import pda.couplings as coup
import pda.new_physics as physics


def reach(mass_list, mat, q_XYZ_list=None, coupling=None, snr_cut=3,
          exposure=1*KG_YR):
    if q_XYZ_list == None:
        # set default q mesh if none given as input
        q_XYZ_list = physics.generate_q_mesh(10, 5, 5)
    if coupling == None:
        # set default coupling, if none given as input
        coup.ScalarE(omega=mass_list, mat=mat, mixing=True)
    rate = r.rate(mass_list, mat, q_XYZ_list=q_XYZ_list,
                  coupling=coupling)
    reach = np.sqrt(snr_cut / (rate * exposure))
    ## dimensionless
    return reach
