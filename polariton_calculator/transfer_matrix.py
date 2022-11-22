'''
transfer_matrix.py
'''

from dataclasses import dataclass
import numpy as np
from scipy import linalg as sla
import physics
import sys
from constants import *
from material import Material

@dataclass
class TransferMatrix:
    nu: np.ndarray
    k: np.ndarray
    j: np.ndarray
    mat: Material
    pol_mixing: bool = False ## turn off pol mixing for now.
    lam: str = 'vi' ## default is velocity independent
    ground_state: str = 'right' ## default is to calculate matrix from 0 to v,k

    def __post_init__(self):
        ## make container for TM
        ## get all ingredients for calc.
        ## get the transfer matrix
        ## save to file.
        self.fn = f'''transfer_{self.mat}_{self.lam}_{self.ground_state}
                    _{self.pol_mixing}.dat'''
        self.tm = self.get_transfer()
        return 0

    def get_uv(self):
        pass

    def get_mass_energy_term(self):
        return np.sqrt(2. * self.mat.m_cell * self.mat.energies)**(-1.)

    def get_transfer(self):
        me = self.get_mass_energy_term()
        if self.pol_mixing:
            UV = self.get_uv()
            print("Not implemented yet!")
            sys.exit()
        else:
            if self.ground_state == 'right':
                dot = np.dot(self.k, np.conj(self.mat.dielectric))
            elif self.ground_state == 'left':
                dot = np.dot(self.k, self.mat.dielectric)
            tm = np.einsum('ij, ik -> ijk', 1j*me, dot)
        return tm
