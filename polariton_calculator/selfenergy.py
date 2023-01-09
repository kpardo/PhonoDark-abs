'''
selfenergy.py
'''

from dataclasses import dataclass
import numpy as np
from constants import *
from material import Material
import couplings as coup
import transfer_matrix as tm


@dataclass(kw_only=True)
class SelfEnergy:
    nu: np.ndarray
    k: np.ndarray
    mat: Material
    coupling: None
    lam: str
    pol_mixing: bool = True
    width: str = 'proportional'

    def __post_init__(self):
        if self.coupling == None:
            # set default coupling if none given as input
            self.coupling = coup.Scalar(q_XYZ_list=self.k)
        self.mat_sq = self.get_mat_sq()
        self.prop = self.get_propagator()
        self.se = self.get_se()

    def get_mat_sq(self):
        right = tm.TransferMatrix(nu=self.nu, k=self.k,
                                  mat=self.mat, pol_mixing=self.pol_mixing, lam=self.lam,
                                  ground_state='right').tm
        left = tm.TransferMatrix(nu=self.nu, k=self.k,
                                 mat=self.mat, pol_mixing=self.pol_mixing, lam=self.lam,
                                 ground_state='left').tm
        # sum over nu and nu' --> left with q, lambda, a, b
        matsq = np.einsum('ijklb, ijkla -> ikab', left, right)
        return matsq

    def L_func(self, omega, omega_0, width):
        '''
        lorentzian
        '''
        return (
            4.0*omega[:, np.newaxis]*omega_0*width *
            ((omega[:, np.newaxis]**2 - omega_0**2) **
             2 + (omega[:, np.newaxis]*width)**2)**(-1)
        )

    def get_propagator(self):
        if self.width == 'proportional':
            width_list = 10**(-3)*np.ones((len(self.mat.energies[0])))
        else:
            raise NotImplementedError
        lorentz = self.L_func(self.nu, self.mat.energies[0], width_list)
        return lorentz

    def get_se(self):
        totse = np.einsum('ikab, jk -> ikabj', 1j *
                          self.coupling.prefac*self.mat_sq, self.prop)
        # dot in relevant vector, given coupling type
        if self.pol_mixing:
            se = np.einsum('ikabj, ia, ib -> ikj', totse,
                           self.coupling.formfac, self.coupling.formfac)
        else:
            # FIXME
            raise NotImplementedError
        # final return has axes q, mat.energies[0], masslist=nu
        return se
