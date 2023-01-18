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
        # tensor product with propagator
        # final matrix has indices q, lambda, a, b, omega=mass_DM
        totse = np.einsum('ikab, jk -> ikabj', 1j *
                          self.coupling.prefac*self.mat_sq, self.prop)
        # dot in relevant vector, given coupling type
        if self.pol_mixing:
            if self.coupling.se_shape == 'scalar':
                se = np.einsum('ikabj, ia, ib -> ikj', totse,
                               self.coupling.formfac, self.coupling.formfac)
            else:
                se0 = np.einsum('ikabj, ia, ib -> ikj', totse,
                                self.coupling.formfaci0, self.coupling.formfaci0)
                sei = np.einsum('ikabj, jan, jbn -> ikjn', totse,
                                self.coupling.formfacij, self.coupling.formfacij)
                se1 = np.zeros(
                    (len(self.k), len(self.mat.energies[0]), len(self.nu), 4), dtype=np.complex)
                se1[:, :, :, 0] = se0
                se1[:, :, :, 1:] = sei

                se = self.mixing_contribution(se1)

        else:
            # FIXME
            raise NotImplementedError
        # final return has axes q, mat.energies[0], masslist=nu
        return se

    def mixing_contribution(self, se):
        '''
        Get contribution from SM photon mixing and returns the "mostly-DM" state
        see Eqn. 18 in draft.
        '''
        # FIXME
        mixing_se = 0.
        photon_se = self.get_photon_se()
        print(np.shape(photon_se))
        print(np.shape(se))
        sums = mixing_se**2 / (self.nu**2 - photon_se)
        final_se = se + sums
        return final_se

    def get_photon_se(self):
        coup2 = coup.Scalar(q_XYZ_list=self.k)
        totpse = np.einsum('ikab, jk -> ikabj', 1j *
                           coup2.prefac*self.mat_sq, self.prop)
        pse = np.einsum('ikabj, ia, ib -> ikj', totpse,
                        coup2.formfac, coup2.formfac)
        return pse
