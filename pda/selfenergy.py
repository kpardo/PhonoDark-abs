'''
selfenergy.py
'''

from dataclasses import dataclass
import numpy as np
from pda.constants import *
from pda.material import Material
import pda.couplings as coup
import pda.transfer_matrix as tm


@dataclass(kw_only=True)
class SelfEnergy:
    nu: np.ndarray
    k: np.ndarray
    mat: Material
    coupling: None
    lam: str
    pol_mixing: bool = True
    width: str = 'best'
    width_val: float = 10**(-3)

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
        try:
            return (
                4.0*width[:, np.newaxis]*(omega[:, np.newaxis])*omega_0 *
                ((omega[:, np.newaxis]**2 - omega_0**2) **
                 2 + (width[:, np.newaxis]*omega[:, np.newaxis])**2)**(-1)
            )
        except:
            return (
                4.0*(omega[:, np.newaxis])*width*omega_0 *
                ((omega[:, np.newaxis]**2 - omega_0**2) **
                 2 + (omega[:, np.newaxis]*width)**2)**(-1)
            )

    def get_propagator(self):
        if self.width == 'constant':
            # default was 10**(-3)
            width_list = self.width_val*np.ones((len(self.mat.energies[0])))
        elif self.width == 'proportional':
            # default was 10**(-2)
            width_list = self.width_val*self.mat.energies[0]
            # width_list = self.width_val*self.nu
        elif self.width == 'best':
            width_list = self.width_val*self.nu**2
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
        if self.coupling.se_shape == 'scalar':
            se = np.einsum('ikabj, ia, ib -> ikj', totse,
                           self.coupling.formfac, self.coupling.formfac)
        elif self.coupling.se_shape == 'vector':
            se0 = np.einsum('ikabj, ia, ib -> ikj', totse,
                            self.coupling.formfaci0, self.coupling.formfaci0)
            sei = np.einsum('ikabj, jan, jbn -> ikjn', totse,
                            self.coupling.formfacij, self.coupling.formfacij)
            se1 = np.zeros(
                (len(self.k), len(self.mat.energies[0]), len(self.nu), 4), dtype=complex)
            se1[:, :, :, 0] = se0
            se1[:, :, :, 1:] = sei

            # se = self.mixing_contribution(se1)
            # FIXME
            se = se1
        elif self.coupling.se_shape == 'vector2':
            ## omega is in first term, q is in second
            se0 = np.einsum('ikabj, ja, jb -> ikj', totse,
                            self.coupling.formfaci0, self.coupling.formfaci0)
            sei = np.einsum('ikabj, ian, ibn -> ikjn', totse,
                            self.coupling.formfacij, self.coupling.formfacij)
            se1 = np.zeros(
                (len(self.k), len(self.mat.energies[0]), len(self.nu), 4), dtype=complex)
            se1[:, :, :, 0] = se0
            se1[:, :, :, 1:] = sei

            # se = self.mixing_contribution(se1)
            # FIXME
            se = se1
        elif self.coupling.se_shape == 'dim5':
            ## q is in first term, w and q are in second
            se0 = np.einsum('ikabj, ia, ib -> ikj', totse,
                            self.coupling.formfaci0, self.coupling.formfaci0)
            sei = np.einsum('ikabj, jian, jibn -> ikjn', totse,
                            self.coupling.formfacij, self.coupling.formfacij)
            se1 = np.zeros(
                (len(self.k), len(self.mat.energies[0]), len(self.nu), 4), dtype=complex)
            se1[:, :, :, 0] = se0
            se1[:, :, :, 1:] = sei

            # se = self.mixing_contribution(se1)
            # FIXME
            se = se1

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
