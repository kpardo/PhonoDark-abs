'''
selfenergy.py
'''

from dataclasses import dataclass
import numpy as np
from pda.constants import *
from pda.material import Material
import pda.couplings as coup
import pda.transfer_matrix as tm
import pda.dielectric as d


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
    mixing: bool = False

    def __post_init__(self):
        if self.coupling == None:
            # set default coupling if none given as input
            self.coupling = coup.Scalar(q_XYZ_list=self.k)
        self.mat_sq = self.get_mat_sq()
        self.prop = self.get_propagator()
        self.se = self.get_se()

    def get_mat_sq(self):
        T = np.einsum('j, knja -> jnka', 1./np.sqrt(self.mat.atom_masses), np.conj(self.mat.bare_ph_eigen_o))
        Tstar = np.conj(T)

        # right = tm.TransferMatrix(nu=self.nu, k=self.k,
        #                           mat=self.mat, pol_mixing=self.pol_mixing, lam=self.lam,
        #                           ground_state='right').tm
        # left = tm.TransferMatrix(nu=self.nu, k=self.k,
        #                          mat=self.mat, pol_mixing=self.pol_mixing, lam=self.lam,
        #                          ground_state='left').tm
        # sum over nu and nu' --> left with q, lambda, a, b
        # matsq = np.einsum('ijklb, ijkla -> ikab', left, right)
        matsq = 1./3. * np.einsum('jnka, mnka -> jmnk', T, Tstar)
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
            energies = self.mat.bare_ph_energy_o[0]
            widths = self.width_val*np.ones((len(energies)))
            prop = (-1.*energies[:, np.newaxis]**2 + self.nu **
                     2 + 1j*widths[:, np.newaxis]*self.nu**2)**(-1)
        else:
            raise NotImplementedError
        # lorentz = self.L_func(self.nu, self.mat.energies[0], width_list)
        return prop

    def get_se(self):
        # tensor product with propagator
        # final matrix has indices q, lambda, a, b, omega=mass_DM
        totse = np.einsum('jmnk, nw -> jmkw',
                          self.coupling.prefac*self.mat_sq, self.prop)
        # dot in relevant vector, given coupling type
        if self.coupling.se_shape == 'scalar':
            se1 = np.einsum('jmkw, jwa, mwa -> kw', totse,
                           self.coupling.formfac, self.coupling.formfac)
        elif self.coupling.se_shape == 'scalar2':
            se1 = np.einsum('ikabj, ja, jb -> ikj', totse,
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

        elif self.coupling.se_shape == 'dim52':
            ## omega and q are in first term, w is in second
            se0 = np.einsum('ikabj, jia, jib -> ikj', totse,
                            self.coupling.formfaci0, self.coupling.formfaci0)
            sei = np.einsum('ikabj, jan, jbn -> ikjn', totse,
                            self.coupling.formfacij, self.coupling.formfacij)
            se1 = np.zeros(
                (len(self.k), len(self.mat.energies[0]), len(self.nu), 4), dtype=complex)
            se1[:, :, :, 0] = se0
            se1[:, :, :, 1:] = sei

        if self.mixing:
            se = self.mixing_contribution(se1)
        else:
            se = se1

        # final return has axes q, mat.energies[0], masslist=nu
        return se

    def mixing_contribution(self, se):
        '''
        Get contribution from SM photon mixing and returns the "mostly-DM" state
        see Eqn. 6 in draft.
        returns full SE, rather than just imaginary part. rate code takes imaginary part.
        '''
        # FIXME probably...this is a dummy version before we have full mixing matrices
        mixing_se = np.zeros((3,3, len(self.nu))) ## probably need to grab from coupling??
        piaa = self.get_photon_se()
        mixingdot = np.einsum('ijk, ijk -> ijk', mixing_se, mixing_se)
        fullmix =  mixingdot / (self.nu**2 - piaa)
        sums = np.einsum('ijk -> k', fullmix)
        final_se = se + sums
        print(final_se - se)
        return final_se

    def get_photon_se(self):
        '''
        \Pi_AA given by Eqn. 32 in the draft. But we can also just grab it from our dielectric code.
        '''
        piaa = d.Dielectric(mat=self.mat, mass=self.nu,
                width_type=self.width, width_val=self.width_val).piaa
        return piaa
