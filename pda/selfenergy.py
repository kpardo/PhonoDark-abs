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
    width_val: float = 10**(-2)

    def __post_init__(self):
        if self.coupling == None:
            # set default coupling if none given as input
            self.coupling = coup.Scalar(q_XYZ_list=self.k)
        if self.coupling.mixing:
            self.mixing = True
        else:
            self.mixing = False
        self.mat_sq = self.get_mat_sq()
        self.prop = self.get_propagator()
        self.se = self.get_se()

    def get_mat_sq(self):
        T = np.einsum('j, knja -> jnka', 1./np.sqrt(self.mat.atom_masses), np.conj(self.mat.bare_ph_eigen_o))
        Tstar = np.conj(T)
        matsq = 1./3. * np.einsum('jnka, mnka -> jmnk', T, Tstar)
        return matsq / self.mat.unit_cell_volume

    def get_propagator(self):
        energies = self.mat.bare_ph_energy_o[0]
        if self.width == 'constant':
            width_list = self.width_val*np.ones((len(energies)))
            prop = 1j * (-1.*energies[:, np.newaxis]**2 + self.nu **
                     2 + 1j*widths[:, np.newaxis]*self.nu)**(-1)
        elif self.width == 'best':
            widths = self.width_val*energies
            prop = 1j * (-1.*energies[:, np.newaxis]**2 + self.nu **
                     2 + 1j*widths[:, np.newaxis]*self.nu)**(-1)
        else:
            raise NotImplementedError
        return prop

    def get_se(self):
        # tensor product with propagator
        # final matrix has indices q, lambda, a, b, omega=mass_DM
        self.totse = -1j*np.einsum('jmnk, nw -> jmkw',
                          self.coupling.prefac*self.mat_sq, self.prop)
        # dot in relevant vector, given coupling type
        if self.coupling.se_shape == 'scalar':
            se1 = np.einsum('jmkw, jwka, mwka -> kw', self.totse,
                           self.coupling.formfac, np.conj(self.coupling.formfac))

        elif self.coupling.se_shape == 'vector':
            se1 = np.einsum('jmkw, jwab, mwab -> kw', 
                         totse, 
                         self.coupling.formfac, 
                         np.conj(self.coupling.formfac))

        elif self.coupling.se_shape == 'dim5':
            se1 = np.einsum('jmkw, jwabk, mwabk -> kw', 
                         totse, 
                         self.coupling.formfac, 
                         np.conj(self.coupling.formfac))

        elif self.coupling.se_shape == 'dim5_s':
            se1 = np.einsum('jmkw, jwabi, mwabi -> kw', 
                         totse, 
                         self.coupling.formfac, 
                         np.conj(self.coupling.formfac))

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
        piaa = self.get_photon_se()
        pi_phi_a_ph = np.einsum('jmkw, jwka, wmba -> wkb', 
                            self.totse, 
                            self.coupling.formfac, 
                            np.conj(self.formfacAA))
        pi_phi_ph_a = np.einsum('jmkw, wjab, mwkb -> wka', 
                            self.totse, 
                            self.formfacAA,
                            np.conj(self.coupling.formfac))
        pi_mix_sq = np.einsum('wkb, wkb -> wk', pi_phi_a_ph + self.coupling.mixing_A_e,
                             pi_phi_ph_a + self.coupling.mixing_A_e)
        fullmix =  pi_mix_sq / ((self.nu**2)[:, np.newaxis] - piaa)

        finalse = se + fullmix.T
        
        return finalse

    def get_photon_se(self):
        '''
        \Pi_AA given by Eqn. 32 in the draft. But we can also just grab it from our dielectric code.
        '''
        piaa_e = self.nu**2 * (1. - 1./3.*np.trace(self.mat.dielectric))
        charge_list = self.mat.Z_list - self.mat.get_Nj()
        
        self.formfacAA =  -E_EM*np.einsum('w, j, ab -> wjab', 
                                self.nu, 
                                charge_list, 
                                np.identity(3))
        piaa_ph = (1/3.0)*np.einsum('jmkw, wjab, wmab -> wk', 
                         self.totse, 
                         self.formfacAA, 
                         np.conj(self.formfacAA))

        piaa = piaa_e[:, np.newaxis] + piaa_ph

        return piaa
