'''
couplings.py
'''

from dataclasses import dataclass
import numpy as np
from pda.constants import *


@dataclass
class Scalar:
    q_XYZ_list: np.ndarray
    name: str = 'scalar'
    texname: str = r'$\mathrm{Scalar}$'
    texop: str = r'$g_\chi \phi \bar{\psi} \psi$'
    texcoupconst: str = r'$g_{\chi}$'
    formfac: np.ndarray = np.zeros((1))
    prefac: np.float64 = E_EM**2
    se_shape: str = 'scalar'

    def __post_init__(self):
        self.formfac = self.q_XYZ_list


@dataclass
class ScalarE:
    q_XYZ_list: np.ndarray
    name: str = 'scalar_e'
    texname: str = r'$\mathrm{Scalar~DM}$'
    texop: str = r'$d_{\phi e e} \frac{\sqrt{4\pi}m_e}{M_{\mathrm{Pl}}} \phi \bar{\psi}\psi$'
    texcoupconst: str = r'$d_{\phi e e}$'
    formfac: np.ndarray = np.zeros((1))
    prefac: np.float64 = E_EM**2 * (4 * np.pi) * (M_ELEC/M_PL)
    se_shape: str = 'scalar'
    # coupling_cns: dict = {'ce': 1,
    #                       'cp': 0,
    #                       'cn': 0}

    def __post_init__(self):
        self.formfac = self.q_XYZ_list


@dataclass
class Pseudoscalar:
    q_XYZ_list: np.ndarray
    name: str = 'pseudoscalar'
    texname: str = r'$\mathrm{Pseudoscalar}$'
    texop: str = r'$g_\chi \phi\bar\psi i\gamma_5\psi$'
    texcoupconst: str = r'$g_{\chi}$'
    formfac: np.ndarray = np.zeros((1))
    prefac: np.float64 = 0.
    se_shape: str = 'scalar'

    def __post_init__(self):
        # raise Warning('Not implemented fully!')
        print('Not implemented fully!')


@dataclass
class Vector:
    q_XYZ_list: np.ndarray
    omega: np.ndarray  # e.g., DM masses
    name: str = 'vector'
    texname: str = r'$\mathrm{Vector}$'
    texop: str = r'$g_\chi \phi_\mu\bar\psi \gamma^\mu\psi$'
    texcoupconst: str = r'$g_{\chi}$'
    formfac: np.ndarray = np.zeros((1))
    prefac: np.float64 = E_EM**2
    se_shape: str = 'vector'

    def __post_init__(self):
        self.formfaci0 = self.q_XYZ_list
        self.formfacij = np.einsum(
            'j, ab -> jab', -1.*self.omega, np.eye(3, 3))


@dataclass
class DarkPhoton:
    q_XYZ_list: np.ndarray
    omega: np.ndarray  # e.g., DM masses
    name: str = 'dark_photon'
    texname: str = r'$\mathrm{Vector~DM}$'
    texop: str = r'$\kappa e \phi_\mu\bar\psi \gamma^\mu\psi$'
    texcoupconst: str = r'$\kappa$'
    formfac: np.ndarray = np.zeros((1))
    prefac: np.float64 = 1.  # e's taken from conversion from g to kappa
    se_shape: str = 'vector'

    def __post_init__(self):
        self.formfaci0 = self.q_XYZ_list
        self.formfacij = np.einsum(
            'j, ab -> jab', -1.*self.omega, np.eye(3, 3))

@dataclass
class U1b:
    q_XYZ_list: np.ndarray
    omega: np.ndarray  # e.g., DM masses
    name: str = 'u1b'
    texname: str = r'$\mathrm{U}(1)\mathrm{b}$'
    #FIXME: not sure if correct op.
    texop: str = r'$g_B \phi_\mu\bar\psi \gamma^\mu\psi$'
    texcoupconst: str = r'$g_B$'
    formfac: np.ndarray = np.zeros((1))
    #FIXME: not sure if correct prefac?
    prefac: np.float64 = 1.  # e's taken from conversion from g to g_B?
    se_shape: str = 'vector'

    def __post_init__(self):
        self.formfaci0 = self.q_XYZ_list
        self.formfacij = np.einsum(
            'j, ab -> jab', -1.*self.omega, np.eye(3, 3))


@dataclass
class AxialVector:
    q_XYZ_list: np.ndarray
    name: str = 'axialvector'
    texname: str = r'$\mathrm{Axial~Vector}$'
    texop: str = r'$g_\chi \phi_\mu\bar\psi\gamma^\mu\gamma^5\psi$'
    texcoupconst: str = r'$g_{\chi}$'
    formfac: np.ndarray = np.zeros((1))
    prefac: np.float64 = 0.
    se_shape: str = 'vector'

    def __post_init__(self):
        # raise Warning('Not implemented fully!')
        print('Not implemented fully!')


@dataclass
class ElectricDipole:
    q_XYZ_list: np.ndarray
    name: str = 'electricdipole'
    texname: str = r'$\mathrm{Electric~Dipole}$'
    texop: str = r'$\frac{g_\chi}{4m_\psi}\phi_{\mu\nu}\bar\psi\sigma^{\mu\nu}i\gamma^5\psi$'
    texcoupconst: str = r'$g_{\chi}$'
    formfac: np.ndarray = np.zeros((1))
    prefac: np.float64 = 0.

    def __post_init__(self):
        # raise Warning('Not implemented fully!')
        print('Not implemented fully!')


@dataclass
class MagneticDipole:
    q_XYZ_list: np.ndarray
    name: str = 'magneticdipole'
    texname: str = r'$\mathrm{Magnetic~Dipole}$'
    texop: str = r'$\frac{g_\chi}{4m_\psi}\phi_{\mu\nu}\bar\psi\sigma^{\mu\nu}\psi$'
    texcoupconst: str = r'$g_{\chi}$'
    formfac: np.ndarray = np.zeros((1))
    prefac: np.float64 = 0.
    se_shape: str = 'scalar'

    def __post_init__(self):
        # raise Warning('Not implemented fully!')
        print('Not implemented fully!')


@dataclass
class Anapole:
    q_XYZ_list: np.ndarray
    name: str = 'anapole'
    texname: str = r'$\mathrm{Anapole}$'
    texop: str = r'$\frac{g_\chi}{4m_\psi^2}\left(\partial^\nu \phi_{\mu\nu}\right)\left(\bar\psi\gamma^\mu\gamma^5\psi\right)$'
    texcoupconst: str = r'$g_{\chi}$'
    formfac: np.ndarray = np.zeros((1))
    prefac: np.float64 = 0.
    se_shape: str = 'scalar'

    def __post_init__(self):
        # raise Warning('Not implemented fully!')
        print('Not implemented fully!')


@dataclass
class Axion_ExternalB:
    q_XYZ_list: np.ndarray
    # 10 Tesla averaged over 3 directions.
    # FIXME: implement more general b field config.
    bfield: np.ndarray = 10**2 * T_To_eV2**2 * 1/3
    name: str = 'axion'
    texname: str = r'$\mathrm{Axion}$'
    texop: str = r'$g_{a\gamma\gamma} a F_{\mu\nu}F^{\mu\nu}$'
    texcoupconst: str = r'$g_{a\gamma\gamma}$'
    formfac: np.ndarray = np.zeros((1))
    prefac: np.float64 = 0.
    se_shape: str = 'scalar'

    def __post_init__(self):
        self.prefac = E_EM**2 * self.bfield
        self.formfac = np.ones((np.shape(self.q_XYZ_list)))

@dataclass
class Axion:
    q_XYZ_list: np.ndarray
    omega: np.ndarray  # e.g., DM masses
    S: np.ndarray # magnetic spin vector
    fermion_coupling: 'e'
    name: str = 'axion'
    texname: str = r'$\mathrm{Axion}$'
    formfac: np.ndarray = np.zeros((1))
    prefac: np.float64 = 0.
    se_shape: str = 'vector2'

    def __post_init__(self):
        self.texcoupconst, mfermion = self.get_coupconst()
        self.prefac = 0.5*E_EM**2*M_PL/mfermion
        self.formfaci0 = np.einsum('a,b -> ab', -2*self.omega,self.S)
        self.formfacij = np.einsum(
            'ja, b -> jab', -2.*self.q_XYZ_list, self.S)

    def get_coupconst(self):
        if self.fermion_coupling == 'e':
            return r'$g_{aee}$', M_ELEC
        elif self.fermion_coupling == 'n':
            return r'$g_{ann}$', M_NEUTRON
        elif self.fermion_coupling == 'p':
            return r'$g_{app}$', M_PROTON
        else:
            raise Warning('Not a valid fermion type!')