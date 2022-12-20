'''
couplings.py
'''

from dataclasses import dataclass
import numpy as np
from constants import *


@dataclass
class Scalar:
    q_XYZ_list: np.ndarray
    name: str = 'scalar'
    texname: str = r'$\mathrm{Scalar}$'
    texop: str = r'$g_\chi \phi \bar{\psi} \psi$'
    texcoupconst: str = r'$g_{\chi}$'
    dotvec: np.ndarray = np.zeros((1))
    prefac: np.float = E_EM**2

    def __post_init__(self):
        self.dotvec = self.q_XYZ_list


@dataclass
class Pseudoscalar:
    q_XYZ_list: np.ndarray
    name: str = 'pseudoscalar'
    texname: str = r'$\mathrm{Pseudoscalar}$'
    texop: str = r'$g_\chi \phi\bar\psi i\gamma_5\psi$'
    texcoupconst: str = r'$g_{\chi}$'
    dotvec: np.ndarray = np.zeros((1))
    prefac: np.float = 0.

    def __post_init__(self):
        raise Warning('Not implemented fully!')


@dataclass
class Vector:
    q_XYZ_list: np.ndarray
    name: str = 'vector'
    texname: str = r'$\mathrm{Vector}$'
    texop: str = r'$g_\chi \phi_\mu\bar\psi \gamma^\mu\psi$'
    texcoupconst: str = r'$g_{\chi}$'
    dotvec: np.ndarray = np.zeros((1))
    prefac: np.float = 0.

    def __post_init__(self):
        raise Warning('Not implemented fully!')


@dataclass
class AxialVector:
    q_XYZ_list: np.ndarray
    name: str = 'axialvector'
    texname: str = r'$\mathrm{Axial~Vector}$'
    texop: str = r'$g_\chi \phi_\mu\bar\psi\gamma^\mu\gamma^5\psi$'
    texcoupconst: str = r'$g_{\chi}$'
    dotvec: np.ndarray = np.zeros((1))
    prefac: np.float = 0.

    def __post_init__(self):
        raise Warning('Not implemented fully!')


@dataclass
class ElectricDipole:
    q_XYZ_list: np.ndarray
    name: str = 'electricdipole'
    texname: str = r'$\mathrm{Electric~Dipole}$'
    texop: str = r'$\frac{g_\chi}{4m_\psi}\phi_{\mu\nu}\bar\psi\sigma^{\mu\nu}i\gamma^5\psi$'
    texcoupconst: str = r'$g_{\chi}$'
    dotvec: np.ndarray = np.zeros((1))
    prefac: np.float = 0.

    def __post_init__(self):
        raise Warning('Not implemented fully!')


@dataclass
class MagneticDipole:
    q_XYZ_list: np.ndarray
    name: str = 'magneticdipole'
    texname: str = r'$\mathrm{Magnetic~Dipole}$'
    texop: str = r'$\frac{g_\chi}{4m_\psi}\phi_{\mu\nu}\bar\psi\sigma^{\mu\nu}\psi$'
    texcoupconst: str = r'$g_{\chi}$'
    dotvec: np.ndarray = np.zeros((1))
    prefac: np.float = 0.

    def __post_init__(self):
        raise Warning('Not implemented fully!')


@dataclass
class Anapole:
    q_XYZ_list: np.ndarray
    name: str = 'anadipole'
    texname: str = r'$\mathrm{Magnetic~Dipole}$'
    texop: str = r'$\frac{g_\chi}{4m_\psi^2}\left(\partial^\nu \phi_{\mu\nu}\right)\left(\bar\psi\gamma^\mu\gamma^5\psi\right)$'
    texcoupconst: str = r'$g_{\chi}$'
    dotvec: np.ndarray = np.zeros((1))
    prefac: np.float = 0.

    def __post_init__(self):
        raise Warning('Not implemented fully!')


@dataclass
class Axion:
    q_XYZ_list: np.ndarray
    # 10 Tesla averaged over 3 directions.
    # FIXME: implement more general b field config.
    bfield: np.ndarray = 10**2 * T_To_eV2**2 * 1/3
    name: str = 'axion'
    texname: str = r'$\mathrm{Scalar}$'
    texop: str = r'$g_{a\gamma\gamma} a F_{\mu\nu}F^{\mu\nu}$'
    texcoupconst: str = r'$g_{a\gamma\gamma}$'
    dotvec: np.ndarray = np.zeros((1))
    prefac: np.float = 0.

    def __post_init__(self):
        self.prefac = E_EM**2 * self.bfield
        self.dotvec = np.ones((np.shape(self.q_XYZ_list)))
