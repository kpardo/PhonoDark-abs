'''
couplings.py
'''

from dataclasses import dataclass
import numpy as np
from scipy import linalg as sla
import sys
from constants import *
from material import Material


@dataclass
class Coupling:
    texname: str
    texop: str
    texcoupconst: str
    dotvec: np.ndarray
    prefac: np.float or np.ndarray


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
class Axion:
    q_XYZ_list: np.ndarray
    # 10 Tesla averaged over 3 directions.
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
