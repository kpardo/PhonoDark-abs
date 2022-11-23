'''
selfenergy.py
'''

from dataclasses import dataclass
import numpy as np
from scipy import linalg as sla
import sys
from constants import *
from material import Material
import transfer_matrix as tm

@dataclass
class SelfEnergy:
    nu: np.ndarray
    k: np.ndarray
    j: np.ndarray
    mat: Material
    pol_mixing: bool
    lam: str

    def __post_init__(self):
        pass

@dataclass
class ScalarSE(SelfEnergy):
    uv_op1: str
    uv_op2: str

    def __post_init__(self):
        self.mat_sq = self.get_mat_sq()
        self.se = self.get_se()
        pass

    def get_mat_sq(self):
        right = tm.TransferMatrix(nu=self.nu, k=self.k, j=self.j,
                mat=self.mat, pol_mixing=self.pol_mixing, lam=self.lam,
                ground_state='right').tm
        left = tm.TransferMatrix(nu=self.nu, k=self.k, j=self.j,
                mat=self.mat, pol_mixing=self.pol_mixing, lam=self.lam,
                ground_state='left').tm
        return left*right

    def get_op_exp(self, op):
        if op == r'$g_\chi \phi \bar{\psi} \psi$':
            exp = E_EM
        else:
            print('! Not implemented !')
        return exp

    def get_se(self):
        opexp1 = self.get_op_exp(self.uv_op1)
        opexp2 = self.get_op_exp(self.uv_op2)
        return 1j*opexp1*opexp2*self.mat_sq
