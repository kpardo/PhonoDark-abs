'''
selfenergy.py
'''

from dataclasses import dataclass
import numpy as np
from constants import *
from material import Material
import transfer_matrix as tm


@dataclass
class SelfEnergy:
    nu: np.ndarray
    k: np.ndarray
    mat: Material
    pol_mixing: bool
    lam: str

    def __post_init__(self):
        self.mat_sq = self.get_mat_sq()
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

    def get_se(self):
        # FIXME: maybe get actual full SE here?
        return 1j*self.mat_sq
