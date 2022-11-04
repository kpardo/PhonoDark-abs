'''
transfer_matrix.py
'''

from dataclasses import dataclass
import numpy as np
from scipy import linalg as sla
import physics
import sys
from constants import *

@dataclass
class TransferMatrix:
    nu: np.ndarray
    k: np.ndarray
    j: np.ndarray
    lam: str
    fn: str = f'transfer_{lam}.dat'

    def __post_init__(self):
        ## make container for TM
        ## get all ingredients for calc.
        ## get the transfer matrix
        ## save to file.
        pass

    def get_dielectric(self):
        pass

    def get_uv(self):
        pass

    def get_transfer(self):
        pass
