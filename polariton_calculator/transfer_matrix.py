'''
transfer_matrix.py

Defines the transfer matrix class and subclasses.
'''
from dataclasses import dataclass
import numpy as np
from scipy import linalg as sla
import physics

@dataclass
class TransferMatrix:
    '''
    Calculates the T matrix for a generic h matrix
    Input: hmatrix
    Outputs: diagonalized T matrix
    '''
    hmatrix: np.ndarray

    def __post_init__(self):
        self.E_mat, self.T_mat = self.get_T_matrix()

    def get_T_matrix(self):
        h_mat_dim = len(self.hmatrix)
        num_o_phon = h_mat_dim//2 - 2

        g_mat_diag = np.ones(h_mat_dim//2)
        g_mat = np.zeros((h_mat_dim, h_mat_dim))
        g_mat[:h_mat_dim//2, :h_mat_dim//2] = np.diag(g_mat_diag)
        g_mat[h_mat_dim//2:, h_mat_dim//2:] = np.diag(-1*g_mat_diag)

        K_mat = sla.cholesky(self.hmatrix)
        K_dag = np.conj(K_mat.T)
        K_mat_inv = np.linalg.inv(K_mat)

        KgK_mat = np.matmul(K_mat, np.matmul(g_mat, K_dag))

        eigen_vals, eigen_vecs = sla.eig(KgK_mat)
        # create dict of eigen vectors and values
        eigen_dict = {(v, i):vec for i,(v,vec) in enumerate(zip(eigen_vals, eigen_vecs))}
        # sort Umat by eigenvalues
        U_mat = np.array(eigen_vecs[np.argsort(eigen_vals)[::-1]], dtype=complex)

        U_mat_dag = np.conj(U_mat.T)

        if not np.allclose(np.matmul(U_mat_dag, U_mat), np.identity(h_mat_dim)):
            print('Eigenvectors of h are NOT orthonormalized.')

        L_mat = np.matmul(U_mat_dag, np.matmul(KgK_mat, U_mat))

        E_mat = np.matmul(g_mat, L_mat)

        sqrt_E_mat = sla.sqrtm(E_mat)

        T_mat = np.matmul(K_mat_inv, np.matmul(U_mat, sqrt_E_mat))

        return [E_mat, T_mat]

@dataclass
class TMatPol(TransferMatrix):
    '''
    Calculates the T matrix for phonon-polariton contributions.
    Inputs: h matrix
    Outputs: diagonalized T matrix
    '''
    pass

@dataclass
class TMatPhonon(TransferMatrix):
    '''
    Inputs: energies, operator, approxs
    '''
    pass

@dataclass
class TMatMagnon(TransferMatrix):
    pass
