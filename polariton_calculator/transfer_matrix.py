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

        ## sort Umat by eigenvalues
        ## descending order for pos. values, ascending for neg.

        ## first sort by descending order
        sort_index = np.argsort(eigen_vals)[::-1]
        ## find index of first negative number in sort
        arg_first_neg = (np.less(eigen_vals[sort_index], 0)).argmax()
        ## flip negative part of indexes
        sort_index[arg_first_neg:] = np.flip(sort_index[arg_first_neg:])
        ## get sort eigen vecs according to sorted eigen vals.
        U_mat = np.array(eigen_vecs[sort_index], dtype=complex)
        U_mat_dag = np.conj(U_mat.T)

        ## below is good unit test later.
        if not np.allclose(np.matmul(U_mat_dag, U_mat), np.identity(h_mat_dim)):
            print('Eigenvectors of h are NOT orthonormalized.')

        L_mat = np.matmul(U_mat_dag, np.matmul(KgK_mat, U_mat))

        E_mat = np.matmul(g_mat, L_mat)

        T_mat = np.matmul(K_mat_inv, np.matmul(U_mat, sla.sqrtm(E_mat)))

        return [E_mat, T_mat]

@dataclass
class TMatPol(TransferMatrix):
    '''
    Calculates the T matrix for phonon-polariton contributions.
    Inputs: ??
    Outputs: diagonalized T matrix
    '''
    qvec: np.ndarray
    dielectric: np.ndarray
    o_phon_energy = np.ndarray

    def __post_init__():
        self.h = self.get_h_matrix()


    def get_n_modes(self, o_phon_energy):
        num_phon_modes = len(o_phon_energy) + 3 ## tot num of phonons
        num_phot_modes = len(o_phon_energy)
        num_o_phot_modes = 2 ## number of photon modes
        num_pol_modes = num_phot_modes + num_o_phon
        return num_phon_modes, num_phot_modes, num_o_phot_modes, num_pol_modes

    def split_energy(energies, split_amount):
    	"""

    	Splits the energy levels by the split_amount.
        Removes degeneracy in eigenvalues

    	"""
    	n_energies = len(energies)
    	for i in range(n_energies):
    		split_amount = split*(i - n_energies/2)
    		energies[i] += split_amount
    	return energies

    def get_h_matrix(self, split_en_level = 10**(-8)):
        modes =self.get_n_modes(self.o_phon_energy)
        num_phon_modes, num_phot_modes, num_o_phot_modes, num_pol_modes = modes

        ## set up h matrix shape & container
        h_mat_dim = 2*num_pol_modes
        h_mat = np.zeros((h_mat_dim, h_mat_dim), dtype=complex)

        ## split energies by a bit to remove eigenvalue degeneracies
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
