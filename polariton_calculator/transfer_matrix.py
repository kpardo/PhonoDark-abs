'''
transfer_matrix.py

Defines the transfer matrix class and subclasses.
'''
from dataclasses import dataclass
import numpy as np
from scipy import linalg as sla
import physics
import sys
from constants import *

@dataclass
class TransferMatrix:
    '''
    Calculates the T matrix for a generic h matrix
    Input: hmatrix
    Outputs: diagonalized T matrix
    '''
    hmatrix: np.ndarray

    def __post_init__(self):
        '''
        gets E and T matrix as part of init.
        '''
        self.E_mat, self.T_mat = self.get_T_matrix()

    def get_T_matrix(self):
        '''
        gets the E and T matrices according to
        2005.10256 Appendix C algorithm
        '''
        h_mat_dim = len(self.hmatrix)

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
        ## flip second half of eigenvalues
        sort_index[h_mat_dim//2:] = np.flip(sort_index[h_mat_dim//2:])
        ## get sorted eigen vecs according to sorted eigen vals.
        U_mat = np.array(eigen_vecs[:,sort_index], dtype=complex)

        U_mat_dag = np.conj(U_mat.T)

        ## below is good unit test later.
        if not np.allclose(np.matmul(U_mat_dag, U_mat), np.identity(h_mat_dim)):
            print('Eigenvectors of h are NOT orthonormalized.')

        L_mat = np.matmul(U_mat_dag, np.matmul(KgK_mat, U_mat))

        E_mat = np.matmul(g_mat, L_mat)

        sqrt_E_mat = sla.sqrtm(E_mat)

        T_mat = np.matmul(K_mat_inv, np.matmul(U_mat, sqrt_E_mat))

        return [E_mat, T_mat]

@dataclass
class TMatPol():
    '''
    Calculates the T matrix for phonon-polariton contributions.
    Inputs: ??
    Outputs: diagonalized T matrix
    '''
    q_vec: np.ndarray
    dielectric: np.ndarray
    V_PC: float
    o_xi_vec: np.ndarray
    phot_eigenvecs: np.ndarray
    o_phon_energy: np.ndarray
    phot_energy: np.ndarray
    dielectric_diag: np.ndarray
    K_sq_mat: np.ndarray

    def __post_init__(self):
        self.h = self.get_h_matrix()


    def get_n_modes(self, o_phon_energy):
        num_phon_modes = len(o_phon_energy) + 3 ## tot num of phonons
        num_o_phon = len(o_phon_energy)
        num_phot_modes = 2 ## number of photon modes
        num_pol_modes = num_phot_modes + num_o_phon
        return num_phon_modes, num_phot_modes, num_o_phon, num_pol_modes

    @staticmethod
    def split_energy(energies, split):
        """
        Splits the energy levels by the split amount.
        Removes degeneracy in eigenvalues
        """
        energies = [e + split * (i - len(energies) / 2)
         for i,e in enumerate(energies)]
        return energies

    def get_h_matrix(self, split_en_level = 10**(-8)):
        modes = self.get_n_modes(self.o_phon_energy)
        num_phon_modes, num_phot_modes, num_o_phon, num_pol_modes = modes

        ## set up h matrix shape & container
        h_mat_dim = 2*num_pol_modes
        h_mat = np.zeros((h_mat_dim, h_mat_dim), dtype=complex)

        ## split energies by a bit to remove eigenvalue degeneracies
        o_phon_energy = self.split_energy(self.o_phon_energy, split_en_level)
        phot_energy = np.real(self.split_energy(self.phot_energy, split_en_level))

        ## term I: free phonon
        h_mat[:num_o_phon, :num_o_phon] += 0.5*np.diag(o_phon_energy)
        h_mat[num_pol_modes:h_mat_dim-2, num_pol_modes:h_mat_dim-2] += 0.5*np.diag(o_phon_energy)

        # term II: free photon
        h_mat[num_o_phon:h_mat_dim//2, num_o_phon:h_mat_dim//2] += 0.5*np.diag(phot_energy)
        h_mat[h_mat_dim-2:h_mat_dim, h_mat_dim-2:h_mat_dim] += 0.5*np.diag(phot_energy)


        # term III: interaction term

        # 2 x num_modes - 3
        a_mat = np.zeros((num_phot_modes, num_o_phon), dtype=complex)

        # used to check invariance of interaction term with respect to a phase
        # rand_phase = np.exp(2*PI*random.random()*1j)

        for lam in range(num_phot_modes):
            for nu in range(num_o_phon):

                a_mat[lam][nu] = (1j/4.0)*E_EM*(np.sqrt(self.V_PC))**(-1)*\
                np.sqrt(self.dielectric_diag[lam])**(-1)*\
                np.sqrt(o_phon_energy[nu]/phot_energy[lam])*\
                np.dot(np.conj(self.phot_eigenvecs[lam]), self.o_xi_vec[nu])

        # num_modes - 3 x 2
        a_mat_dag = np.conj(a_mat.T)

        h_mat[num_o_phon:h_mat_dim//2, :num_o_phon] += a_mat
        h_mat[:num_o_phon, num_o_phon:h_mat_dim//2] += a_mat_dag
        h_mat[h_mat_dim-2:h_mat_dim, :num_o_phon] += a_mat
        h_mat[:num_o_phon, h_mat_dim-2:h_mat_dim] += a_mat_dag

        h_mat[num_o_phon:h_mat_dim//2, h_mat_dim//2:h_mat_dim-2] += -a_mat
        h_mat[h_mat_dim//2:h_mat_dim-2, num_o_phon:h_mat_dim//2] += -a_mat_dag
        h_mat[h_mat_dim-2:h_mat_dim, h_mat_dim//2:h_mat_dim-2] += -a_mat
        h_mat[h_mat_dim//2:h_mat_dim-2, h_mat_dim-2:h_mat_dim] += -a_mat_dag
        # term IV: NAC term (Term 2 Eq. A25 in 2005.10256)
        b_mat = np.zeros((num_o_phon, num_o_phon), dtype=complex)

        q_dir = self.q_vec / np.linalg.norm(self.q_vec)

        b_mat = np.array([ (1/4.0)*(E_EM**2/self.V_PC)*\
            (np.dot(q_dir, np.matmul(self.dielectric, q_dir)))**(-1)*\
            (np.sqrt(o_phon_energy[nu]*o_phon_energy[nup]))**(-1)*\
            np.dot(q_dir, np.conj(self.o_xi_vec[nu]))*\
            np.dot(q_dir,self. o_xi_vec[nup])
            for nu in range(num_o_phon) for nup in range(num_o_phon)],
            dtype=complex).reshape((num_o_phon, num_o_phon))
        h_mat[:num_o_phon, :num_o_phon] += b_mat
        h_mat[:num_o_phon, h_mat_dim//2:h_mat_dim-2] += b_mat
        h_mat[h_mat_dim//2:h_mat_dim-2, :num_o_phon] += b_mat
        h_mat[h_mat_dim//2:h_mat_dim-2, h_mat_dim//2:h_mat_dim-2] += b_mat


        # term V: inhomogeneous mass term

        c_mat = np.zeros((num_phot_modes, num_phot_modes), dtype=complex)

        for l in range(num_phot_modes):
            for lp in range(num_phot_modes):

                if l != lp:

                    c_mat[l][lp] = (1/4.0)*(np.sqrt(self.dielectric_diag[l]*self.dielectric_diag[lp]))**(-1)*\
                    (np.sqrt(phot_energy[l]*phot_energy[lp]))**(-1)*\
                    np.dot(np.conj(self.phot_eigenvecs[l]),
                    np.matmul(self.K_sq_mat, self.phot_eigenvecs[lp]))

        h_mat[num_o_phon:h_mat_dim//2, num_o_phon:h_mat_dim//2] += c_mat
        h_mat[num_o_phon:h_mat_dim//2, h_mat_dim-2:h_mat_dim] += c_mat
        h_mat[h_mat_dim-2:h_mat_dim, num_o_phon:h_mat_dim//2] += c_mat
        h_mat[h_mat_dim-2:h_mat_dim, h_mat_dim-2:h_mat_dim] += c_mat

        # checks on matrix -- would be good unit tests later.
        if not np.allclose(h_mat, np.conj(h_mat.T)):
            print('WARNING: h_mat is not Hermitian.')
            print(h_mat)

        if not np.all(np.linalg.eigvals(h_mat) > 0):
            print('ERROR: h_mat is not positive definite. Cholesky decomposition will fail.')

            print(np.linalg.eigvals(h_mat))

        return h_mat

@dataclass
class TMatPhonon(TransferMatrix):
    '''
    Inputs: energies, operator, approxs
    '''
    pass

@dataclass
class TMatMagnon(TransferMatrix):
    pass
