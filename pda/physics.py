# some physics functions

import numpy as np
from scipy import linalg as sla

from scipy import integrate
from scipy import special

import math
import os
import phonopy

import phonopy_funcs
from constants import *
import my_math

import random

def create_xi_vecs(born, eigenvectors, atom_masses):
	"""
	returns the xi_vectors

	[k][nu][alpha]
	"""

	n_k = len(eigenvectors)
	num_modes = len(eigenvectors[0])
	num_atoms = len(atom_masses)

	xi_vecs = np.zeros((n_k, num_modes, 3), dtype=np.complex128)

	for k in range(len(eigenvectors)):
		for nu in range(len(eigenvectors[k])):
			for alpha in range(3):
				for j in range(num_atoms):

					xi_vecs[k][nu][alpha] += (atom_masses[j])**(-0.5)\
										*np.matmul(born[j], eigenvectors[k][nu][j])[alpha]

	return xi_vecs

def create_photon_eigensys(q, dielectric, born, V_PC, atom_masses):
	"""
	given a q vector and a dielectric, creates the basis vectors which satisfy

	q . dielectric . e = 0

	and diagonalize the dielectric

	e . dielectric . e = e1.e1 dielectric_1 + ...

	returns [ [e1, e2], (eigenvalues of projected dielectric), (photon energy) ]
	"""

	# get vectors orthogonal to q . dielectric

	q_dielectric = np.matmul(q, dielectric)

	orthog_basis = my_math.create_perp_orthonormal_basis(q_dielectric)

	# compute e_nu^* . dielectric . e_nup
	projected_dielectric = np.zeros((2, 2))

	for nu in range(2):
		for nup in range(2):

			projected_dielectric[nu][nup] = np.dot(
												np.conj(orthog_basis[nu]),
												np.matmul(
													dielectric,
													orthog_basis[nup]
													)
												)

	eigen_sys_proj_dielectric = sla.eigh(projected_dielectric)
	eigen_vals_proj_dielectric = eigen_sys_proj_dielectric[0]
	eigen_vecs_proj_dielectric = eigen_sys_proj_dielectric[1]

	basis = np.zeros((2, 3), dtype=complex)

	for lam in range(2):
		for i in range(3):
			for lamp in range(2):

				basis[lam][i] += eigen_vecs_proj_dielectric[lam][lamp]*orthog_basis[lamp][i]

	omega_photon = np.zeros(2, dtype=complex)

	# build K^2 mat
	K_sq_mat = create_K_sq_mat(q, born, V_PC, atom_masses)

	for lam in range(2):
		omega_photon[lam] = np.sqrt( eigen_vals_proj_dielectric[lam]**(-1)*\
							np.dot(
								np.conj(basis[lam]),
								np.matmul(
									K_sq_mat,
									basis[lam]
									)
								)
							)

	return [basis, eigen_vals_proj_dielectric, omega_photon]


def create_K_sq_mat(q, born, V_PC, atom_masses):

	K_sq_mat = np.zeros((3, 3), dtype=complex)

	K_sq_mat += np.dot(q, q)*np.diag(np.ones(3))

	for j in range(len(born)):
		K_sq_mat += (E_EM**2/V_PC)*(atom_masses[j])**(-1)*\
							np.matmul(born[j], born[j].T)

	for i in range(3):
		for j in range(3):

			K_sq_mat[i][j] -= q[i]*q[j]

	return K_sq_mat


def split_energy(energies, split):
	"""

	Splits the energy levels by the amount split. Removes degeneracy in eigenvalues

	"""

	n_energies = len(energies)

	for i in range(n_energies):

		split_amount = split*(i - n_energies/2)

		energies[i] += split_amount

	return energies

## KP: careful with modifying below.
def create_h_mat(q_vec, dielectric, V_PC, o_xi_vec, phot_eigenvecs, o_phon_energy, phot_energy,
					dielectric_diag, K_sq_mat):
	"""
	returns the h matrix at q
	"""

	# total number of phonons
	num_phon_modes = len(o_phon_energy) + 3

	# number of optical phonon modes
	num_o_phon = num_phon_modes - 3

	# number of photon modes
	num_phot_modes = 2

	# number of polariton modes
	num_pol_modes = num_phot_modes + num_o_phon

	# dimension of h matrix
	h_mat_dim = 2*num_pol_modes

	h_mat = np.zeros((h_mat_dim, h_mat_dim), dtype=complex)

	# split energies, remove degeneracies

	o_phon_energy = split_energy(o_phon_energy, 10**(-8))
	phot_energy = np.real(split_energy(phot_energy, 10**(-8)))

	# term I: free phonon

	h_mat[:num_o_phon, :num_o_phon] += 0.5*np.diag(o_phon_energy)
	h_mat[(h_mat_dim//2):h_mat_dim-2, (h_mat_dim//2):h_mat_dim-2] += 0.5*np.diag(o_phon_energy)

	# term II: free photon

	h_mat[num_o_phon:h_mat_dim//2, num_o_phon:h_mat_dim//2] += 0.5*np.diag(phot_energy)
	h_mat[h_mat_dim-2:h_mat_dim, h_mat_dim-2:h_mat_dim] += 0.5*np.diag(phot_energy)

	# perfect diagonalization will give a zero energy mode.
	tol = 0

	h_mat[:, :] += tol*np.diag(np.ones(h_mat_dim))

	# term III: interaction term

	# 2 x num_modes - 3
	a_mat = np.zeros((num_phot_modes, num_o_phon), dtype=complex)

	# used to check invariance of interaction term with respect to a phase
	# rand_phase = np.exp(2*PI*random.random()*1j)

	for lam in range(num_phot_modes):
		for nu in range(num_o_phon):

			a_mat[lam][nu] = (1j/4.0)*E_EM*(np.sqrt(V_PC))**(-1)*\
							np.sqrt(dielectric_diag[lam])**(-1)*\
							np.sqrt(o_phon_energy[nu]/phot_energy[lam])*\
							np.dot(np.conj(phot_eigenvecs[lam]), o_xi_vec[nu])

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

	# term IV: NAC term

	b_mat = np.zeros((num_o_phon, num_o_phon), dtype=complex)

	q_dir = q_vec / np.linalg.norm(q_vec)

	for nu in range(num_o_phon):
		for nup in range(num_o_phon):

			b_mat[nu][nup] = (1/4.0)*(E_EM**2/V_PC)*\
								(np.dot(q_dir, np.matmul(dielectric, q_dir)))**(-1)*\
								(np.sqrt(o_phon_energy[nu]*o_phon_energy[nup]))**(-1)*\
								np.dot(q_dir, np.conj(o_xi_vec[nu]))*\
								np.dot(q_dir, o_xi_vec[nup])

	h_mat[:num_o_phon, :num_o_phon] += b_mat
	h_mat[:num_o_phon, h_mat_dim//2:h_mat_dim-2] += b_mat
	h_mat[h_mat_dim//2:h_mat_dim-2, :num_o_phon] += b_mat
	h_mat[h_mat_dim//2:h_mat_dim-2, h_mat_dim//2:h_mat_dim-2] += b_mat

	# term V: inhomogeneous mass term

	c_mat = np.zeros((num_phot_modes, num_phot_modes), dtype=complex)

	for l in range(num_phot_modes):
		for lp in range(num_phot_modes):

			if l != lp:

				c_mat[l][lp] = (1/4.0)*(np.sqrt(dielectric_diag[l]*dielectric_diag[lp]))**(-1)*\
								(np.sqrt(phot_energy[l]*phot_energy[lp]))**(-1)*\
								np.dot(np.conj(phot_eigenvecs[l]),
										np.matmul(K_sq_mat, phot_eigenvecs[lp]))

	h_mat[num_o_phon:h_mat_dim//2, num_o_phon:h_mat_dim//2] += c_mat
	h_mat[num_o_phon:h_mat_dim//2, h_mat_dim-2:h_mat_dim] += c_mat
	h_mat[h_mat_dim-2:h_mat_dim, num_o_phon:h_mat_dim//2] += c_mat
	h_mat[h_mat_dim-2:h_mat_dim, h_mat_dim-2:h_mat_dim] += c_mat

	# checks on matrix

	if not np.allclose(h_mat, np.conj(h_mat.T)):
		print('WARNING: h_mat is not Hermitian.')
		print(h_mat)

	if not np.all(np.linalg.eigvals(h_mat) > 0):
		print('ERROR: h_mat is not positive definite. Cholesky decomposition will fail.')

		print(np.linalg.eigvals(h_mat))

	return h_mat

def sort_eigevals(e_val):

	num_o_phon = len(e_val)//2 - 2
	# sort maximum to minimum
	e_val.sort(key=lambda x:x[0], reverse=True)

	# reverse negative values
	e_val_neg = e_val[num_o_phon+2:]
	for i in range(num_o_phon+2):
		e_val[num_o_phon+2+i] = e_val_neg[-(i+1)]

	return e_val

def get_E_mat_T_mat(h_mat, h_mat_dim):
	"""
	returns the E and T matrices for a given h mat
	"""

	# Cholesky decomposition

	num_o_phon = h_mat_dim//2 - 2

	g_mat_diag = np.ones(h_mat_dim//2)
	g_mat = np.zeros((h_mat_dim, h_mat_dim))
	g_mat[:h_mat_dim//2, :h_mat_dim//2] = np.diag(g_mat_diag)
	g_mat[h_mat_dim//2:, h_mat_dim//2:] = np.diag(-1*g_mat_diag)

	K_mat = sla.cholesky(h_mat)
	K_dag = np.conj(K_mat.T)
	K_mat_inv = np.linalg.inv(K_mat)

	KgK_mat = np.matmul(K_mat, np.matmul(g_mat, K_dag))

	eigen_sys = sla.eig(KgK_mat)
	eigen_vals = eigen_sys[0]
	eigen_vecs = eigen_sys[1]

	# sort eigenvectors

	eigen_dict = {}

	for i in range(h_mat_dim):
		eigen_dict[(eigen_vals[i],i)] = eigen_vecs[:,i]

	e_val_sorted = sort_eigevals(list(eigen_dict.keys()))

	U_mat = np.zeros((h_mat_dim, h_mat_dim), dtype=complex)

	counter = 0

	for i in range(len(e_val_sorted)):

		U_mat[:,counter] = eigen_dict[e_val_sorted[i]]
		counter += 1

	U_mat_dag = np.conj(U_mat.T)

	if not np.allclose(np.matmul(U_mat_dag, U_mat), np.identity(h_mat_dim)):
		print('Eigenvectors of h are NOT orthonormalized.')

	L_mat = np.matmul(U_mat_dag, np.matmul(KgK_mat, U_mat))

	E_mat = np.matmul(g_mat, L_mat)

	sqrt_E_mat = sla.sqrtm(E_mat)

	T_mat = np.matmul(K_mat_inv, np.matmul(U_mat, sqrt_E_mat))

	return [E_mat, T_mat]


def check_phonon_eigenvectors(MATERIAL, q_XYZ_list, phonon_eigenvectors):

	print('Checking phonon eigenvectors...')

	dir_path = os.path.dirname(os.path.realpath(__file__))

	POSCAR_PATH     = dir_path+"/material_data/"+MATERIAL+"/POSCAR"
	FORCE_SETS_PATH = dir_path+"/material_data/"+MATERIAL+"/FORCE_SETS"
	BORN_PATH       = dir_path+"/material_data/"+MATERIAL+"/BORN"

	phonon_file = phonopy.load(		supercell_matrix=supercell_data[MATERIAL],
								primitive_matrix='auto',
					  			unitcell_filename=POSCAR_PATH,
					  			force_sets_filename=FORCE_SETS_PATH
					  		)

	[	num_atoms,
		num_modes,
		pos_red_to_XYZ,
		pos_XYZ_to_red,
		recip_red_to_XYZ,
		recip_XYZ_to_red,
		eq_positions,
		atom_masses,
		born,
		dielectric,
		V_PC		] = phonopy_funcs.get_phonon_file_data(phonon_file, MATERIAL)

	[bare_ph_eigen_minus_q, bare_ph_energy_minus_q] = phonopy_funcs.run_phonopy(phonon_file,
				q_XYZ_list_to_k_red_list(-1*q_XYZ_list, recip_XYZ_to_red))

	if not np.allclose(bare_ph_eigen_minus_q, np.conj(phonon_eigenvectors)):
		print('xi_-k != xi_k^*')
		print()

	num_modes = len(phonon_eigenvectors[0])
	num_atoms = len(phonon_eigenvectors[0][0])

	for q in range(len(phonon_eigenvectors)):

		orthog_check_j = np.zeros((num_atoms,num_atoms),dtype=complex)
		orthog_check_nu = np.zeros((num_modes,num_modes),dtype=complex)

		for nu1 in range(num_modes):

			for i1 in range(num_atoms):
				for j in range(num_atoms):


					orthog_check_j[i1][j] += np.dot(
											np.conj(phonon_eigenvectors[q][nu1][i1]),
											phonon_eigenvectors[q][nu1][j]
											)

		for i2 in range(num_atoms):

			for nu2 in range(num_modes):
				for nup in range(num_modes):

					orthog_check_nu[nu2][nup] += np.dot(
											np.conj(phonon_eigenvectors[q][nu2][i2]),
											phonon_eigenvectors[q][nup][i2]
											)

		if not np.allclose(orthog_check_j, np.diag(3*np.ones(num_atoms))):

			print('sum_nu e*_j, nu . e_i nu != 3 delta_ij')

			print(orthog_check_j)

		if not np.allclose(orthog_check_nu, np.diag(np.ones(num_modes))):

			print('sum_i e*_i, nu . e_i nup != delta_nu nup')

			print(orthog_check_nu)

	print('Done!')

def check_photon_eigenvectors(q_XYZ_list, dielectric, born, V_PC, atom_masses):

	print('Checking photon eigenvectors...')

	for q in range(len(q_XYZ_list)):

		eigen_sys_q = create_photon_eigensys(q_XYZ_list[q], dielectric, born, V_PC, atom_masses)
		eigen_sys_minus_q = create_photon_eigensys(-1*q_XYZ_list[q], dielectric, born, V_PC, atom_masses)

		basis_q = eigen_sys_q[0]
		basis_minus_q = eigen_sys_minus_q[0]

		orthog_check = np.zeros((2,2),dtype=complex)

		for i in range(2):
			for j in range(2):

				orthog_check[i][j] = np.dot(np.conj(basis_q[i]), basis_q[j])

		if not np.allclose(np.diag(np.ones(2)), orthog_check):

			print('ERROR: e_k not orthonormal.')
			print(orthog_check)

		q_dir = q_XYZ_list[q]/np.linalg.norm(q_XYZ_list[q])

		# q . dielectric . basis[lam] = 0

		if (
			np.abs(np.dot(q_dir, np.matmul(dielectric, basis_q[0]))) > 10**(-8) or
			np.abs(np.dot(q_dir, np.matmul(dielectric, basis_q[1]))) > 10**(-8)
			):

			print('ERROR: k . dielectric . e_k != 0.')

		if not np.allclose(basis_q, np.conj(basis_minus_q)):
			print('ERROR. e_k != e_(-k)*.')


	print('Done!')

def create_photon_eigenvec_list(q_list, dielectric, born, V_PC, atom_masses):

	out = []

	for q in range(len(q_list)):

		out.append(create_photon_eigensys(q_list[q], dielectric, born, V_PC, atom_masses)[0])

	return np.array(out)


def create_photon_eigenvals_list(q_list, dielectric, born, V_PC, atom_masses):
	"""
	returns a list of eigenvectors at each k
	"""

	out = []

	for q in range(len(q_list)):

		eigen_sys = create_photon_eigensys(q_list[q], dielectric, born, V_PC, atom_masses)
		out.append([
			eigen_sys[1],
			eigen_sys[2]
			])

	return np.array(out)


def q_XYZ_list_to_k_red_list(q_list, recip_XYZ_to_red):
	"""
	conver the coordinates from physical to reduced.
	"""

	k_list = []

	for q in range(len(q_list)):
		k_list.append(np.matmul(recip_XYZ_to_red, q_list[q]))

	return np.array(k_list)


def f_maxwell_boltzmann(v, vE):
	"""
	v, vE: numpy arrays of length 3
	"""

	mag_vpvE = sla.norm(v + vE)

	if mag_vpvE > VESC:
		return 0
	else:
		return ( N0**(-1)*np.exp(-mag_vpvE**2/V0**2) )


def int_vel_test():
	"""
	checks normalization of f_maxwell_boltzmann
	"""

	def integrand(v, theta, phi):

		vVec = np.array(
			[
				v*np.sin(theta)*np.cos(phi),
				v*np.sin(theta)*np.sin(phi),
				v*np.cos(theta)
			])

		jacobian = np.sin(theta)*v**2

		return jacobian*f_maxwell_boltzmann(vVec, VE*np.array([0, 0, 1]))


	return integrate.nquad(integrand, [[0, 2*VESC], [0, PI], [0, 2*PI]])


def int_vel_dist(theta, phi, vEVec):
	"""
	returns the integral of the maxwell boltzmann distribution over
	v
	"""

	def integrand(v):

		vVec = np.array(
			[
				v*np.sin(theta)*np.cos(phi),
				v*np.sin(theta)*np.sin(phi),
				v*np.cos(theta)
			])

		jacobian = v**2

		return jacobian*f_maxwell_boltzmann(vVec, vEVec)

	return integrate.nquad(integrand, [[0, 2*VESC]])

def L_func(omega, omega_0, width):

	return (
		4.0*omega*omega_0*width*( ( omega**2 - omega_0**2 )**2 + (omega*width)**2 )**(-1)
		)

## KP
def gaussian(omega, omega_0, width):
	# this doesn't seem to do a good job -- lorentzian is better.
	return np.pi * np.sqrt(2) * (width)**(-1) * np.exp(-np.sqrt(2)*(omega - omega_0)**2 / width**2)
