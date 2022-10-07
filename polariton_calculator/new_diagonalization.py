# Calculates the absorption rate of axions on phonons accounting for the polariton effect
# conventions follow Dropbox/AxionEFTMagnon/Axion/Kevin/'Mar 19, 2020 polariton v3'.pdf

import numpy as np
import math
import os

import phonopy
from constants import *
import phonopy_funcs
import physics
import my_math
import transfer_matrix as tm

###########

def calculate_pol_E_T(MATERIAL, q_XYZ_list):
	"""
	take a material name and list of q points and retruns the energy of the modes
	at each q-point and the T matrix
	"""

	dir_path = os.path.dirname(os.path.realpath(__file__))

	POSCAR_PATH     = dir_path+"/material_data/"+MATERIAL+"/POSCAR"
	FORCE_SETS_PATH = dir_path+"/material_data/"+MATERIAL+"/FORCE_SETS"
	BORN_PATH       = dir_path+"/material_data/"+MATERIAL+"/BORN"

	# phonon information
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

	# k - points to run phonopy
	k_red_list = physics.q_XYZ_list_to_k_red_list(q_XYZ_list, recip_XYZ_to_red)
	n_q = len(q_XYZ_list)

	#####################

	# PHOTON CALCULATION

	photon_eigenvecs_list = physics.create_photon_eigenvec_list(q_XYZ_list, dielectric, born, V_PC, atom_masses)
	photon_eigenvals_list = physics.create_photon_eigenvals_list(q_XYZ_list, dielectric, born, V_PC, atom_masses)

	dielectric_diag_list = photon_eigenvals_list[:, 0]

	# print(dielectric_diag_list)
	photon_energy_list = photon_eigenvals_list[:, 1]

	# check that photon eigenvectors have properties we expect
	physics.check_photon_eigenvectors(q_XYZ_list, dielectric, born, V_PC, atom_masses)

	#####################

	# PHONOPY CALCULATION

	[ph_eigenvectors, ph_omega] = phonopy_funcs.run_phonopy(phonon_file, k_red_list)

	ph_omega_o = ph_omega[:, 3:]
	ph_eigenvectors_o = ph_eigenvectors[:, 3:, :, :]

	o_xi_vecs_list = physics.create_xi_vecs(born, ph_eigenvectors_o, atom_masses)

	physics.check_phonon_eigenvectors(MATERIAL, q_XYZ_list, ph_eigenvectors)
	######################

	# CREATE h MATRIX

	h_mat_dim = 2*(num_modes - 3 + 2)

	g_mat_diag = np.ones(h_mat_dim//2)
	g_mat = np.zeros((h_mat_dim, h_mat_dim))
	g_mat[:h_mat_dim//2, :h_mat_dim//2] = np.diag(g_mat_diag)
	g_mat[h_mat_dim//2:, h_mat_dim//2:] = np.diag(-1*g_mat_diag)

	polariton_energy_list = np.zeros((n_q, num_modes - 3 + 2))
	polariton_T_mat_list = np.zeros((n_q, 2*(num_modes - 3 + 2), 2*(num_modes - 3 + 2)), dtype=complex)

	print('Performing diagonalization...')
	print()
	for q in range(n_q):

		q_vec = q_XYZ_list[q]

		h_mat = physics.create_h_mat(q_vec,
									dielectric,
									V_PC,
									o_xi_vecs_list[q],
									photon_eigenvecs_list[q],
									ph_omega_o[q],
									photon_energy_list[q],
									dielectric_diag_list[q],
									physics.create_K_sq_mat(q_XYZ_list[q],
																born,
																V_PC,
																atom_masses
															)
									)

		#[E_mat, T_mat] = physics.get_E_mat_T_mat(h_mat, h_mat_dim)
		##KP
		[E_mat, T_mat] = tm.TransferMatrix(h_mat).get_T_matrix()

		num_pol_modes = h_mat_dim//2

		polariton_energy_list[q] = 2.0*np.real(np.diag(E_mat)[:num_pol_modes])

		T_mat_inv = np.linalg.inv(T_mat)
		T_mat_inv_dag = np.conj(T_mat_inv.T)
		T_mat_dag = np.conj(T_mat.T)

		T_daghT_mat = np.matmul(T_mat_dag, np.matmul(h_mat, T_mat))
		T_invgT_inv_dag_mat = np.matmul(T_mat_inv, np.matmul(g_mat, T_mat_inv_dag))
		T_g_Tdag = np.matmul(T_mat, np.matmul(g_mat, T_mat_dag))

		# make sure T_mat satisfies canonical commutation relations and diagonalizes h

		atol_param = 10**(-6)

		if not np.allclose(T_daghT_mat, E_mat, atol=atol_param):
			print('Hamiltonian is NOT diagonalized with T matrix.')

			print(np.array_str(T_daghT_mat, precision=4, suppress_small=True))
			print()
			print(np.array_str(E_mat, precision=4, suppress_small=True))

		if not np.allclose(T_invgT_inv_dag_mat, g_mat, atol=atol_param):
			print('T matrix does NOT satisfy commutation relations.')
			print(np.array_str(T_invgT_inv_dag_mat, precision=4, suppress_small=True))

		if not np.allclose(g_mat, T_g_Tdag, atol=atol_param):
			print('T matrix does NOT satisfy commutation relations.')
			print(np.array_str(T_g_Tdag, precision=4, suppress_small=True))

		polariton_T_mat_list[q] = T_mat
	print('Done!')
	print()

	return [polariton_energy_list, polariton_T_mat_list]
