# calculates the scattering rate

import os
import numpy as np
import phonopy
import math

from constants import *
import physics
import new_diagonalization as diagonalization
import phonopy_funcs
import my_math

def generate_q_mesh(q_mag, num_q_theta, num_q_phi):

	q_XYZ_list = []


	num_q = num_q_phi*num_q_theta
	for i in range(num_q_phi):
		for j in range(num_q_theta):

			chi = (i + 0.5)/num_q_phi
			beta = (j + 0.5)/num_q_theta

			phi = 2*PI*chi
			theta = np.arccos(2*beta - 1)

			q_XYZ_list.append(
								q_mag*np.array([
								np.sin(theta)*np.cos(phi),
								np.sin(theta)*np.sin(phi),
								np.cos(theta)
								])
							)

	q_XYZ_list = np.array(q_XYZ_list, dtype=np.float64)

	return q_XYZ_list

def calculate_phi_mat(q_XYZ_list, dielectric, T_mat_list, bare_ph_energy, xi_vec_list, vEVec):

	x_hat = np.array([1, 0, 0])
	y_hat = np.array([0, 1, 0])
	z_hat = np.array([0, 0, 1])

	num_q = len(q_XYZ_list)

	num_pol_modes = len(T_mat_list[0])//2

	phi_mat = np.zeros((num_pol_modes, 3, 3), dtype=complex)

	jacob = 4*PI/num_q

	for q in range(num_q):

		q_vec = q_XYZ_list[q]

		q_dir = q_vec / np.linalg.norm(q_vec)

		# get theta/phi values
		theta = math.acos(np.dot(z_hat, q_dir))
		phi = math.atan2(np.dot(q_dir, y_hat),np.dot(q_dir, x_hat))

		int_vel_dist_val = physics.int_vel_dist(theta, phi, vEVec)[0]

		xi_vec = xi_vec_list[q]

		dielectric_inv = np.linalg.inv(dielectric)

		# n_factor = (np.dot(q_dir, np.matmul(dielectric, q_dir)))**(-1)

		T11_conj = np.conj(T_mat_list[q][:num_pol_modes-2, :num_pol_modes])

		T21_conj = np.conj(T_mat_list[q][num_pol_modes:2*num_pol_modes - 2, :num_pol_modes])

		for lam in range(num_pol_modes-2):
			for nu in range(num_pol_modes - 2):
				for nup in range(num_pol_modes - 2):
					for a in range(3):
						for b in range(3):

							phi_mat[lam][a][b] += jacob*int_vel_dist_val*\
									(T11_conj[nu][lam] + T21_conj[nu][lam])*\
									np.conj((T11_conj[nup][lam] + T21_conj[nup][lam]))*\
									np.sqrt(bare_ph_energy_o[q][nu]*bare_ph_energy_o[q][nup])**(-1)*\
									np.conj(np.matmul(dielectric_inv, xi_vec[nu])[a])*\
									np.matmul(dielectric_inv, xi_vec[nup])[b]

	return phi_mat

# def calculate_phi_mat_mix(q_XYZ_list, dielectric, T_mat_list, bare_ph_energy,
# 								xi_vec_list, vEVec):

# 	x_hat = np.array([1, 0, 0])
# 	y_hat = np.array([0, 1, 0])
# 	z_hat = np.array([0, 0, 1])

# 	num_q = len(q_XYZ_list)

# 	num_pol_modes = len(T_mat_list[0])//2

# 	phi_mat = np.zeros((num_pol_modes, 3, 3), dtype=complex)

# 	jacob = 4*PI/num_q

# 	for q in range(num_q):

# 		q_vec = q_XYZ_list[q]

# 		q_dir = q_vec / np.linalg.norm(q_vec)

# 		# get theta/phi values
# 		theta = math.acos(np.dot(z_hat, q_dir))
# 		phi = math.atan2(np.dot(q_dir, y_hat),np.dot(q_dir, x_hat))

# 		int_vel_dist_val = physics.int_vel_dist(theta, phi, vEVec)[0]

# 		xi_vec = xi_vec_list[q]

# 		n_factor = (np.dot(q_dir, np.matmul(dielectric, q_dir)))**(-1)

# 		T11_conj = np.conj(T_mat_list[q][:num_pol_modes-2, :num_pol_modes])

# 		T21_conj = np.conj(T_mat_list[q][num_pol_modes:2*num_pol_modes - 2, :num_pol_modes])

# 		mixing_matrix = np.linalg.inv(T_mat_list[q])[:num_pol_modes, num_pol_modes - 2:num_pol_modes]

# 		for lam in range(num_pol_modes-2):

# 			mixing_param_sq = mixing_matrix[lam][0]*np.conj(mixing_matrix[lam][0]) + \
# 								mixing_matrix[lam][1]*np.conj(mixing_matrix[lam][1])

# 			for nu in range(num_pol_modes - 2):
# 				for nup in range(num_pol_modes - 2):
# 					for a in range(3):
# 						for b in range(3):

# 							phi_mat[lam][a][b] += mixing_param_sq*jacob*int_vel_dist_val*(n_factor)**2*\
# 									(T11_conj[nu][lam] + T21_conj[nu][lam])*\
# 									np.conj((T11_conj[nup][lam] + T21_conj[nup][lam]))*\
# 									np.sqrt(bare_ph_energy_o[q][nu]*bare_ph_energy_o[q][nup])**(-1)*\
# 									np.conj(xi_vec[nu][a])*xi_vec[nup][b]

# 	return phi_mat


def rate1(m, pol_energy_list, width_list, b_field, phi_mat):

	rate = 0.0

	for lam in range(len(pol_energy_list[0]) - 2):

		rate += (1/m_cell)*(E_EM**2*RHO_DM)*m**(-2)*\
					physics.L_func(m, pol_energy_list[0][lam], width_list[lam])*\
					np.dot(b_field, np.matmul(phi_mat[lam], b_field))

	return rate


## KP
def rate1_b_average(m, pol_energy_list, width_list, phi_mat, widthfunc='lorentzian'):

	# rate = 0.0
	# for lam in range(len(pol_energy_list[0]) - 2):
	#
	# 	rate += (1/m_cell)*(E_EM**2*RHO_DM)*m**(-2)*\
	# 				physics.L_func(m, pol_energy_list[0][lam], width_list[lam])*\
	# 				(1/3)*T_To_eV2**2*np.trace(phi_mat[lam])
	prefac = (1/m_cell)*(E_EM**2*RHO_DM)*m**(-2)*(1./3.)*T_To_eV2**2
	if widthfunc == 'lorentzian':
		width = physics.L_func(m, pol_energy_list[0], width_list)
	elif widthfunc == 'gaussian':
		width = physics.gaussian(m, pol_energy_list[0], width_list)
	return np.sum(prefac * width * np.trace(phi_mat, axis1=1, axis2=2))


def gayy_reach(m, exposure, pol_energy_list, width_list, b_field, phi_mat, n_cut):
	"""
	returns the reach in gayy in GeV
	"""

	return np.sqrt(n_cut/(rate1(m, pol_energy_list, width_list, b_field, phi_mat)*
						exposure*KG_YR))*inveV_To_invGeV


def gayy_reach_b_average(m, exposure, pol_energy_list, width_list, phi_mat, n_cut):
	"""
	returns the reach in gayy in GeV
	"""

	return np.sqrt(n_cut/(rate1_b_average(m, pol_energy_list, width_list, phi_mat)*
						exposure*KG_YR))*inveV_To_invGeV


def get_rel_mode_contribution(phi_mat):

	phi_mat_trace = np.zeros(len(phi_mat))

	for lam in range(len(phi_mat) - 2):

		phi_mat_trace[lam] = np.real(np.trace(phi_mat[lam]))

	phi_mat_trace_max = max(phi_mat_trace)

	phi_mat_trace = phi_mat_trace / phi_mat_trace_max

	return phi_mat_trace

def get_mode_contribution(phi_mat):

	phi_mat_trace = np.zeros(len(phi_mat))

	for lam in range(len(phi_mat) - 2):

		phi_mat_trace[lam] = np.real(np.trace(phi_mat[lam]))

	return phi_mat_trace

def check_U_plus_V(T_mat):

	num_pol = len(T_mat)//2

	print(num_pol)
	T11_conj = np.conj(T_mat[:num_pol, :num_pol])

	T21_conj = np.conj(T_mat[num_pol:, :num_pol])

	UpV = T11_conj + T21_conj

	print('U_(nup, nu)^*_p + V_(nup, nu)_(-p) : ')
	print()

	for i in range(num_pol):
		for j in range(num_pol):

			if UpV[i][j]*np.conj(UpV[i][j]) < 10**(-6):
				UpV[i][j] = 0

	my_math.matprint(UpV*np.conj(UpV))
	print()


def compare_mats(m1, m2, tol):
	"""
	compares two matrices, if the maximum difference between them is less than tol
	returns True, otherwise False
	"""

	diff_mat = m1 - m2

	diff_mat_sq = (m1 - m2)*np.conj(m1 - m2)

	max_diff_sq = np.amax(diff_mat_sq)

	if max_diff_sq < tol:
		return [True, max_diff_sq]
	else:
		return [False, max_diff_sq]


def check_T_mat_properties(q_XYZ_list, MATERIAL):

	print('!! CHECKING PROPERTIES OF T matrix AT EACH Q')
	print('!! COMMENT OUT FOR FASTER SPEED')

	[pol_energy_list, pol_T_list] = diagonalization.calculate_pol_E_T(MATERIAL, q_XYZ_list)

	[pol_energy_list_mq, pol_T_list_mq] = diagonalization.calculate_pol_E_T(MATERIAL, -1*q_XYZ_list)

	num_pol = len(pol_energy_list_mq[0])

	for q in range(len(q_XYZ_list)):

		T11q = pol_T_list[q][:num_pol, :num_pol]
		T12q = pol_T_list[q][:num_pol, num_pol:]
		T21q = pol_T_list[q][num_pol:, :num_pol]
		T22q = pol_T_list[q][num_pol:, num_pol:]

		T11mq = pol_T_list_mq[q][:num_pol, :num_pol]
		T12mq = pol_T_list_mq[q][:num_pol, num_pol:]
		T21mq = pol_T_list_mq[q][num_pol:, :num_pol]
		T22mq = pol_T_list_mq[q][num_pol:, num_pol:]

		tol_param = 10**(-3)

		if not compare_mats(T11q, np.conj(T22mq), tol_param)[0]:
			print("T11_q != T22_(-q)^*")
			print(q)
			print('Max difference :')
			print(compare_mats(T11q, np.conj(T22mq), tol_param)[1])

			# for nu in range(num_pol):
			# 	print(nu)
			# 	print(compare_mats(T11q[nu], np.conj(T22mq)[nu], tol_param)[1])

			# print(T11q[28])
			# print(np.conj(T22mq)[28])
			# print((T11q[28] - np.conj(T22mq)[28])*np.conj(T11q[28] - np.conj(T22mq)[28]))
			# print()
			# exit()
		if not compare_mats(T12q, np.conj(T21mq), tol_param):
			print("T12q != T21_(-q)^*")
			print(q)
			print(compare_mats(T12q, np.conj(T21mq), tol_param)[1])
			print()

		if not compare_mats(T21q, np.conj(T12mq), tol_param):
			print("T21q != T12_(-q)^*")
			print(q)
			print(compare_mats(T21q, np.conj(T12mq), tol_param)[1])
			print()

		if not compare_mats(T22q, np.conj(T11mq), tol_param):
			print("T22_q != T11_(-q)^*")
			print(q)
			print(compare_mats(T22q, np.conj(T11mq), tol_param)[1])
			print()

##################################
dir_path = os.path.dirname(os.path.realpath(__file__))
all_material_list = os.listdir(os.path.join(dir_path, 'material_data'))

b_field_mag = 10 # T
exp = 1			 # kg-yr

n_pol_cut = 3

# born_diag = False
# born_sym = False

# # change BORN to identity matrix (check that rate -> 0 when all atoms couple in the same way)
# abs_born = False

run_dict = {
				'materials': ['GaAs', 'SiO2', 'Al2O3', 'CaWO4'],
				'bfield': [
							T_To_eV2*np.array([b_field_mag, 0, 0]),
							T_To_eV2*np.array([0, b_field_mag, 0]),
							T_To_eV2*np.array([0, 0, b_field_mag]),
							'average'
						],
				'descrip': [
							'Bx',
							'By',
							'Bz',
							'Baverage'
				]
			}

for m in range(len(run_dict['materials'])):

	MATERIAL = run_dict['materials'][m]

	print('Material number '+str(m + 1)+' / '+str(len(run_dict['materials']))+' : '+MATERIAL)
	print()

	q_XYZ_list = generate_q_mesh(10**(-4), 5, 5)

	# print(q_XYZ_list[12])

	# check that T matrices satisfy the q -> -q properties
	# q_XYZ_list = np.array([[10**(-4), 0, 0]])

	check_T_mat_properties(q_XYZ_list, MATERIAL)

	[pol_energy_list, pol_T_list] = diagonalization.calculate_pol_E_T(MATERIAL, q_XYZ_list)

	# check_U_plus_V(pol_T_list[0])

	# exit()

	num_pol = len(pol_T_list[0])//2

	# check_U_plus_V(pol_T_list[0])

	dir_path = os.path.dirname(os.path.realpath(__file__))

	# exit()

	# zero_pt = np.zeros(2, dtype=complex)

	# for lam in range(2):
	# 	for nu in range(num_pol):
	# 		zero_pt[lam] += T21[num_pol-2+lam][nu]*np.conj(T21[num_pol-2+lam][nu])

	# print(zero_pt[lam])
	# print()

	# my_math.matprint(T11[num_pol-2:, :]*np.conj(T11[num_pol-2:, :]) +
	# 				T21[num_pol-2:, :]*np.conj(T21[num_pol-2:, :]))

	# exit()

	# Umat = pol_T_list[0][num_pol - 2:num_pol, :num_pol]

	# print('T_22_q')
	# my_math.matprint(pol_T_list[0][num_pol:, num_pol:])
	# print()
	# print('T_11_(-q)^*')
	# my_math.matprint(np.conj(pol_T_list[1][:num_pol, :num_pol]))

	# exit()




	# my_math.matprint(pol_T_list[0][2*num_pol - 2:2*num_pol, num_pol:2*num_pol])

	# Vmat = pol_T_list[0][num_pol - 2:num_pol, num_pol:]

	# exit()

	# mixing_matrix_sq = np.zeros((num_pol, 2), dtype=complex)

	# for nu in range(num_pol):
	# 	for lam in range(2):

	# 		mixing_matrix_sq[nu][lam] = ( Umat[lam][nu]*np.conj(Umat[lam][nu]) +\
	# 										Vmat[lam][nu]*np.conj(Vmat[lam][nu])
	# 									)

	# file = open('./data/'+MATERIAL+'_mixing_matrix.csv', 'w')

	# file.write('sum_lam <0| N_lam |0>\n')



	# file.write(str(np.real(zero_pt))+'\n')
	# file.write('\n')

	# file.write('theta^2_(nu, lam) == |U_(3n+lam, nu)|^2 + |V_(3n+lam, nu)|^2,'+
	# 			' theta_nu0^2 + theta_nu1^2, U_(3n+lam, nu),V_(3n+lam, nu), E_nup\n')
	# for nup in range(num_pol):
	# 	file.write(
	# 				str(np.real(mixing_matrix_sq[nup][0]))+' , '
	# 				+str(np.real(mixing_matrix_sq[nup][1]))+' , '+
	# 				str(
	# 					np.real(mixing_matrix_sq[nup][0] + mixing_matrix_sq[nup][1])
	# 					)+' , '+
	# 				str(Umat[:, nup])+' , '+
	# 				str(Vmat[:, nup])+' , '+
	# 				str(pol_energy_list[0][nup])+'\n')
	# file.close()

	# exit()

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

	# if born_diag:

	# 	for i in range(len(born)):
	# 		born_xx = born[i, 0, 0]
	# 		born_yy = born[i, 1, 1]
	# 		born_zz = born[i, 2, 2]

	# 		born[i] = [[born_xx, 0, 0], [0, born_yy, 0], [0, 0, born_zz]]

	# if born_sym:

	# 	for i in range(len(born)):

	# 		old_born = born[i]

	# 		born[i] = (0.5)*(old_born + old_born.T)

	if not np.allclose(sum(born), np.zeros((3,3))):
		print('Acoustic sum rule violated.')
		print(sum(born))

	m_cell = np.sum(atom_masses)

	# width_list = 10**(-2)*pol_energy_list[0]
	width_list = 10**(-3)*np.ones((len(pol_energy_list[0])))
	# width_list = 10**(-1)*pol_energy_list[0]
	# width_list = 0.075*np.ones((len(pol_energy_list[0])))
	[bare_ph_eigen, bare_ph_energy] = phonopy_funcs.run_phonopy(phonon_file,
						physics.q_XYZ_list_to_k_red_list(q_XYZ_list, recip_XYZ_to_red))

	bare_ph_energy_o = bare_ph_energy[:, 3:]

	bare_ph_eigen_o = bare_ph_eigen[:, 3:, :, :]

	xi_vec_list = physics.create_xi_vecs(born, bare_ph_eigen_o, atom_masses)

	print('Computing phi matrix for '+MATERIAL+'...')
	print()

	# if abs_born:
	# 	print('EDITING BORN')
	# 	for j in range(len(born)):

	# 		born[j] = np.diag(np.ones(3))

	# 	xi_vec_list = physics.create_xi_vecs(born, bare_ph_eigen_o, atom_masses)


	phi_mat = calculate_phi_mat(q_XYZ_list, dielectric, pol_T_list, bare_ph_energy, xi_vec_list,
									np.array([0, 0, VE]))

	mode_contrib = get_rel_mode_contribution(phi_mat)

	file = open('./data/'+MATERIAL+'_rel_mode_contribution_and_energy.csv', 'w')
	file.write('relative contribution to rate (dimensionless), Energy (eV) \n')

	for lam in range(len(pol_energy_list[0]) - 2):
		file.write(str(mode_contrib[lam])+' , '+str(pol_energy_list[0][lam])+'\n')

	file.close()

	# phi_mat_mix = calculate_phi_mat_mix(q_XYZ_list, dielectric, pol_T_list, bare_ph_energy, xi_vec_list,
	# 								np.array([0, 0, VE]))

	print('Done!\n')

	for f in range(len(run_dict['bfield'])):

		B_field = run_dict['bfield'][f]
		descrip = run_dict['descrip'][f]

		print('Computing rate for B field : '+str(B_field))
		print(f'Description is {descrip}')
		print()

		file = open('./data/'+MATERIAL+'_gayy_Reach_'+descrip+'.csv', 'w')
		file.write('m (eV), gayy (GeV^(-1))\n')

		# file2 = open('./data/'+MATERIAL+'_photon_gayy_Reach_'+descrip+'.csv', 'w')
		# file2.write('m (eV), gayy (GeV^(-1))\n')

		## KP
		num_m = int(1.e6)
		m_list = np.logspace(-2, 0, num_m)
		indices = np.searchsorted(m_list, pol_energy_list[0,:-2])
		m_list = np.insert(m_list, indices, pol_energy_list[0,:-2])
		## to -2 in pol_energy_list to avoid acoustic modes.


		if not isinstance(B_field, str):
			print('skipping')
			# for m in m_list:
			#
			# 	file.write(str(m)+', '
			# 		+str(np.real(gayy_reach(m, exp, pol_energy_list, width_list,
			# 								B_field, phi_mat, n_pol_cut)))+'\n')

				# file2.write(str(m_list[mass_i])+', '
				# 	+str(np.real(gayy_reach(m_list[mass_i], exp, pol_energy_list, width_list,
				# 							B_field, phi_mat_mix, n_phot_cut)))+'\n')


		else:

			for m in m_list:

				file.write(str(m)+', '
					+str(np.real(gayy_reach_b_average(m, exp*b_field_mag**2, pol_energy_list,
														width_list, phi_mat, n_pol_cut)))+'\n')

				# file2.write(str(m_list[mass_i])+', '
				# 	+str(np.real(gayy_reach_b_average(m_list[mass_i], exp*b_field_mag**2, pol_energy_list,
				# 										width_list, phi_mat_mix, n_phot_cut)))+'\n')

		print('Done!')
		print()

		file.close()
