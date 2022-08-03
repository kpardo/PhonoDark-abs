# phonopy functions


import os
import phonopy

from constants import *

def run_phonopy(phonon_file, k_mesh):
	"""
	Returns eigenvectors and frequencies in eV
	"""

	# run phonopy in mesh mode 
	phonon_file.run_qpoints(
							k_mesh,
							with_eigenvectors=True
							)

	n_k = len(k_mesh)

	mesh_dict = phonon_file.get_qpoints_dict()

	eigenvectors_pre = mesh_dict['eigenvectors']

	# convert frequencies to correct units
	omega = 2*PI*(THz_To_eV)*mesh_dict['frequencies']

	num_atoms = phonon_file.primitive.get_number_of_atoms()
	num_modes = 3*num_atoms 

	# q, nu, i, alpha
	eigenvectors = np.zeros((n_k, num_modes, num_atoms, 3), dtype=np.complex128)

	# sort the eigenvectors
	for q in range(n_k):
		for nu in range(num_modes):
			eigenvectors[q][nu][:][:] = np.array_split(eigenvectors_pre[q].T[nu], 
												num_atoms)

	return [eigenvectors, omega]

def get_phonon_file_data(phonon_file, material):
	"""
	Returns:

	n_atoms - number of atoms in primitive cell

	n_modes - number of modes = 3*n_atoms

	Transformation matrices

	pos_red_to_XYZ - reduced coordinate positions to Xyz

	pos_XYZ_to_red - XYZ coordinates to red

	recip_red_to_XYZ - reduced coordinates to XYZ

	recip_XYZ_to_red - XYZ coordinates to reduced

	eq_positions - equilibrium positions of atoms

	atom_masses - masses of atoms in eV

	"""

	num_atoms = phonon_file.primitive.get_number_of_atoms()
	num_modes = 3*num_atoms 

	eq_positions = phonon_file.primitive.get_positions()

	atom_masses = AMU_To_eV*phonon_file.primitive.get_masses()

	primitive_mat = phonon_file.primitive.get_cell()

	pos_red_to_XYZ = Ang_To_inveV*np.transpose(primitive_mat)
	pos_XYZ_to_red = np.linalg.inv(pos_red_to_XYZ)

	a_vec = np.matmul(pos_red_to_XYZ, [1, 0, 0])
	b_vec = np.matmul(pos_red_to_XYZ, [0, 1, 0])
	c_vec = np.matmul(pos_red_to_XYZ, [0, 0, 1])

	recip_lat_a = 2*PI*(np.cross(b_vec, c_vec))/(np.matmul(a_vec, np.cross(b_vec, c_vec)))
	recip_lat_b = 2*PI*(np.cross(c_vec, a_vec))/(np.matmul(b_vec, np.cross(c_vec, a_vec)))
	recip_lat_c = 2*PI*(np.cross(a_vec, b_vec))/(np.matmul(c_vec, np.cross(a_vec, b_vec)))

	recip_red_to_XYZ = np.transpose([recip_lat_a, recip_lat_b, recip_lat_c])
	recip_XYZ_to_red = np.linalg.inv(recip_red_to_XYZ)

	dir_path = os.path.dirname(os.path.realpath(__file__))

	POSCAR_PATH     = dir_path+"/material_data/"+material+"/POSCAR"
	FORCE_SETS_PATH = dir_path+"/material_data/"+material+"/FORCE_SETS"
	BORN_PATH       = dir_path+"/material_data/"+material+"/BORN"

	phonon_file2 = phonopy.load(		
								supercell_matrix=supercell_data[material],
								primitive_matrix='auto',
					  			unitcell_filename=POSCAR_PATH,
					  			force_sets_filename=FORCE_SETS_PATH,
					  			born_filename=BORN_PATH
					  		)

	born = phonon_file2.nac_params['born']
	dielectric = phonon_file2.nac_params['dielectric']

	# print('\nVolume of primitive cell = '+str(np.linalg.det(pos_red_to_XYZ)*inveV_To_Ang**3)+' Ang^3\n')
	V_PC = np.linalg.det(pos_red_to_XYZ)

	return [
				num_atoms, 
				num_modes,
				pos_red_to_XYZ,
				pos_XYZ_to_red,
				recip_red_to_XYZ,
				recip_XYZ_to_red, 
				eq_positions,
				atom_masses,
				born,
				dielectric,
				V_PC
			]