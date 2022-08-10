# generates data to plot the band structure

import os
import numpy as np 
import phonopy

from constants import *
import diagonalization
import physics
import phonopy_funcs


MATERIAL = 'GaAs'

# create q list

num_q = 1000
q_XYZ_list = []

for q in range(num_q):

	qz = (q+1)/(num_q/0.2)

	q_XYZ_list.append([0, 0, qz])

q_XYZ_list = np.array(q_XYZ_list)

# phonon calculation

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

print('Phonon calculation...')

[bare_ph_eigen, bare_ph_energy] = phonopy_funcs.run_phonopy(phonon_file, 
						physics.q_XYZ_list_to_k_red_list(q_XYZ_list, recip_XYZ_to_red))

bare_o_ph_energy = bare_ph_energy[:, 3:]

print('Done!\n')

# polariton calculation

print('Polariton calculation...')

[pol_energy_list, pol_T_list] = diagonalization.calculate_pol_E_T(MATERIAL, q_XYZ_list)

print('Done!\n')

# write data to file

file = open('./data/'+MATERIAL+'_band_info.csv', 'w')

file.write('dielectric:\n\n')

for i in range(3):
	for j in range(3):

		if j == 2:
			file.write(str(dielectric[i][j]))
		else:
			file.write(str(dielectric[i][j])+',')


	file.write('\n')

file.write('\n')
file.write('q (eV), [ optical phonon energy ], [ polariton energy ]\n\n')

for q in range(num_q):
	file.write(str(np.linalg.norm(q_XYZ_list[q]))+',')

	for nu in range(len(bare_o_ph_energy[q])):
		file.write(str(bare_o_ph_energy[q][nu])+',')

	for lam in range(len(pol_energy_list[q])):

		if lam == len(pol_energy_list[q]) - 1:
			file.write(str(pol_energy_list[q][lam]))
		else:
			file.write(str(pol_energy_list[q][lam])+',')
	file.write('\n')

file.close()








