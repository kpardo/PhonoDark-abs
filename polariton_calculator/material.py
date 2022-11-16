'''
material.py
defines the material class.
'''

from dataclasses import dataclass
import numpy as np
from scipy import linalg as sla
import sys
import phonopy
import os

from constants import *
import new_physics as physics
import new_diagonalization as diagonalization
import phonopy_funcs

@dataclass
class Material:
    name: str
    q_xyz: np.ndarray

    def __post_init__(self):
        self.energies, self.UVmats = self.get_energies()
        self.num_pol = len(self.UVmats[0])//2
        self.dielectric, self.born, self.V_PC, self.m_cell, self.bare_ph_energy_o, self.bare_ph_eigen_o = self.get_phonopy_data()
        self.xi_vec_list = self.get_xi_vecs()
        return 0

    def get_energies(self):
        [pol_energy_list, pol_T_list] = diagonalization.calculate_pol_E_T(self.name, self.q_xyz)
        return pol_energy_list, pol_T_list

    def get_phonopy_data(self):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        POSCAR_PATH     = f"{dir_path}/material_data/{self.name}/POSCAR"
        FORCE_SETS_PATH = f"{dir_path}/material_data/{self.name}/FORCE_SETS"
        BORN_PATH       = f"{dir_path}/material_data/{self.name}/BORN"

        phonon_file = phonopy.load(		supercell_matrix=supercell_data[self.name],
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
        	V_PC		] = phonopy_funcs.get_phonon_file_data(phonon_file, self.name)


        self.atom_masses = atom_masses
        m_cell = np.sum(atom_masses)
        [bare_ph_eigen, bare_ph_energy] = phonopy_funcs.run_phonopy(phonon_file,
        					physics.q_XYZ_list_to_k_red_list(self.q_xyz, recip_XYZ_to_red))

        bare_ph_energy_o = bare_ph_energy[:, 3:]

        bare_ph_eigen_o = bare_ph_eigen[:, 3:, :, :]
        return dielectric, born, V_PC, m_cell, bare_ph_energy_o, bare_ph_eigen_o

    def get_xi_vecs(self):
        xi_vec_list = physics.create_xi_vecs(self.born, self.bare_ph_eigen_o, self.atom_masses)
        return xi_vec_list
