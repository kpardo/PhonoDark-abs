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
import pymatgen.core as pmgCore
import re

from pda.constants import *
import pda.new_physics as physics
import pda.new_diagonalization as diagonalization
import pda.phonopy_funcs as phonopy_funcs
import pda.uvmatrix as uv
from pda import __path__
# import re


@dataclass
class Material:
    name: str
    q_xyz: np.ndarray

    def __post_init__(self):
        [self.dielectric, self.born, self.V_PC, self.m_cell,
            self.bare_ph_energy_o, self.bare_ph_eigen_o] = self.get_phonopy_data()
        self.energies, self.UVmats = self.get_energies()
        self.xi_vec_list = self.get_xi_vecs()
        self.num_pol = len(self.UVmats[0])//2
        self.tex_name = self.get_fancy_name()
        return 0

    def get_energies(self):
        [pol_energy_list, pol_T_list] = diagonalization.calculate_pol_E_T(
            self.name, self.q_xyz)

        # [pol_energy_list, pol_T_list] = uv.UVMatPol(q_vec=self.q_xyz
        #                                             dielectric=self.dielectric
        #                                             V_PC=self.V_pc
        #                                             o_xi_vec=self.xi_vec_list
        #                                             phot_eigenvecs:
        #                                             o_phon_energy=self.bare_ph_energy_o
        #                                             phot_energy:
        #                                             dielectric_diag: np.ndarray
        #                                             K_sq_mat: np.ndarray)
        return pol_energy_list, pol_T_list

    def get_phonopy_data(self):
        dir_path = os.path.join(os.path.dirname(__path__[0]), "data")
        POSCAR_PATH = f"{dir_path}/material_data/{self.name}/POSCAR"
        FORCE_SETS_PATH = f"{dir_path}/material_data/{self.name}/FORCE_SETS"
        BORN_PATH = f"{dir_path}/material_data/{self.name}/BORN"

        phonon_file = phonopy.load(supercell_matrix=supercell_data[self.name],
                                   primitive_matrix='auto',
                                   unitcell_filename=POSCAR_PATH,
                                   force_sets_filename=FORCE_SETS_PATH
                                   )
        ## get list of protons, neutrons, and symbols
        self.Z_list = phonon_file.primitive.get_atomic_numbers()
        self.N_n_list = phonon_file.primitive.get_masses() - self.Z_list
        self.symbols = phonon_file.primitive.symbols

        [num_atoms,
         num_modes,
         pos_red_to_XYZ,
         pos_XYZ_to_red,
         recip_red_to_XYZ,
         recip_XYZ_to_red,
         eq_positions,
         atom_masses,
         born,
         dielectric,
         V_PC] = phonopy_funcs.get_phonon_file_data(phonon_file, self.name)

        self.atom_masses = atom_masses
        m_cell = np.sum(atom_masses)
        [bare_ph_eigen,
         bare_ph_energy] = phonopy_funcs.run_phonopy(phonon_file,
                                                     physics.q_XYZ_list_to_k_red_list(self.q_xyz, recip_XYZ_to_red))

        bare_ph_energy_o = bare_ph_energy[:, 3:]

        bare_ph_eigen_o = bare_ph_eigen[:, 3:, :, :]
        return dielectric, born, V_PC, m_cell, bare_ph_energy_o, bare_ph_eigen_o

    def get_xi_vecs(self):
        xi_vec_list = physics.create_xi_vecs(
            self.born, self.bare_ph_eigen_o, self.atom_masses)
        return xi_vec_list

    def get_Nj(self):
        ## from Tanner
        composition = pmgCore.Composition(self.name)
        oxi_state_guesses = composition.oxi_state_guesses()
        # self.symbols = re.findall(r'[A-Z][a-z]*', self.name)
        
        N_e_list = []
        for s,symbol in enumerate(self.symbols):

            oxi_number = 0

            if len(oxi_state_guesses) >= 1:
                if symbol in oxi_state_guesses[0]:
                    oxi_number = oxi_state_guesses[0][symbol]
            
            N_e_list.append( self.Z_list[s] - oxi_number)

        return np.array(N_e_list)

    def get_fancy_name(self):
        name = self.name
        # FIXME
        pass