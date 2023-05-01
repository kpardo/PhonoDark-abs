'''
dielectric.py
'''

from dataclasses import dataclass
import numpy as np
from constants import *
from material import Material
import couplings as coup
import phonopy
import phonopy_funcs
import new_physics as physics
import new_diagonalization as diagonalization


@dataclass(kw_only=True)
class Dielectric:
    mat: Material
    mass: np.ndarray

    def __post_init__(self):
        self.dielectric = self.get_dielectric()
        self.eps = self.get_eps()
        self.imeps = self.get_imeps()

    def run_material_no_born(self):
        name = self.mat.name
        dir_path = '../'
        POSCAR_PATH = f"{dir_path}/material_data/{name}/POSCAR"
        FORCE_SETS_PATH = f"{dir_path}/material_data/{name}/FORCE_SETS"
        BORN_PATH = f"{dir_path}/material_data/{name}/BORN"

        phonon_file = phonopy.load(supercell_matrix=supercell_data[name],
                                   primitive_matrix='auto',
                                   unitcell_filename=POSCAR_PATH,
                                   force_sets_filename=FORCE_SETS_PATH,
                                   is_nac=False
                                   )
        [num_atoms,
         num_modes,
         pos_red_to_XYZ,
         pos_XYZ_to_red,
         recip_red_to_XYZ,
         recip_XYZ_to_red,
         eq_positions,
         atom_masses,
         born,
         epsinf,
         V_PC] = phonopy_funcs.get_phonon_file_data(phonon_file, name)
        [bare_ph_eigen,
         bare_ph_energy] = phonopy_funcs.run_phonopy(phonon_file,
                                                     physics.q_XYZ_list_to_k_red_list(self.mat.q_xyz, recip_XYZ_to_red))
        bare_ph_energy_o = bare_ph_energy[:, 3:]
        bare_ph_eigen_o = bare_ph_eigen[:, 3:, :, :]
        return bare_ph_energy_o[0], bare_ph_eigen_o[0], V_PC, atom_masses, epsinf

    def get_dielectric(self):
        energies, eigenvectors, vpc, atom_masses, epsinf = self.run_material_no_born()
        widths = 0.0001*np.ones((len(energies)))
        xi = np.einsum('jik, j, nji -> ni', self.mat.born, 1. /
                       np.sqrt(atom_masses), eigenvectors)
        # fix should be xi instead of epsilon --
        eigs = np.einsum('li, lk -> lik', xi, np.conj(xi))
        propdenom = (energies[:, np.newaxis]**2 - self.mass **
                     2 + 1j*widths[:, np.newaxis]*self.mass)**(-1)
        fullprop = np.einsum('lik, lm -> ikm', eigs, propdenom)
        # should have epsinf + the rest returned
        return epsinf[:, :, np.newaxis] + E_EM**2/vpc * fullprop

    def get_eps(self):
        return 1./3.*np.trace(self.dielectric, axis1=0, axis2=1)

    def get_imeps(self):
        # use trick from stack exchange to get inv
        # https://stackoverflow.com/questions/41850712/compute-inverse-of-2d-arrays-along-the-third-axis-in-a-3d-array-without-loops
        inv = np.linalg.inv(self.dielectric.T).T
        return 1./3.*np.trace(inv, axis1=0, axis2=1)
