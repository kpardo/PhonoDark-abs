'''
dielectric.py
'''

from dataclasses import dataclass
import numpy as np
from pda.constants import *
from pda.material import Material
import pda.couplings as coup
import phonopy
import pda.phonopy_funcs as phonopy_funcs
import pda.new_physics as physics
import pda.new_diagonalization as diagonalization


@dataclass(kw_only=True)
class Dielectric:
    mat: Material
    mass: np.ndarray
    width_val: float = 0.01
    width_type: str = 'best'

    def __post_init__(self):
        self.dielectric = self.get_dielectric()
        self.eps = self.get_eps()
        self.imeps = self.get_imeps()

    def run_material_no_born(self):
        name = self.mat.name
        dir_path = '../data'
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
                                                     physics.q_XYZ_list_to_k_red_list(self.mat.q_XYZ_list, recip_XYZ_to_red))
        bare_ph_energy_o = bare_ph_energy[:, 3:]
        bare_ph_eigen_o = bare_ph_eigen[:, 3:, :, :]
        return bare_ph_energy_o[0], bare_ph_eigen_o[0], V_PC, atom_masses, epsinf

    def get_dielectric(self):
        energies, eigenvectors, vpc, atom_masses, epsinf = self.run_material_no_born()
        if self.width_type == 'constant':
            widths = self.width_val*np.ones((len(energies)))
            propdenom = (energies[:, np.newaxis]**2 - self.mass **
                     2 + 1j*widths[:, np.newaxis]*self.mass)**(-1)
        elif self.width_type == 'best':
            widths = self.width_val*energies
            propdenom = (energies[:, np.newaxis]**2 - self.mass **
                     2 - 1j*widths[:, np.newaxis]*self.mass)**(-1)
        xi = np.einsum('jik, j, mjk -> mi', self.mat.born, 1. /
                       np.sqrt(atom_masses), eigenvectors)
        eigs = 1./3.*np.einsum('mi, mi -> m', xi, np.conj(xi))
        fullprop = np.einsum('m, mw -> w', eigs, propdenom)
        return 1./3.*np.trace(epsinf) + E_EM**2/(vpc) * fullprop

    def get_eps(self):
        # return 1./3.*np.trace(self.dielectric, axis1=0, axis2=1)
        return self.dielectric

    def get_imeps(self):
        # use trick from stack exchange to get inv
        # https://stackoverflow.com/questions/41850712/compute-inverse-of-2d-arrays-along-the-third-axis-in-a-3d-array-without-loops
        # inv = np.linalg.inv(self.dielectric.T).T
        # return 1./3.*np.trace(inv, axis1=0, axis2=1)
        return 1./self.eps
