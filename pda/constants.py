# file of constants
# good convention would be to capitalize all constants

import numpy as np
from scipy import special

# using Natural (eV) units
M_PL = 1.22*10**(19 + 9)
M_ELEC = 511.0*10**(3)
M_PROTON = 938 * 10**6
M_NEUTRON = 940 * 10**6
PI = np.pi
E_EM = np.sqrt(4.0*PI/137.0)
THz_To_eV = 6.58*10**(-4)
inveV_To_Ang = 1973.27
Ang_To_inveV = 5.06773*10**(-4)
AMU_To_eV = 9.31*10**8
invcmet_To_eV = 1.973*10**(-5)
GeV_To_eV = 10**9
inveV_To_invGeV = 10**9
kmet_per_sec_to_none = 3.34*10**(-6)

T_To_eV2 = 195.27

RHO_DM = 0.4*GeV_To_eV*invcmet_To_eV**3

KG_YR = 2.69*10**58

# maxwell boltzmann parameters
V0 = 230*kmet_per_sec_to_none
VE = 240*kmet_per_sec_to_none
VESC = 600*kmet_per_sec_to_none

N0 = PI**(3/2)*V0**2*(V0*special.erf(VESC/V0) - (2/np.sqrt(PI))*VESC*np.exp(-VESC**2/V0**2))

# keep track of the supercells used when generating phonon files
supercell_data = {
	'ZnS': [2, 2, 2],
	'CsI': [2, 2, 2],
	'GaAs': [2, 2, 2],
	'SiO2': [3, 3, 3],
	'Al2O3': [2, 2, 2],
	'FeBr2': [4, 4, 3],
	'InSb': [2, 2, 2],
	'LiF': [2, 2, 2],
	'NaCl': [2, 2, 2],
	'MgO': [2, 2, 2],
	'GaSb': [2, 2, 2],
	'NaI': [2, 2, 2],
	'PbS': [2, 2, 2],
	'PbSe': [2, 2, 2],
	'PbTe': [2, 2, 2],
	'CaF2': [3, 3, 3],
	'AlN': [3, 3, 2],
	'CaWO4': [2, 2, 1],
	'MgF2': [2, 2, 2],
	'ZnO': [2, 2, 2],
	'NaF': [2, 2, 2],
	'GaN': [3, 3, 2],
	'Al2O3_db': [2, 2, 1]		
}