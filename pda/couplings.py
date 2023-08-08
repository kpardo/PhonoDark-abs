'''
couplings.py
'''

from dataclasses import dataclass
import numpy as np
from pda.constants import *

@dataclass
class ScalarE:
    q_XYZ_list: np.ndarray
    omega: np.ndarray ## e.g., DM masses
    mat: None
    name: str = 'scalar_e'
    texname: str = r'$\mathrm{Scalar~DM}$'
    texop: str = r'$d_{\phi e e} \frac{\sqrt{4\pi}m_e}{M_{\mathrm{Pl}}} \phi \bar{\psi}\psi$'
    texcoupconst: str = r'$d_{\phi e e}$'
    formfac: np.ndarray = np.zeros((1))
    prefac: np.float64 = 1. * (4 * np.pi)**(-1) * (M_ELEC/M_PL)**2
    # prefac: np.float64 = 1.
    se_shape: str = 'scalar'
    mixing_phia: np.ndarray = np.zeros((1))
    mixing: bool = False
    ce: np.float64 = 1.
    cp: np.float64 = 0.
    cn: np.float64 = 0.

    def __post_init__(self):
        Nj = self.mat.get_Nj()
        Np = self.mat.Z_list
        Nn = Nn = self.mat.N_n_list
        qmag = self.omega*V0
        qhat = 1./np.linalg.norm(self.q_XYZ_list, axis=1)[:, np.newaxis]*self.q_XYZ_list
        Fe =  np.einsum('j, w, ki -> jwki', Nj, qmag, qhat)
        Fp =  np.einsum('j, w, ki -> jwki', Np, qmag, qhat)
        Fn =  np.einsum('j, w, ki -> jwki', Nn, qmag, qhat)
        self.formfac = self.ce * Fe + self.cp * Fp + self.cn * Fn
        if self.mixing:
            epsfactor = (1- 1./3.*np.trace(self.mat.dielectric))
            self.mixing_A_e = self.ce * epsfactor * E_EM**(-1) * ( 
                    np.einsum('w, ki -> wki', 
                            qmag*self.omega,
                            qhat))

@dataclass
class DarkPhoton:
    q_XYZ_list: np.ndarray
    omega: np.ndarray  # e.g., DM masses
    mat: None
    name: str = 'dark_photon'
    texname: str = r'$\mathrm{Vector~DM}$'
    texop: str = r'$\kappa e \phi_\mu\bar\psi \gamma^\mu\psi$'
    texcoupconst: str = r'$\kappa$'
    formfac: np.ndarray = np.zeros((1))
    prefac: np.float64 = 1. # e's cancel from conversion from g to kappa
    se_shape: str = 'vector'
    ce: np.float64 = 1.
    cp: np.float64 = -1.
    cn: np.float64 = 0.
    mixing: bool = False

    def __post_init__(self):
        Ne = self.mat.get_Nj()
        Np = self.mat.Z_list
        Nn = self.mat.N_n_list
        Fe = np.einsum('j, w, ab -> jwab', Ne, self.omega, np.identity(3))
        Fp = np.einsum('j, w, ab -> jwab', Np, self.omega, np.identity(3))
        Fn = np.einsum('j, w, ab -> jwab', Nn, self.omega, np.identity(3))
        self.formfac = self.ce * Fe + self.cp * Fp + self.cn * Fn
        if self.mixing:
            eps_infty = (1/3.0)*np.trace(self.mat.dielectric)
            piaa_e = self.omega**2 * ( 1.0 - eps_infty )
            self.mixing_A_e =  E_EM**(-1) * \
                np.einsum('w, ab -> wab', piaa_e, np.identity(3))

@dataclass
class BminusL:
    q_XYZ_list: np.ndarray
    omega: np.ndarray  # e.g., DM masses
    mat: None
    name: str = 'bminsl'
    texname: str = r'$\mathrm{U}(1)\mathrm{b}$'
    texop: str = r'$g_B \phi_\mu\bar\psi \gamma^\mu\psi$'
    texcoupconst: str = r'$g_{B-L}$'
    formfac: np.ndarray = np.zeros((1))
    prefac: np.float64 = 1.
    se_shape: str = 'vector'
    ce: np.float64 = 1.
    cp: np.float64 = 0.
    cn: np.float64 = 0.
    mixing: bool = False

    def __post_init__(self):
        Ne = self.mat.get_Nj()
        Np = self.mat.Z_list
        Nn = self.mat.N_n_list
        Fe = np.einsum('j, w, ab -> jwab', Ne, self.omega, np.identity(3))
        Fp = np.einsum('j, w, ab -> jwab', Np, self.omega, np.identity(3))
        Fn = np.einsum('j, w, ab -> jwab', Nn, self.omega, np.identity(3))
        self.formfac = self.ce * Fe + self.cp * Fp + self.cn * Fn
        if self.mixing:
            eps_infty = (1/3.0)*np.trace(self.mat.dielectric)
            piaa_e = self.omega**2 * ( 1.0 - eps_infty )
            self.mixing_A_e =  E_EM**(-1) * \
                np.einsum('w, ab -> wab', piaa_e, np.identity(3))

@dataclass
class ElectricDipole:
    q_XYZ_list: np.ndarray
    omega: np.ndarray  # e.g., DM masses
    mat: None
    S: np.ndarray = np.zeros((1)) # magnetic spin vector
    mo: bool = False
    name: str = 'electricdipole'
    texname: str = r'$\mathrm{Electric~Dipole}$'
    texop: str = r'$\frac{g_\chi}{4m_\psi}\phi_{\mu\nu}\bar\psi\sigma^{\mu\nu}i\gamma^5\psi$'
    texcoupconst: str = r'$d_{E}$'
    formfac: np.ndarray = np.zeros((1))
    prefac: np.float64 = 1.
    se_shape: str = 'dim5'
    ce: np.float64 = 1.
    cp: np.float64 = 0.
    cn: np.float64 = 0.
    mixing: bool = False
    

    def __post_init__(self):
        if self.mo:
            Nj = self.mat.get_Nj()
            self.S = self.S*np.ones((len(Nj), 3))
            qhat = 1./np.linalg.norm(self.q_XYZ_list, axis=1)[:, np.newaxis]*self.q_XYZ_list
            
            Fe = (np.einsum('w, kc, jc, ab -> jwabk', 
                                    2.*1j*self.omega**2*V0, 
                                    qhat, self.S, np.identity(3)))
            self.formfac = self.ce * Fe
            ##FIXME: add cn and cp terms.
        elif ~self.mo:
            ##FIXME: should just return 0s
            self.formfaci0 = np.zeros(np.shape(self.q_XYZ_list))
            self.formfacij = np.einsum('j, abc, ic -> jiab', -1.*self.omega, self.levicivita(), self.q_XYZ_list)

    def levicivita(self):
        eijk = np.zeros((3, 3, 3))
        eijk[0, 1, 2] = eijk[1, 2, 0] = eijk[2, 0, 1] = 1
        eijk[0, 2, 1] = eijk[2, 1, 0] = eijk[1, 0, 2] = -1
        return eijk


@dataclass
class MagneticDipole:
    q_XYZ_list: np.ndarray
    omega: np.ndarray  # e.g., DM masses
    mat: None
    S: np.ndarray = np.zeros((1))  # magnetic spin vector
    mo: bool = False ## magnetic ordering
    name: str = 'magneticdipole'
    texname: str = r'$\mathrm{Magnetic~Dipole}$'
    texop: str = r'$\frac{g_\chi}{4m_\psi}\phi_{\mu\nu}\bar\psi\sigma^{\mu\nu}\psi$'
    texcoupconst: str = r'$d_{M}$'
    formfac: np.ndarray = np.zeros((1))
    # prefac: np.float64 = 1. / E_EM**2
    prefac: np.float64 = 1.
    se_shape: str = 'dim5'
    ce: np.float64 = 1.
    cp: np.float64 = 0.
    cn: np.float64 = 0.
    mixing: bool = False

    def __post_init__(self):
        if self.mo:
            self.se_shape = 'dim5_s'
            Nj = self.mat.get_Nj()
            self.S = self.S*np.ones((len(Nj), 3))
            Fe = (np.einsum('w, jb, bai -> jwabi',
                            2.*1j*self.omega**2,
                            self.S, self.levicivita()))
            self.formfac = self.ce * Fe
        else:
            self.se_shape = 'vector'
            Nj = self.mat.get_Nj()
            Fe = (np.einsum('w, j, ab -> jwab',
                            self.omega**3/(2.*M_ELEC),
                            Nj, np.eye(3,3)))
            self.formfacij = self.ce * Fe

    def levicivita(self):
        ## stack exchange, probably.
        eijk = np.zeros((3, 3, 3))
        eijk[0, 1, 2] = eijk[1, 2, 0] = eijk[2, 0, 1] = 1
        eijk[0, 2, 1] = eijk[2, 1, 0] = eijk[1, 0, 2] = -1
        return eijk

@dataclass
class Axion:
    q_XYZ_list: np.ndarray
    omega: np.ndarray  # e.g., DM masses
    S: np.ndarray # magnetic spin vector
    mat: None = None # don't really need, but whatever.
    mixing: bool = False
    fermion_coupling: str = 'e'
    name: str = 'axion'
    texname: str = r'$\mathrm{Axion}$'
    formfac: np.ndarray = np.zeros((1))
    prefac: np.float64 = 1.
    se_shape: str = 'scalar'

    def __post_init__(self):
        self.texcoupconst, mfermion = self.get_coupconst()
        self.formfac = np.einsum('w,jb -> jwb', -1j*self.omega**2/mfermion, self.S)

    def get_coupconst(self):
        if self.fermion_coupling == 'e':
            Nj = self.mat.get_Nj()
            self.S = self.S*np.ones((len(Nj), 3))
            return r'$g_{aee}$', M_ELEC
        elif self.fermion_coupling == 'n':
            Nn = self.mat.N_n_list
            self.S = self.S*np.ones((len(Nn), 3))
            self.S = Nn[:, np.newaxis]*self.S
            return r'$g_{ann}$', M_NEUTRON
        elif self.fermion_coupling == 'p':
            Np = self.mat.Z_list
            self.S = self.S*np.ones((len(Np), 3))
            self.S = Np[:, np.newaxis]*self.S
            return r'$g_{app}$', M_PROTON
        else:
            raise Warning('Not a valid fermion type!')