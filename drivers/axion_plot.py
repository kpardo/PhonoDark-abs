import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator
import numpy as np
import seaborn as sns
import astropy.units as u
import pandas as pd

import pda.material as material
import pda.transfer_matrix as tm
import pda.selfenergy as se
import pda.rate as r
import pda.reach as re
import pda.constants as const
import pda.couplings as coup

from mp_api.client import MPRester


# FIXME: put into plotting script.
""
## set fig params
sns.set_context("paper")
sns.set_style('ticks')
sns.set_palette('colorblind')
figparams = {
        'text.latex.preamble': r'\usepackage{amsmath} \boldmath',
        'text.usetex':True,
        'axes.labelsize':22.,
        'xtick.labelsize':16,
        'ytick.labelsize':16,
        'figure.figsize':[10., 8.],
        'font.family':'DejaVu Sans',
        'legend.fontsize':18}
plt.rcParams.update(figparams)
cs = plt.rcParams['axes.prop_cycle'].by_key()['color']
import matplotlib
matplotlib.use('Agg')

u.set_enabled_equivalencies(u.mass_energy())

qs = r.generate_q_mesh(10**(-4), 5, 5)
mlist = np.logspace(-2,np.log10(0.5),int(1e3));

## load other limits
rg = pd.read_csv('../data/limit_data/AxionLimits/RedGiants.txt', skiprows=2, delimiter='\s+', names=['mass [eV]', 'g'])
ns_ann = pd.read_csv('../data/limit_data/AxionLimits/NeutronStars_ann.txt', skiprows=3, delimiter='\s+', names=['mass [eV]', 'g'])
ns_app = pd.read_csv('../data/limit_data/AxionLimits/NeutronStars_app.txt', skiprows=3, delimiter='\s+', names=['mass [eV]', 'g'])

# QCD line, g_aee
C_ae_DFSZ_upper = 1.0/3.0
C_ae_DFSZ_lower = 0.024
C_ae_KSVZ = 2e-4
## FIXME This is all a mess.

def generate_QCD_axion_line_gaee(log10_m_min, log10_m_max, C_ae, n = 2):
    """
        Generates interpolated limit corresponding the QCD axion line.
    """
    
    log10_masses = np.linspace(log10_m_min, log10_m_max, n)
    
    return [ log10_masses, [ np.log10(8.943e-11*C_ae*10.**log10_m) for log10_m in log10_masses ] ]

# QCD line, g_ann
C_an_DFSZ_upper = 0.16
C_an_DFSZ_lower = 0.26
C_an_KSVZ = 0.02

def generate_QCD_axion_line_gann(log10_m_min, log10_m_max, C_an, n = 2):
    """
        Generates interpolated limit corresponding the QCD axion line.
    """
    
    log10_masses = np.linspace(log10_m_min, log10_m_max, n)
    
    return [ log10_masses, [ np.log10(1.644e-7*C_an*10.**log10_m) for log10_m in log10_masses ] ]

# QCD line, g_app
C_ap_DFSZ_upper = 0.2
C_ap_DFSZ_lower = 0.6
C_ap_KSVZ = 0.47

def generate_QCD_axion_line_gapp(log10_m_min, log10_m_max, C_ap, n = 2):
    """
        Generates interpolated limit corresponding the QCD axion line.
    """
    
    log10_masses = np.linspace(log10_m_min, log10_m_max, n)
    
    return [ log10_masses, [ np.log10(1.64e-7*C_ap*10.**log10_m) for log10_m in log10_masses ] ]
log10_m_min = 1
log10_m_max = 3

[ log10_m_eV, log10_gaee_KSVZ ] = generate_QCD_axion_line_gaee(log10_m_min - 3, log10_m_max - 3, C_ae_KSVZ)
[ log10_m_eV, log10_gaee_DFSZ_lower ] = generate_QCD_axion_line_gaee(log10_m_min - 3, log10_m_max - 3, C_ae_DFSZ_lower)
[ log10_m_eV, log10_gaee_DFSZ_upper ] = generate_QCD_axion_line_gaee(log10_m_min - 3, log10_m_max - 3, C_ae_DFSZ_upper)
[ log10_m_eV, log10_gann_KSVZ ] = generate_QCD_axion_line_gann(log10_m_min - 3, log10_m_max - 3, C_an_KSVZ)
[ log10_m_eV, log10_gann_DFSZ_lower ] = generate_QCD_axion_line_gann(log10_m_min - 3, log10_m_max - 3, C_an_DFSZ_lower)
[ log10_m_eV, log10_gann_DFSZ_upper ] = generate_QCD_axion_line_gann(log10_m_min - 3, log10_m_max - 3, C_an_DFSZ_upper)
[log10_m_eV, log10_gapp_KSVZ] = generate_QCD_axion_line_gapp(
    log10_m_min - 3, log10_m_max - 3, C_ap_KSVZ)
[ log10_m_eV, log10_gapp_DFSZ_lower ] = generate_QCD_axion_line_gapp(log10_m_min - 3, log10_m_max - 3, C_ap_DFSZ_lower)
[ log10_m_eV, log10_gapp_DFSZ_upper ] = generate_QCD_axion_line_gapp(log10_m_min - 3, log10_m_max - 3, C_ap_DFSZ_upper)

## start plotting -- steal some of Tanner's formatting
ncols = 3
nrows = 1
f, ax = plt.subplots(nrows=nrows, ncols=ncols,
                        figsize=(7*1.1*ncols, 7*nrows))

y_labels = [ r'$g_{aee}$', r'$g_{ann}$', r'$g_{app}$' ]
[axx.set_ylabel(y) for axx,y in zip(ax, y_labels)]
[axx.set_xlabel(r'$m_a~[\rm{meV}]$') for axx in ax]
[axx.set_xscale('log') for axx in ax]
[axx.set_yscale('log') for axx in ax]
[axx.minorticks_on() for axx in ax]
[axx.set_xlim([10, 1000]) for axx in ax]
ax[0].set_ylim([1.e-16, 1.e-11])
[axx.set_ylim([1.e-14, 1.e-7]) for axx in ax[1:]]

## plot our limits
matlist = ['FeBr2']
fermions = ['e', 'n', 'p']
for m in matlist:
        mat = material.Material(m, qs)
        with MPRester("9vCkS05eZPuFj169jSiZCNf9P5E6ckik") as mpr:
         magnetism_doc = mpr.magnetism.search(material_ids=["mp-22880"])
        S = magnetism_doc[0].magmoms
        for i,axx in enumerate(ax):
                 coupling = coup.Axion(qs, mlist, S, fermions[i])
                 reach = re.reach(mlist, qs, mat, coupling=coupling, pol_mixing=True)
                #  reach *= 1./mat.m_cell
                 print(np.min(reach))
                 axx.loglog(mlist*1000, reach, color=cs[3], label=f'$\mathrm{{{mat.name}}}$', lw=2)



## plot other constraints
ax[0].plot(rg['mass [eV]']*1000, rg['g'], c=cs[4], lw=2)
ax[0].fill_between(rg['mass [eV]']*1000, rg['g'], 1.e-10*np.ones(len(rg)), alpha=0.3, color=cs[4])
ax[0].text(10**2.75, 10**-12.8, r'$\mathrm{RG}$', fontsize = 25, color = cs[4])

ax[1].plot(ns_ann['mass [eV]']*1000, ns_ann['g'], c=cs[7], lw=2)
ax[1].fill_between(ns_ann['mass [eV]']*1000, ns_ann['g'], 1.e-6*np.ones(len(rg)), alpha=0.3, color=cs[7])
ax[1].text(10**1.05, 10**-8.8, r'$\mathrm{NS}$', fontsize = 25, color = cs[7])

ax[2].plot(ns_app['mass [eV]']*1000, ns_app['g'], c=cs[7], lw=2)
ax[2].fill_between(ns_app['mass [eV]']*1000, ns_app['g'], 1.e-6*np.ones(len(rg)), alpha=0.3, color=cs[7])
ax[2].text(10**2.75, 10**-8.6, r'$\mathrm{NS}$', fontsize = 25, color = cs[7])


ax[0].plot((10**np.array(log10_m_eV))*1000, 10**np.array(log10_gaee_KSVZ), c='gold', linewidth = 2)
ax[0].fill_between((10**np.array(log10_m_eV))*1000, 10**np.array(log10_gaee_DFSZ_upper), 10**np.array(log10_gaee_DFSZ_lower),
                   color = 'gold', alpha=0.3) 

ax[1].plot(
    (10**np.array(log10_m_eV))*1000,
    10**np.array(log10_gann_KSVZ),
    color = 'gold',
    linewidth = 2)


ax[1].fill_between((10**np.array(log10_m_eV))*1000, 10**np.array(log10_gann_DFSZ_upper), 10**np.array(log10_gann_DFSZ_lower),
                   color = 'gold', alpha=0.3)

ax[2].plot(
    (10**np.array(log10_m_eV))*1000,
    10**np.array(log10_gapp_KSVZ), color = 'gold', lw=2, alpha=0.3)

ax[2].fill_between((10**np.array(log10_m_eV))*1000, 10**np.array(log10_gapp_DFSZ_upper), 10**np.array(log10_gapp_DFSZ_lower),
                   color = 'gold', alpha=0.3)

ax[0].text(10**1.05, 10**-15.55, r'$\mathrm{KSVZ}$', 
             rotation = 20, fontsize = 30, color = (0.93, 0.8, 0.58))
ax[1].text(10**1.05, 10**-10.75, r'$\mathrm{KSVZ}$', 
             rotation = 20, fontsize = 30, color = (0.93, 0.8, 0.58))
ax[2].text(10**1.05, 10**-8.9, r'$\mathrm{KSVZ}$', 
             rotation = 20, fontsize = 30, color = (0.93, 0.8, 0.58))

ax[0].text(10**1.05, 10**-13.5, r'$\mathrm{DFSZ}$',
             rotation=20, fontsize=30, color=(0.93, 0.8, 0.58))
ax[1].text(10**1.05, 10**-9.8, r'$\mathrm{DFSZ}$',
             rotation=20, fontsize=30, color=(0.93, 0.8, 0.58))
ax[2].text(10**1.05, 10**-9.75, r'$\mathrm{DFSZ}$',
             rotation=20, fontsize=30, color=(0.93, 0.8, 0.58))


ax[0].legend(loc='lower right', fontsize=16)
## save fig.
f.tight_layout()
f.savefig('../results/plots/axion_fig.pdf')