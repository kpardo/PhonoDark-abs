import matplotlib.pyplot as plt
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

### load other constraints
ff = pd.read_csv('../data/limit_data/AxionLimits/FifthForce.txt', skiprows=3, delimiter='\s+', names=['mass [eV]', 'd_e'])
rg = pd.read_csv('../data/limit_data/AxionLimits/RedGiants_electron.txt', skiprows=3, delimiter='\s+', names=['mass [eV]', 'd_e'])
wd = pd.read_csv('../data/limit_data/AxionLimits/WhiteDwarfs_electron.txt', skiprows=3, delimiter='\s+', names=['mass [eV]', 'd_e'])

# FIXME: Also move to plotting script?
def plot_coupling(coupling, ax=None, legend=True, mo=False):
    if ax==None:
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
    if mo == False:
        matlist = ['GaAs', 'Al2O3', 'SiO2', 'FeBr2']
    else:
        matlist = ['FeS2']
    for i, m in enumerate(matlist):
        if ax==None:
            print(m)
        mat = material.Material(m, qs)
        reach = re.reach(mlist, qs, mat, coupling=coupling, pol_mixing=True)
        if coupling.name == 'dark_photon':
            reach *= const.M_ELEC/mat.m_cell
        ax.loglog(mlist*1000, reach, color=cs[i], label=f'$\mathrm{{{mat.name}}}$', lw=2)
    if legend:
        ax.legend(fontsize=16, loc='lower right')
    if ax==None:
        fig.savefig(f'../results/plots/{coupling.name}.pdf')
    else:
        return ax

f = plt.figure()
ax = f.add_subplot(1,1,1)
co = coup.ScalarE(q_XYZ_list=qs)
ax.plot(ff['mass [eV]']*1000, ff['d_e'], c=cs[5], lw=2)
ax.plot(rg['mass [eV]']*1000, rg['d_e'], c=cs[4], lw=2)
ax.plot(wd['mass [eV]']*1000, wd['d_e'], c=cs[7], lw=2)

ax.fill_between(ff['mass [eV]']*1000, ff['d_e'], 1.e30*np.ones(len(ff)), alpha=0.3, color=cs[5])
ax.fill_between(rg['mass [eV]']*1000, rg['d_e'], 1.e30*np.ones(len(rg)), alpha=0.3, color=cs[4])
ax.fill_between(wd['mass [eV]']*1000, wd['d_e'], 1.e30*np.ones(len(wd)), alpha=0.3, color=cs[7])

plot_coupling(co, ax=ax, legend=True)
plt.yscale('log')
plt.xscale('log')
plt.ylim([1.e4, 1.e9])
plt.xlim([10, 1000])
plt.ylabel(r'$d_e$')
plt.xlabel(r'$m_\phi \, [\mathrm{meV}]$')
plt.text(10**1.025, 10**4.8, r'$\mathrm{Fifth}\;\mathrm{Force}$', 
          rotation = 55, fontsize = 25, color = cs[5])
plt.text(10**1.05, 10**6.75, r'$\mathrm{RG}$', fontsize = 25, color = cs[4])
plt.text(10**1.05, 10**6.1, r'$\mathrm{WD}$', fontsize = 25, color = cs[7])
plt.tight_layout()
plt.savefig('../results/plots/scalar_de.pdf')
