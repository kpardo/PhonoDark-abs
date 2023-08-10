import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator
import numpy as np
import seaborn as sns
import astropy.units as u
import pandas as pd

import pda.material as material
import pda.reach as re
import pda.constants as const
import pda.couplings as coup

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

## set DM mass list
mlist = np.logspace(-2,np.log10(0.5),int(1e3));

## load other limits
## load kappa data. Fig. 11 from https://arxiv.org/pdf/2104.12786.pdf
ga_kappa = pd.read_csv('../data/limit_data/kappa/gaas_kappa.dat', names=['mv', 'kappa'])
si_kappa = pd.read_csv('../data/limit_data/kappa/sio2_kappa.dat', names=['mv', 'kappa'])
al_kappa = pd.read_csv('../data/limit_data/kappa/al2o3_kappa.dat', names=['mv', 'kappa'])

ff = pd.read_csv('../data/limit_data/AxionLimits/FifthForce_BL.txt', skiprows=3, delimiter='\s+', names=['mass [eV]', 'g_BL'])

# FIXME: move to plotting script
def plot_darkphot(ax=None, legend=True, mo=False, data=False):
    if ax==None:
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
    if mo == False:
        matlist = ['GaAs', 'Al2O3', 'SiO2']
    else:
        matlist = ['FeS2']
    for i, m in enumerate(matlist):
        if ax==None:
            print(m)
        mat = material.Material(m)
        coupling = coup.DarkPhoton(omega=mlist, mat=mat, mixing=True)
        reach = re.reach(mlist, mat, coupling=coupling)
        print(np.min(reach))
        ax.loglog(mlist*1000, reach*coupling.gx_conv, color=cs[i], label=f'$\mathrm{{{mat.name}}}$', lw=2)
    if legend:
        if data:
            ax.loglog(1,1, ls='dashed', color='black', label=r'$\mathrm{Data}$')
        ax.legend(fontsize=16, loc='lower right')
    if ax==None:
        fig.savefig(f'../results/{coupling.name}.pdf')
    else:
        return ax
# FIXME: move to plotting script
def plot_BL(ax=None, legend=True, mo=False, data=False):
    if ax==None:
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
    if mo == False:
        matlist = ['GaAs', 'Al2O3', 'SiO2']
    else:
        matlist = ['FeS2']
    for i, m in enumerate(matlist):
        if ax==None:
            print(m)
        mat = material.Material(m)
        coupling = coup.BminusL(omega=mlist, mat=mat, mixing=True)
        reach = re.reach(mlist, mat, coupling=coupling)
        print(np.min(reach))
        ax.loglog(mlist*1000, reach, color=cs[i], label=f'$\mathrm{{{mat.name}}}$', lw=2)
    if legend:
        if data:
            ax.loglog(1,1, ls='dashed', color='black', label=r'$\mathrm{Data}$')
        ax.legend(fontsize=16, loc='lower right')
    if ax==None:
        fig.savefig(f'../results/{coupling.name}.pdf')
    else:
        return ax

## start plotting -- steal some of Tanner's formatting
## plot each separately
## plot kinetic mixing first
ncols = 2
nrows = 1
f, ax = plt.subplots(nrows=nrows, ncols=ncols,
                        figsize=(7*1.1*ncols, 7*nrows))

ax[0].set_ylabel(r'$\kappa$')
ax[0].set_xlabel(r'$m_V~[\rm{meV}]$')
ax[0].set_xscale('log')
ax[0].set_yscale('log')
ax[0].minorticks_on()
ax[0].set_xlim([10, 1000])
ax[0].set_ylim([1.e-18, 1.e-14])

## plot our constraints
plot_darkphot(ax=ax[0], legend=True, data=True)
ax[0].plot((ga_kappa['mv'].to_numpy()*u.eV).value*1000, ga_kappa['kappa'], ls='dashed', c=cs[0])
ax[0].plot((al_kappa['mv'].to_numpy()*u.eV).value*1000, al_kappa['kappa'], ls='dashed', c=cs[1])
ax[0].plot((si_kappa['mv'].to_numpy()*u.eV).value*1000, si_kappa['kappa'], ls='dashed', c=cs[2])

## g_B-L
ax[1].set_ylabel(r'$g_{B-L}$')
ax[1].set_xlabel(r'$m_V~[\rm{meV}]$')
ax[1].set_xscale('log')
ax[1].set_yscale('log')
ax[1].minorticks_on()
ax[1].set_xlim([10, 1000])
ax[1].set_ylim([1.e-19, 1.e-13])

## plot our constraints
plot_BL(ax=ax[1], legend=True, data=True)

## plot other limits
ax[1].plot(ff['mass [eV]'].to_numpy()*1000, ff['g_BL'].to_numpy(), c=cs[5])
ax[1].fill_between(ff['mass [eV]'].to_numpy()*1000, 
ff['g_BL'].to_numpy(), 
1.e-10*np.ones(len(ff)), color=cs[5], alpha=0.3
)
ax[1].text(10**1.05, 3*10**-13, r'$\mathrm{Fifth} \; \mathrm{Force}$',
            rotation = 0, fontsize = 30, color = cs[5])

## save fig.
f.tight_layout()
f.savefig('../results/vector_fig.pdf')
