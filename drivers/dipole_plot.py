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

# FIXME: move to plotting script
def plot_coupling(coupling, ax=None, legend=True, mo=False):
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
        mat = material.Material(m, qs)
        reach = re.reach(mlist, qs, mat, coupling=coupling, pol_mixing=True)
        if coupling.name == 'dark_photon':
            reach *= 1.e-9 ## FIXME -- should kappa have units??
        if coupling.name == 'bminsl':
            reach *= 1.e-9*(mat.m_cell*1.e-9) ## FIXME: not sure about normalization.
        if (coupling.name == 'magneticdipole') or (coupling.name == 'electricdipole'):
            reach *= 1.
        print(np.min(reach))
        ax.loglog(mlist*1000, reach, color=cs[i], label=f'$\mathrm{{{mat.name}}}$', lw=2)
    if legend:
        ax.legend(fontsize=16, loc='lower right')
    if ax==None:
        fig.savefig(f'../results/plots/{coupling.name}.pdf')
    else:
        return ax

## start plotting -- steal some of Tanner's formatting
ncols = 2
nrows = 2
f, ax = plt.subplots(nrows=nrows, ncols=ncols,
                        figsize=(7*1.1*ncols, 7*nrows))


y_labels = np.repeat([ r'$d_M~[\mathrm{GeV}^{-1}]$', r'$d_E~[\mathrm{GeV}^{-1}]$'], 2)
[axx.set_ylabel(y) for axx,y in zip(ax.flatten(), y_labels)]
[axx.set_xlabel(r'$m_V~[\rm{meV}]$') for axx in ax.flatten()]
[axx.set_xscale('log') for axx in ax.flatten()]
[axx.set_yscale('log') for axx in ax.flatten()]
[axx.minorticks_on() for axx in ax.flatten()]
[axx.set_xlim([10, 1000]) for axx in ax.flatten()]


## save fig.
f.tight_layout()
f.savefig('../results/plots/dipole_fig.pdf')