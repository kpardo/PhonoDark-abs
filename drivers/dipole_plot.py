import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator
import numpy as np
import seaborn as sns
import astropy.units as u
import pandas as pd

import pda.material as material
import pda.selfenergy as se
import pda.rate as r
import pda.reach as re
import pda.constants as const
import pda.couplings as coup
from pda.plotting import *

from mp_api.client import MPRester

# mlist = np.linspace(0.01,1,int(1e5));
mlist = np.logspace(-2, 0, int(1e3))

## setup plotting
ncols = 2
nrows = 1
fig, axes = plt.subplots(nrows=nrows, ncols=ncols,
                         figsize=(7*1.1*ncols, 7*nrows))

log10_m_min = 1
log10_m_max = 3

log10_d_min_list = [-11, -11]
log10_d_max_list = [[3, -2], [-2, -2]]

y_labels = [r'$d_M \, [\mathrm{GeV}^{-1}]$', r'$d_E \, [\mathrm{GeV}^{-1}]$']

# for r in range(nrows):
for c in range(ncols):
    set_custom_tick_options(axes[c])
    set_custom_axes(axes[c], 'x', log10_m_min, log10_m_max,
                    ax_type='log',
                    label=r'$m_V \, [\mathrm{meV}]$',
                    show_first=True)
    set_custom_axes(axes[c], 'y', log10_d_min_list[c], log10_d_max_list[0][c],
                    ax_type='log',
                    label=y_labels[c],
                    show_first=False,
                    step=2)

## plot our limits
matlist = ['FeBr2', 'GaAs', 'Al2O3']
mats = [material.Material(m) for m in matlist]
lslist = ['solid', ':', ':']
Slist = []
# with MPRester("9vCkS05eZPuFj169jSiZCNf9P5E6ckik") as mpr:
                # magnetism_doc = mpr.magnetism.search(material_ids=["mp-22880"])
                # Slist.append(magnetism_doc[0].magmoms)
Slist.append([0, 0, 1.8]) #FeBr2
Slist.append([0, 0, 0.5]) #GaAs
Slist.append([0, 0, 0.5]) #Al2O3
couplist = [coup.ElectricDipole(omega=mlist, mat=m, S=s, mo=True) for (m,s) in zip(mats, Slist)]
colorlist = ['darkorange', 'firebrick', 'midnightblue']
plot_coupling(axes[1], couplist, colors=colorlist, ls=lslist)
couplist = [coup.MagneticDipole(omega=mlist, mat=m, S=s, mo=True) for (m,s) in zip(mats, Slist)]
plot_coupling(axes[0], couplist, colors=colorlist, ls=lslist)

# ## No Magnetic Ordering
matlist = ['GaAs', 'Al2O3', 'SiO2']
mats = [material.Material(m) for m in matlist]
colorlist = ['firebrick', 'midnightblue', 'forestgreen']
couplist = [coup.MagneticDipole(omega=mlist, mat=m, mixing=True, mo=False) for m in mats]
plot_coupling(axes[0], couplist, colors=colorlist)

## Plot other constraints
therm_x = np.linspace(1, 3, 2)
therm_y = -6*np.linspace(1, 1, 2)

plot_filled_region(axes[0],
                   therm_x,
                   therm_y,
                   log10_d_max_list[0][0],
                   color='salmon',
                   linestyle='--'
                   )

plot_filled_region(axes[1],
                   therm_x,
                   therm_y,
                   log10_d_max_list[0][0],
                   color='salmon',
                   linestyle='--'
                   )

## text
axes[0].text(1.3, 0, r'$\mathrm{GaAs}$',
             fontsize=30, color='firebrick')

axes[0].text(1.7, -1.6, r'$\mathrm{Al}_2\mathrm{O}_3$',
             fontsize=30, color='midnightblue')

axes[0].text(2.2, -2, r'$\mathrm{SiO}_2$',
             fontsize=30, color='forestgreen')

axes[0].text(1.6, -8.75, r'$\mathrm{GaAs}^*$',
             fontsize=30, color='firebrick')

axes[0].text(2.05, -8.5, r'$\mathrm{Al}_2\mathrm{O}_3^*$',
             fontsize=30, color='midnightblue')

axes[0].text(1.03, -7.9, r'$\mathrm{FeBr}_2$',
             fontsize=30, color='darkorange')

axes[1].text(1.6, -5, r'$\mathrm{GaAs}^*$',
             fontsize=30, color='firebrick')

axes[1].text(2.05, -5, r'$\mathrm{Al}_2\mathrm{O}_3^*$',
             fontsize=30, color='midnightblue')

axes[1].text(1.05, -4.2, r'$\mathrm{FeBr}_2$',
             fontsize=30, color='darkorange')


axes[0].text(2.95, -6.2, r'$\mathrm{Therm.}$',
             fontsize=25, color='salmon', va='top', ha='right')

axes[1].text(2.95, -6.2, r'$\mathrm{Therm.}$',
             fontsize=25, color='salmon', va='top', ha='right')

## save fig
fig.tight_layout()
plt.savefig('../results/dipole_fig.pdf', 
                bbox_inches='tight', pad_inches = 0.075)