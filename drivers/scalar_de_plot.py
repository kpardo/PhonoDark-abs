import matplotlib.pyplot as plt
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

u.set_enabled_equivalencies(u.mass_energy())

mlist = np.linspace(0.01, 1, int(1e5))

### load other constraints
ff = pd.read_csv('../data/limit_data/AxionLimits/FifthForce.txt', skiprows=3, delimiter='\s+', names=['mass [eV]', 'd_e'])
rg = pd.read_csv('../data/limit_data/AxionLimits/RedGiants_electron.txt', skiprows=3, delimiter='\s+', names=['mass [eV]', 'd_e'])
wd = pd.read_csv('../data/limit_data/AxionLimits/WhiteDwarfs_electron.txt', skiprows=3, delimiter='\s+', names=['mass [eV]', 'd_e'])

## define in order to get conversion later
mat = material.Material('GaAs')
coupling = coup.ScalarE(omega=mlist, mat=mat)

## plot setup
ncols = 1
nrows = 1
fig, axes = plt.subplots(nrows=nrows, ncols=ncols, 
                             figsize=(7*1.2*ncols, 7*nrows))

log10_m_min = 1
log10_m_max = 3

log10_d_min = 4
log10_d_max = 9

y_label = [ r'$d_e$' ]

ax2 = axes.twinx()
set_custom_tick_options(ax2, major_left = False, minor_left = False)

set_custom_tick_options(axes, major_right = False, minor_right = False)
set_custom_axes(axes, 'x', log10_m_min, log10_m_max,
           ax_type = 'log', 
           label = r'$m_\phi \, [\mathrm{meV}]$',
           show_first = True)
set_custom_axes(axes, 'y', log10_d_min, log10_d_max,
           ax_type = 'log', 
           label = r'$d_e$', 
           show_first = False)


log_g_min = np.log10(10**log10_d_min / coupling.gx_conv)
log_g_max = np.log10(10**log10_d_max / coupling.gx_conv)

set_log_yticks2(ax2, 
                int(np.floor(log_g_min)), int(np.ceil(log_g_max)),
                log_g_min, log_g_max
               )
ax2.set_ylabel(r'$g_e$', fontsize = 30)

## plot our constraints
matlist = ['GaAs', 'Al2O3', 'SiO2']
mats = [material.Material(m) for m in matlist]
couplist = [coup.ScalarE(omega=mlist, mat=m) for m in mats]
colorlist = ['firebrick', 'midnightblue', 'forestgreen']
plot_coupling(axes, couplist, colors=colorlist)

## plot other constraints
plot_filled_region(axes,
                   np.log10(ff['mass [eV]']*1000), 
                   np.log10(ff['d_e']),
                   log10_d_max,
                   color='teal')
plot_filled_region(axes,
                   np.log10(rg['mass [eV]']*1000), 
                   np.log10(rg['d_e']),
                   log10_d_max,
                   color='salmon')
plot_filled_region(axes,
                   np.log10(wd['mass [eV]']*1000), 
                   np.log10(wd['d_e']),
                   log10_d_max,
                   color='orange')

## text
axes.text(1.025, 4.9, r'$\mathrm{Fifth}\;\mathrm{Force}$', 
          rotation = 55, fontsize = 30, color = 'teal')
axes.text(1.05, 6.75, r'$\mathrm{RG}$', fontsize = 30, color = 'salmon')
axes.text(1.05, 6.1, r'$\mathrm{WD}$', fontsize = 30, color = 'orange')

axes.text(1.3, 7.75, r'$\mathrm{GaAs}$',
            fontsize = 30, color = 'firebrick')

axes.text(1.7, 6.8, r'$\mathrm{Al}_2\mathrm{O}_3$',
            fontsize = 30, color = 'midnightblue')

axes.text(2.2, 7.5, r'$\mathrm{SiO}_2$',
            fontsize = 30, color = 'forestgreen')

fig.tight_layout()
    
plt.savefig('../results/scalar_de.pdf', 
            bbox_inches='tight', pad_inches = 0.075)
