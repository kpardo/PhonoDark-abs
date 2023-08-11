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
from pda.plotting import *

## set DM mass list
mlist = np.logspace(-2,np.log10(0.5),int(1e3));

## load other limits
## load kappa data. Fig. 11 from https://arxiv.org/pdf/2104.12786.pdf, + FF constraints
ga_kappa = pd.read_csv('../data/limit_data/kappa/gaas_kappa.dat', names=['mv', 'kappa'])
si_kappa = pd.read_csv('../data/limit_data/kappa/sio2_kappa.dat', names=['mv', 'kappa'])
al_kappa = pd.read_csv('../data/limit_data/kappa/al2o3_kappa.dat', names=['mv', 'kappa'])
ff = pd.read_csv('../data/limit_data/AxionLimits/FifthForce_BL.txt', skiprows=3, delimiter='\s+', names=['mass [eV]', 'g_BL'])


## set up plots
ncols = 2
nrows = 1
fig, axes = plt.subplots(nrows=nrows, ncols=ncols, 
                             figsize=(7*1.1*ncols, 7*nrows))

log10_m_min = 1
log10_m_max = 3

log10_d_min_list = [ -18, -19 ]
log10_d_max_list = [ -14, -13 ]

y_labels = [ r'$\kappa$', r'$g_{B - L}$' ]

for c in range(ncols):
    set_custom_tick_options(axes[c])
    set_custom_axes(axes[c], 'x', log10_m_min, log10_m_max,
               ax_type = 'log', 
               label = r'$m_V \, [\mathrm{meV}]$',
               show_first = True)
    set_custom_axes(axes[c], 'y', log10_d_min_list[c], log10_d_max_list[c],
               ax_type = 'log', 
               label = y_labels[c], 
               show_first = False)

## plot our constraints
matlist = ['GaAs', 'Al2O3', 'SiO2']
mats = [material.Material(m) for m in matlist]
couplist = [coup.DarkPhoton(omega=mlist, mat=m) for m in mats]
colorlist = ['firebrick', 'midnightblue', 'forestgreen']
plot_coupling(axes[0], couplist, colors=colorlist)

couplist = [coup.BminusL(omega=mlist, mat=m) for m in mats]
plot_coupling(axes[1], couplist, colors=colorlist)

## plot other limits & data
axes[0].plot(np.log10((ga_kappa['mv'].to_numpy()*u.eV).value*1000), np.log10(ga_kappa['kappa']), ls=':', c=colorlist[0])
axes[0].plot(np.log10((al_kappa['mv'].to_numpy()*u.eV).value*1000), np.log10(al_kappa['kappa']), ls=':', c=colorlist[1])
axes[0].plot(np.log10((si_kappa['mv'].to_numpy()*u.eV).value*1000), np.log10(si_kappa['kappa']), ls=':', c=colorlist[2])

plot_filled_region(axes[1], 
                   np.log10(ff['mass [eV]'].to_numpy()*1000),
                   np.log10(ff['g_BL'].to_numpy()),
                   log10_d_max_list[1],
                   color = 'teal') 

#### text

axes[1].text(1.4, -13.5, r'$\mathrm{Fifth} \; \mathrm{Force}$',
            rotation = 0, fontsize = 30, color = 'teal')

axes[0].text(1.3, -17, r'$\mathrm{GaAs}$',
            fontsize = 30, color = 'firebrick')

axes[0].text(1.8, -17.7, r'$\mathrm{Al}_2\mathrm{O}_3$',
            fontsize = 30, color = 'midnightblue')

axes[0].text(2.2, -17.4, r'$\mathrm{SiO}_2$',
            fontsize = 30, color = 'forestgreen')

axes[1].text(1.3, -17.3, r'$\mathrm{GaAs}$',
            fontsize = 30, color = 'firebrick')

axes[1].text(1.8, -18.5, r'$\mathrm{Al}_2\mathrm{O}_3$',
            fontsize = 30, color = 'midnightblue')

axes[1].text(2.2, -17.75, r'$\mathrm{SiO}_2$',
            fontsize = 30, color = 'forestgreen')


####

fig.tight_layout()
    

plt.savefig('../results/vector_fig.pdf',
                bbox_inches='tight', pad_inches = 0.075)
