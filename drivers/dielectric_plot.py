from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
from matplotlib.ticker import FormatStrFormatter
import matplotlib
import sys
from pda.constants import *
import pda.couplings as coup
import pda.reach as re
import pda.rate as r
import pda.selfenergy as se
import pda.material as material
import pda.dielectric as d
from pda.plotting import *

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import astropy.units as u
import astropy.constants as const
import pandas as pd

DATA_DIR = '../data'
mlist = np.linspace(0.01,0.5,int(1e5))

colorlist = ['firebrick', 'midnightblue', 'forestgreen']

## load data files
def load_data(name):
    try:
        dat = pd.read_csv(f'{DATA_DIR}/measured_dielectric_data/{name}.csv', skiprows=1,
        names=('omega', 're', 'im'), delimiter='\s+')
        return dat['omega'].to_numpy(), dat['im'].to_numpy()
    except:
        dat_o = pd.read_csv(f'{DATA_DIR}/measured_dielectric_data/{name}_o.csv', skiprows=1,
        names=('omega', 're', 'im'), delimiter='\s+')
        dat_e = pd.read_csv(f'{DATA_DIR}/measured_dielectric_data/{name}_e.csv', skiprows=1,
        names=('omega', 're', 'im'), delimiter='\s+')
        imdat = 1./3.*dat_o['im'].to_numpy() + 2./3.*dat_e['im'].to_numpy()
        return dat_o['omega'].to_numpy(), imdat

## load inv data files
def load_imev_data(name):
    try:
        dat = pd.read_csv(f'{DATA_DIR}/measured_dielectric_data/{name}.csv', skiprows=1,
        names=('omega', 're', 'im'), delimiter='\s+')
        imev = np.imag(-1./(dat['re'].to_numpy()+1j*dat['im'].to_numpy()))
        return dat['omega'].to_numpy(), imev
    except:
        dat_o = pd.read_csv(f'{DATA_DIR}/measured_dielectric_data/{name}_o.csv', skiprows=1,
        names=('omega', 're', 'im'), delimiter='\s+')
        dat_e = pd.read_csv(f'{DATA_DIR}/measured_dielectric_data/{name}_e.csv', skiprows=1,
        names=('omega', 're', 'im'), delimiter='\s+')
        imev_o = np.imag(-1./(dat_o['im'].to_numpy()*1j + dat_o['re'].to_numpy()))
        imev_e = np.imag(-1./(dat_e['im'].to_numpy()*1j + dat_e['re'].to_numpy()))
        imev = 1./3.*imev_o + 2./3.*imev_e
        return dat_o['omega'].to_numpy(), imev


ncols = 3
nrows = 2
fig, axes = plt.subplots(nrows=nrows, ncols=ncols, 
                             figsize=(7*1.1*ncols, 7*nrows))

w_min_list = [ 10, 40, 30 ]
w_max_list = [ 50, 140, 190 ]
step_list = [ 10, 20, 20 ]

y_min_list = [ [ -4, -2, -3 ], [ -6, -4, -4 ] ]
y_max_list = [ [ 3, 3, 3 ],  [ 1, 2, 2 ] ]

y_label = [ r'$\mathrm{Im}\left[ \varepsilon(\omega) \right]$', r'$\mathrm{Im} \left[ -1/\varepsilon(\omega) \right]$' ]

for r in range(nrows):
    for c in range(ncols):
        
        set_custom_tick_options(axes[r, c])
        set_custom_axes(axes[r, c], 'x', w_min_list[c], w_max_list[c],
                   ax_type = 'lin', 
                   label = r'$\omega \, [\mathrm{meV}]$',
                   show_first = True,
                   step = step_list[c])
        set_custom_axes(axes[r, c], 'y', y_min_list[r][c], y_max_list[r][c],
                   ax_type = 'log', 
                   label = y_label[r], 
                   show_first = False)

mats = ['GaAs','Al2O3','SiO2']
wvals1 = [10**(-2), 10**(-1), 10**(-3)]
wvals = np.vstack([wvals1, wvals1, wvals1])

for i,m in enumerate(mats):
    om, die = load_data(m)
    axes[0,i].plot(om*1000, np.log10(die), color=colorlist[i], ls=':')
    mat = material.Material(m)
    axes[0,i].plot(mlist*1000, np.log10(np.imag(d.Dielectric(mat=mat, mass=mlist, width_type='best',width_val=wvals[i,0]).eps)),
                ls='solid', color=colorlist[i])
    axes[0,i].fill_between(mlist*1000, np.log10(np.imag(d.Dielectric(mat=mat, mass=mlist, width_type='best', width_val=wvals[i,1]).eps)),
                       np.log10(np.imag(d.Dielectric(mat=mat, mass=mlist,  width_type='best', width_val=wvals[i, 2]).eps)), color=colorlist[i],
                       alpha=0.15)
    

for i,m in enumerate(mats):
    om, die = load_imev_data(m)
    axes[1,i].plot(om*1000., np.log10(die), color=colorlist[i], ls=':')
    mat = material.Material(m)
    axes[1,i].plot(mlist*1000, np.log10(np.imag(-1.*d.Dielectric(mat=mat, mass=mlist, width_type='best',width_val=wvals[i,0]).imeps)),
                ls='solid', color=colorlist[i], label=r'$\mathrm{This~Work}$')
    axes[1,i].fill_between(mlist*1000, np.log10(np.imag(-1.*d.Dielectric(mat=mat, mass=mlist, width_type='best', width_val=wvals[i,1]).imeps)),
                       np.log10(np.imag(-1.*d.Dielectric(mat=mat, mass=mlist,  width_type='best', width_val=wvals[i, 2]).imeps)), color=colorlist[i],
                       alpha=0.15)


axes[0, 0].text(12, 2.25, r'$\mathrm{GaAs}$',
                color='firebrick', fontsize=30)
axes[1, 0].text(12, 0.25, r'$\mathrm{GaAs}$',
                color='firebrick', fontsize=30)

axes[0, 1].text(120, 2.25, r'$\mathrm{Al}_2\mathrm{O}_3$',
                color='midnightblue', fontsize=30)
axes[1, 1].text(45, 1.25, r'$\mathrm{Al}_2\mathrm{O}_3$',
                color='midnightblue', fontsize=30)

axes[0, 2].text(35, 2.25, r'$\mathrm{SiO}_2$',
                color='forestgreen', fontsize=30)
axes[1, 2].text(35, 1.25, r'$\mathrm{SiO}_2$',
                color='forestgreen', fontsize=30)

## save fig
fig.tight_layout()
    
plt.savefig('../results/dielectric_plot.pdf', 
                bbox_inches='tight', pad_inches = 0.075)
