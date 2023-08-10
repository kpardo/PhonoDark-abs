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
mlist = np.logspace(-2,np.log10(0.5),int(1e5))

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


f, ax = plt.subplots(2,3)
mats = ['GaAs','Al2O3','SiO2']
wvals1 = [10**(-2), 10**(-1), 10**(-3)]
wvals = np.vstack([wvals1, wvals1, wvals1])
xmins = [0.01, 0.04, 0.030]
xmaxs = [0.05, 0.14, 0.190]
for i,m in enumerate(mats):
    om, die = load_data(m)
    ax[0,i].loglog(om*1000, die, color='black', label=r'$\mathrm{Data}$')
    mat = material.Material(m)
    ax[0,i].loglog(mlist*1000, np.imag(d.Dielectric(mat=mat, mass=mlist, width_type='best',width_val=wvals[i,0]).eps),
                ls='solid', color=cs[i], label=r'$\mathrm{This~Work}$')
    ax[0,i].fill_between(mlist*1000, np.imag(d.Dielectric(mat=mat, mass=mlist, width_type='best', width_val=wvals[i,1]).eps),
                       np.imag(d.Dielectric(mat=mat, mass=mlist,  width_type='best', width_val=wvals[i, 2]).eps), color=cs[i],
                       alpha=0.3)
    ax[0,i].set_xlabel(r'$\omega~[\rm{meV}]$')
    ax[0,i].set_ylabel(r'$\mathrm{Im}(\epsilon(\omega))$')
    ax[0,i].text(0.9, 0.95, f'$\mathrm{{{m}}}$',
     horizontalalignment='center',
     verticalalignment='center',
     transform = ax[0,i].transAxes,
     fontsize=18)
    ax[0,i].set_xscale('linear')
    # ax[0,i].set_xlim([xmins[i]*1000, xmaxs[i]*1000])
    # ax[0,i].xaxis.set_minor_locator(MultipleLocator(1))

ax[0,0].legend(loc='upper left')

for i,m in enumerate(mats):
    om, die = load_imev_data(m)
    ax[1,i].loglog(om*1000., die, color='black', label=r'$\mathrm{Data}$')
    mat = material.Material(m)
    ax[1,i].loglog(mlist*1000, np.imag(d.Dielectric(mat=mat, mass=mlist, width_type='best',width_val=wvals[i,0]).imeps),
                ls='solid', color=cs[i], label=r'$\mathrm{This~Work}$')
    ax[1,i].fill_between(mlist*1000, np.imag(d.Dielectric(mat=mat, mass=mlist, width_type='best', width_val=wvals[i,1]).imeps),
                       np.imag(d.Dielectric(mat=mat, mass=mlist,  width_type='best', width_val=wvals[i, 2]).imeps), color=cs[i],
                       alpha=0.3)
    ax[1,i].set_xlabel(r'$\omega~[\rm{meV}]$')
    ax[1,i].set_ylabel(r'$\mathrm{Im}\left(-1/{\epsilon(\omega)}\right)$')
    ax[1,i].text(0.9, 0.95, f'$\mathrm{{{m}}}$',
     horizontalalignment='center',
     verticalalignment='center',
     transform = ax[1,i].transAxes,
     fontsize=18)
    ax[1,i].set_xscale('linear')
    ax[1,i].set_xlim([xmins[i]*1000, xmaxs[i]*1000])
    ax[1,i].xaxis.set_minor_locator(MultipleLocator(1))

f.tight_layout()
f.savefig('../results/dielectric_plot.pdf')
