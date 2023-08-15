import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator
import numpy as np
import seaborn as sns
import astropy.units as u
import pandas as pd


import pda.reach as re
import pda.material as material
import pda.constants as const
import pda.couplings as coup
from pda.plotting import *

from mp_api.client import MPRester

## set the mass list
mlist = np.linspace(0.01, 1, int(1e5))

## load other limits
rg = pd.read_csv('../data/limit_data/AxionLimits/RedGiants.txt', skiprows=2, delimiter='\s+', names=['mass [eV]', 'g'])
ns_ann = pd.read_csv('../data/limit_data/AxionLimits/NeutronStars_ann.txt', skiprows=3, delimiter='\s+', names=['mass [eV]', 'g'])
ns_app = pd.read_csv('../data/limit_data/AxionLimits/NeutronStars_app.txt', skiprows=3, delimiter='\s+', names=['mass [eV]', 'g'])

# QCD line, g_aee
C_ae_DFSZ_upper = 1.0/3.0
C_ae_DFSZ_lower = 0.024
C_ae_KSVZ = 2e-4

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

## setup plot
ncols = 3
nrows = 1
fig, axes = plt.subplots(nrows=nrows, ncols=ncols, 
                             figsize=(7*1.1*ncols, 7*nrows))


log10_m_min = 1
log10_m_max = 3

log10_d_min_list = [ -16, -12, -12 ]
log10_d_max_list = [ -9, -4, -4 ]

y_labels = [ r'$g_{aee}$', r'$g_{ann}$', r'$g_{app}$' ]

for c in range(ncols):
    set_custom_tick_options(axes[c])
    set_custom_axes(axes[c], 'x', log10_m_min, log10_m_max,
               ax_type = 'log', 
               label = r'$m_a \, [\mathrm{meV}]$',
               show_first = True)
    set_custom_axes(axes[c], 'y', log10_d_min_list[c], log10_d_max_list[c],
               ax_type = 'log', 
               label = y_labels[c], 
               show_first = False)
## plot our limits
matlist = ['FeBr2']
mats = [material.Material(m) for m in matlist]
lslist = ['solid']
Slist = np.array([
            [0.0, 0.0, 1.8],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0]]
        )
couplist = [coup.Axion(omega=mlist, mat=mats[0], S=Slist)]
plot_coupling(axes[0], couplist, colors=['darkorange'], ls=lslist)

matlist = ['GaAs', 'Al2O3']
mats = [material.Material(m) for m in matlist]
lslist = [':', ':']
Slist = []
Slist.append([0,0,0.5])
Slist.append([0,0,0.5])
couplist = [coup.Axion(omega=mlist, mat=m, S=s) for (m,s) in zip(mats, Slist)]
colorlist = ['firebrick', 'midnightblue']
plot_coupling(axes[0], couplist, colors=colorlist, ls=lslist)
couplist = [coup.Axion(omega=mlist, mat=m, S=s, fermion_coupling='n') for (m,s) in zip(mats, Slist)]
plot_coupling(axes[1], couplist, colors=colorlist, ls=lslist)
couplist = [coup.Axion(omega=mlist, mat=m, S=s, fermion_coupling='p') for (m,s) in zip(mats, Slist)]
plot_coupling(axes[2], couplist, colors=colorlist, ls=lslist)


## plot other constraints
[ log10_m_eV, log10_gaee_KSVZ ] = generate_QCD_axion_line_gaee(log10_m_min - 3, log10_m_max - 3, C_ae_KSVZ)

axes[0].plot(
    log10_m_eV + 3,
    log10_gaee_KSVZ,
    color = (0.93, 0.8, 0.58),
    linewidth = 2
)

[ log10_m_eV, log10_gaee_DFSZ_lower ] = generate_QCD_axion_line_gaee(log10_m_min - 3, log10_m_max - 3, C_ae_DFSZ_lower)
[ log10_m_eV, log10_gaee_DFSZ_upper ] = generate_QCD_axion_line_gaee(log10_m_min - 3, log10_m_max - 3, C_ae_DFSZ_upper)

plot_filled_region(axes[0], 
                   log10_m_eV + 3, log10_gaee_DFSZ_upper, log10_gaee_DFSZ_lower,
                   color = (0.93, 0.8, 0.58),
                   bound_line = False) 

[ log10_m_eV, log10_gann_KSVZ ] = generate_QCD_axion_line_gann(log10_m_min - 3, log10_m_max - 3, C_an_KSVZ)

axes[1].plot(
    log10_m_eV + 3,
    log10_gann_KSVZ,
    color = (0.93, 0.8, 0.58),
    linewidth = 2
)

[ log10_m_eV, log10_gann_DFSZ_lower ] = generate_QCD_axion_line_gann(log10_m_min - 3, log10_m_max - 3, C_an_DFSZ_lower)
[ log10_m_eV, log10_gann_DFSZ_upper ] = generate_QCD_axion_line_gann(log10_m_min - 3, log10_m_max - 3, C_an_DFSZ_upper)

plot_filled_region(axes[1], 
                   log10_m_eV + 3, log10_gann_DFSZ_upper, log10_gann_DFSZ_lower,
                   color = (0.93, 0.8, 0.58),
                   bound_line = False) 

[ log10_m_eV, log10_gapp_KSVZ ] = generate_QCD_axion_line_gapp(log10_m_min - 3, log10_m_max - 3, C_ap_KSVZ)

axes[2].plot(
    log10_m_eV + 3,
    log10_gapp_KSVZ,
    color = (0.93, 0.8, 0.58),
    linewidth = 2
)

[ log10_m_eV, log10_gapp_DFSZ_lower ] = generate_QCD_axion_line_gapp(log10_m_min - 3, log10_m_max - 3, C_ap_DFSZ_lower)
[ log10_m_eV, log10_gapp_DFSZ_upper ] = generate_QCD_axion_line_gapp(log10_m_min - 3, log10_m_max - 3, C_ap_DFSZ_upper)

plot_filled_region(axes[2], 
                   log10_m_eV + 3, log10_gapp_DFSZ_upper, log10_gapp_DFSZ_lower,
                   color = (0.93, 0.8, 0.58),
                   bound_line = False) 

plot_filled_region(axes[0],
                   np.log10(rg['mass [eV]']*1000), 
                   np.log10(rg['g']),
                   log10_d_max_list[0],
                   color='salmon')
plot_filled_region(axes[2],
                   np.log10(ns_ann['mass [eV]']*1000), 
                   np.log10(ns_ann['g']),
                   log10_d_max_list[2],
                   color='teal')

plot_filled_region(axes[1],
                   np.log10(ns_app['mass [eV]']*1000), 
                   np.log10(ns_app['g']),
                   log10_d_max_list[1],
                   color='teal')



#### text
axes[0].text(1.6, -11.6, r'$\mathrm{GaAs}^*$',
            fontsize = 30, color = 'firebrick')

axes[0].text(2.1, -10.25, r'$\mathrm{Al}_2\mathrm{O}_3^*$',
            fontsize = 30, color = 'midnightblue')

axes[0].text(1.025, -10.9, r'$\mathrm{Fe}\mathrm{Br}_2$',
            fontsize = 30, color = 'darkorange')

axes[1].text(1.6, -6.75, r'$\mathrm{GaAs}^*$',
            fontsize = 30, color = 'firebrick')

axes[1].text(2.05, -7, r'$\mathrm{Al}_2\mathrm{O}_3^*$',
            fontsize = 30, color = 'midnightblue')

axes[2].text(1.6, -6.75, r'$\mathrm{GaAs}^*$',
            fontsize = 30, color = 'firebrick')

axes[2].text(2.05, -6.75, r'$\mathrm{Al}_2\mathrm{O}_3^*$',
            fontsize = 30, color = 'midnightblue')

axes[0].text(2.9, -13.3, r'$\mathrm{RG}$', fontsize = 25, color = 'salmon', ha = 'right')

axes[1].text(1.1, -8.75, r'$\mathrm{NS}$', fontsize = 30, color = 'teal')
axes[2].text(2.7, -8.7, r'$\mathrm{NS}$', fontsize = 30, color = 'teal')

axes[0].text(1.05, -15.55, r'$\mathrm{KSVZ}$', 
             rotation = 15, fontsize = 30, color = (0.93, 0.8, 0.58))
axes[1].text(1.05, -10.9, r'$\mathrm{KSVZ}$', 
             rotation = 15, fontsize = 30, color = (0.93, 0.8, 0.58))
axes[2].text(1.05, -8.9, r'$\mathrm{KSVZ}$', 
             rotation = 15, fontsize = 30, color = (0.93, 0.8, 0.58))

axes[0].text(1.05, -14, r'$\mathrm{DFSZ}$', 
             rotation = 15, fontsize = 30, color = (0.93, 0.8, 0.58))
axes[1].text(1.05, -10, r'$\mathrm{DFSZ}$', 
             rotation = 15, fontsize = 30, color = (0.93, 0.8, 0.58))
axes[2].text(1.05, -9.9, r'$\mathrm{DFSZ}$', 
             rotation = 15, fontsize = 30, color = (0.93, 0.8, 0.58))

## save fig.
fig.tight_layout()
    
plt.savefig('../results/axion_fig.pdf', 
            bbox_inches='tight', pad_inches = 0.075)