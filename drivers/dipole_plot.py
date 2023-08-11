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
from pda.plotting import *

from mp_api.client import MPRester

mlist = np.linspace(0.01,1,int(1e5));

## setup plotting


## plot our limits
matlist = ['FeBr2', 'GaAs', 'Al2O3']
mats = [material.Material(m) for m in matlist]
lslist = ['solid', ':', ':']
Slist = []
# with MPRester("9vCkS05eZPuFj169jSiZCNf9P5E6ckik") as mpr:
                # magnetism_doc = mpr.magnetism.search(material_ids=["mp-22880"])
                # Slist.append(magnetism_doc[0].magmoms)
Slist.append([3.6, 0, 0]) #FeBr2
Slist.append([0,0,0.5])
Slist.append([0,0,0.5])
couplist = [coup.ElectricDipole(omega=mlist, mat=m, S=s) for m in mats]
colorlist = ['darkorange', 'firebrick', 'midnightblue']
plot_coupling(axes[0], couplist, colors=colorlist, ls=lslist)
couplist = [coup.MagneticDipole(omega=mlist, mat=m) for m in mats]
plot_coupling(axes[1], couplist[1:], colors=colorlist[1:], ls=lslist[1:])
couplist = [coup.MagneticDipole(omega=mlist, mat=m, S=s) for m in mats]
plot_coupling(axes[1], couplist[1:], colors=colorlist[1:], ls=lslist[1:])

