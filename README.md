# PhonoDark-abs: Dark Matter -- Phonon absorption code
This code computes the absorption of various dark matter models onto phonon modes in semiconductor materials. It is based on the scattering code: [PhonoDark](https://github.com/tanner-trickle/PhonoDark).

## Installation
1. Download the code from this github page. 

2. In the home folder of the code, run: ``python setup.py install''. 

## Basic Usage
This code allows you to specify both targets and couplings. 
An example notebook is given in ``drivers/example_notebook.ipynb''. In addition, scripts to re-create the figures in [arXiv:XXXX.XXXXXX](LINKTBA) are also included in the ``drivers'' folder.

### Targets
The code ``pda/material.py'' contains the material class. This is a general class that will read phonon data files that you store in ``data/material_data/''. 

### Couplings
The code ``pda/couplings.py'' contains the various coupling classes we have implemented. There are various options for each class (e.g., you can set the coupling to electrons and protons separtely for many classes). To add your own coupling, follow the examples shown there.

## Attribution
If you use this code, please cite this paper:
[Effective Field Theory for Dark Matter Absorption on Single Phonons](LINKTBA),

with the citation:
'''
CITATION TBA
'''




