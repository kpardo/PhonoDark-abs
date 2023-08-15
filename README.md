# PhonoDark-abs: Dark Matter -- Phonon absorption code
This code computes the absorption of various dark matter models onto phonon modes in semiconductor materials. It is based on the scattering code: [PhonoDark](https://github.com/tanner-trickle/PhonoDark).

## Installation
1. Download the code from this github page. 

2. In the home folder of the code, run: ``python setup.py install``. 

## Basic Usage
This code allows you to specify both targets and couplings. 
An example notebook is given in [``drivers/example_notebook.ipynb``](drivers/example_notebook.ipynb). Python scripts to re-create the figures in [arXiv:2308.06314](https://arxiv.org/abs/2308.06314) are also included in the [``drivers``](drivers/) folder.

### Targets
The code [``pda/material.py``](pda/material.py) contains the material class. This is a general class that will read phonon data files that you store in [``data/material_data/``](data/material_data/). 

### Couplings
The code [``pda/couplings.py``](pda/couplings.py) contains the various coupling classes we have implemented. There are various options for each class (e.g., you can set the coupling to electrons and protons separtely for many classes). To add your own coupling, follow the examples shown there.

## Attribution
If you use this code, please cite this paper:

[Effective Field Theory for Dark Matter Absorption on Single Phonons](https://arxiv.org/abs/2308.06314)

with the citation:

```
  @article{Mitridate:2023izi,
      author = "Mitridate, Andrea and Pardo, Kris and Trickle, Tanner and Zurek, Kathryn M.",
      title = "{Effective Field Theory for Dark Matter Absorption on Single Phonons}",
      eprint = "2308.06314",
      archivePrefix = "arXiv",
      primaryClass = "hep-ph",
      reportNumber = "CALT-TH-2023-032, DESY-23-113, FERMILAB-PUB-23-417-T",
      month = "8",
      year = "2023"
  }
```




