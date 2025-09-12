# DetaNet-MD
## Introduction
DetaNet-MD is a machine learning molecular dynamics framework that integrates the DetaNet model with velocity-Verlet and RPMD to simulate dynamic infrared and Raman spectra. It provides an efficient and transferable solution for real-time prediction of vibrational spectra across diverse molecular and material systems.
## Requirements
- python=3.12.9  
- pytorch=2.4.1  
- pytorch-lightning=2.5.0  
- pytorch_geometric=2.1.0  
- pytorch_scatter=2.1.2  
- pytorch_sparse=0.6.18  
- e3nn=0.5.5  
- torchmd-net=2.4.0  
- ase=3.24.0  
- schnetpack=2.1.1  
- ipi=3.1  
## How to Use DetaNet-MD Package
### Prepare data
We trained DetaNet on [QMe14S](https://figshare.com/s/889262a4e999b5c9a5b3) to obtain a universal force field. The QMe14S dataset includes energy, force, dipole moment, and polarizability for 186,102 small isolated organic molecules, covering both equilibrium and nonequilibrium configurations sampled using atom-centered density matrix propagation (ADMP) with the Gaussian 16 package. We randomly split the QMe14S dataset into training, validation, and test sets with percentages of 90%, 5%, and 5%, respectively.
### Training
The training scripts for different properties are provided in the `DetaNet-MD/training/training_models` directory:

- `train_energy.ipynb` – training on energies  
- `train_force.ipynb` – training on forces  
- `train_dipole.ipynb` – training on dipole moments  
- `train_polar.ipynb` – training on polarizabilities  

#### How to Run
You can open and run the training scripts using Jupyter Notebook:

```bash
jupyter notebook DetaNet-MD/training/training_models/train_energy.ipynb
