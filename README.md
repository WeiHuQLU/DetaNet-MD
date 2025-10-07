# DetaNet-MD
## Introduction
DetaNet-MD is a machine learning molecular dynamics framework that integrates the DetaNet model with velocity-Verlet and RPMD to simulate dynamic infrared and Raman spectra. It provides an efficient and transferable solution for real-time prediction of vibrational spectra across diverse molecular and material systems.
## System requirements
The development version of the package has been tested on CentOS Linux 7 (Core).
## Installation guide

We recommend using [Miniforge](https://github.com/conda-forge/miniforge/) instead of Anaconda to download and manage the installation packages.

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

### Example of environment setup

Create a new environment and install the required dependencies:

```bash
# Create and activate a conda environment
conda create -n detanet-md 
conda activate detanet-md

# Install PyTorch 
conda install pytorch=2.4.1 pytorch-lightning=2.5.0 

# Install related packages
conda install pytorch_geometric=2.1.0 pytorch_scatter=2.1.2 pytorch_sparse=0.6.18 torchmd-net=2.4.0

# Install remaining dependencies
pip install e3nn==0.5.5 ase==3.24.0 schnetpack==2.1.1 ipi==3.1 
```
Typical installation time on a "normal" desktop computer is approximately 1.5 hours.

## Demo
### Prepare data
#### QMe14S Dataset 
We trained DetaNet on [QMe14S](https://figshare.com/s/889262a4e999b5c9a5b3) to obtain a universal force field. The QMe14S dataset includes energy, force, dipole moment, and polarizability for 186,102 small isolated organic molecules, covering both equilibrium and nonequilibrium configurations sampled using atom-centered density matrix propagation (ADMP) with the Gaussian 16 package. We randomly split the QMe14S dataset into training, validation, and test sets with percentages of 90%, 5%, and 5%, respectively.
#### Other Datasets 
We evaluated the transferability of the DetaNet model to diverse and complex systems, including organic and inorganic crystals, molecular aggregates, and polypeptides.  
The datasets used for transfer learning, based on the QMe14S pre-trained model, are available at:  
[https://figshare.com/s/043b1ace6546b4221a43](https://figshare.com/s/043b1ace6546b4221a43)

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
```
The trained models are stored in:

- `DetaNet-MD/training/training_parameters/QMe14S/`
  
#### Transfer Learning 
Before training, the pre-trained model parameters in  `DetaNet-MD/training/training_parameters/QMe14S/` should be loaded. For organic and inorganic crystals, molecular aggregates, and polypeptides, we selected 2,000 configurations for training each system.
The models obtained from transfer learning are stored in: 

- `DetaNet-MD/training/training_parameters/`
  
### MD and RPMD Simulations

This package enables molecular dynamics (MD) and ring polymer molecular dynamics (RPMD) simulations through interfaces between DetaNet and external frameworks such as SchnetPack, ASE, i-PI, and torchmd-net.

#### MD Simulations
The MD example script is provided in:  
`DetaNet-MD/md/example/run_md.ipynb`  

You can run it with Jupyter Notebook:

```bash
jupyter notebook DetaNet-MD/md/example/run_md.ipynb
```
#### RPMD Simulations
Two approaches are available for running RPMD:

##### 1. Via SchNetPack interface

Example script:  
`DetaNet-MD/rpmd/example_2/run_rpmd.ipynb`  

Run with Jupyter Notebook:
```bash
jupyter notebook DetaNet-MD/rpmd/example_2/run_rpmd.ipynb
```
##### 2. Via ASE and i-PI interface
These examples demonstrate how to connect i-PI to client codes using ASE as a middleware.  

- i-PI (server) ⟷ ASE (client) ⟷ DetNet-Force code  

Example script:  
`DetaNet-MD/rpmd/example_1/run.sh`  

Run on a cluster with:
```bash
sbatch run.sh
```
### Calculation of IR and Raman spectra
Example script:  
`DetaNet-MD/md/example/calculated_spectra.ipynb`  

Run with Jupyter Notebook:
```bash
jupyter notebook DetaNet-MD/md/example/calculated_spectra.ipynb
```
Example data for spectra calculation is available at: https://figshare.com/s/043b1ace6546b4221a43
