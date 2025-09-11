#!/bin/bash
#SBATCH -J i-PI
#SBATCH -N 1
#SBATCH --ntasks-per-node=56
#SBATCH -p hfacnormal01




module purge
source /public/home/miniforge3/etc/profile.d/conda.sh
source /public/home/miniforge3/etc/profile.d/mamba.sh
mamba activate MD                   

IPI=i-pi
PYTHON=python

${IPI} input.xml &> log.i-pi & 

sleep 10

${PYTHON} run-ase.py & 

wait

