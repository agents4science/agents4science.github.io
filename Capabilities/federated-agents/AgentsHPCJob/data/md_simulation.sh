#!/bin/bash
#SBATCH --job-name=md_nanoparticle
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=64
#SBATCH --time=12:00:00
#SBATCH --partition=compute
#SBATCH --output=md_sim_%j.out
#SBATCH --error=md_sim_%j.err

# Molecular dynamics simulation of Cu nanoparticle in water
# Uses LAMMPS with ReaxFF potential

module load lammps/2023.08
module load intel-mpi

echo "Starting MD simulation at $(date)"
echo "Running on nodes: $SLURM_JOB_NODELIST"

cd $SLURM_SUBMIT_DIR

# Run LAMMPS
srun lmp -in input.lammps > lammps.log 2>&1

echo "Simulation completed at $(date)"
