#!/bin/bash
#SBATCH --job-name=dft_catalyst
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=32
#SBATCH --time=04:00:00
#SBATCH --partition=compute
#SBATCH --output=dft_catalyst_%j.out
#SBATCH --error=dft_catalyst_%j.err

# DFT calculation for Cu-catalyst CO2 adsorption
# Uses VASP with PBE-D3 functional

module load vasp/6.3.0
module load intel-mpi

echo "Starting DFT calculation at $(date)"
echo "Running on nodes: $SLURM_JOB_NODELIST"

cd $SLURM_SUBMIT_DIR

# Run VASP
srun vasp_std > vasp.log 2>&1

echo "Calculation completed at $(date)"
