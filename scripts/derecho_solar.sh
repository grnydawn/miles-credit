#!/bin/bash -l
#PBS -N gwm_scaler
#PBS -l select=1:ncpus=128:mpiprocs=128:ngpus=0:mem=200GB
#PBS -l walltime=06:00:00
#PBS -A NAML0001
#PBS -q main
#PBS -l job_priority=regular
#PBS -j oe
#PBS -k eod
module load conda craype/2.7.23 cray-mpich/8.1.27
conda activate hcredit
cd ..
mpiexec -n 1 --ppn 128 python -u applications/calc_global_solar.py -o /glade/derecho/scratch/dgagne/credit_scalers/
