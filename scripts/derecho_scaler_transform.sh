#!/bin/bash -l
#PBS -N gwm_scaler
#PBS -l select=4:ncpus=128:mpiprocs=128:ngpus=0:mem=200GB
#PBS -l walltime=03:00:00
#PBS -A NAML0001
#PBS -q main
#PBS -l job_priority=regular
#PBS -j oe
#PBS -k eod
module load conda craype/2.7.23 cray-mpich/8.1.27
conda activate hcredit
cd ..
mpiexec -n 512 python -u -m mpi4py applications/scaler.py \
  -c config/crossformer.yml \
  -t 1h \
  -o /glade/derecho/scratch/dgagne/credit_scalers/ \
  -d /glade/derecho/scratch/dgagne/era5_quantile/ \
  -s /glade/derecho/scratch/dgagne/credit_scalers/era5_quantile_scalers_2024-03-30_00:28.parquet \
  -r
