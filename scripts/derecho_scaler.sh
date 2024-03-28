#!/bin/bash -l
#PBS -N gwm_scaler
#PBS -l select=19:ncpus=128:mpiprocs=2:ngpus=0:mem=200GB
#PBS -l walltime=12:00:00
#PBS -A NAML0001
#PBS -q main
#PBS -l job_priority=regular
#PBS -j oe
#PBS -k eod
module load conda craype/2.7.23 cray-mpich/8.1.27
conda activate hcredit
cd ..
mpiexec -n 37 --ppn 2 python -u applications/scaler.py -c config/crossformer.yml -p 64 -t 256 -o /glade/derecho/scratch/dgagne/credit_scalers/
