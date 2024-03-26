#!/bin/bash -l
#PBS -N gwm_scaler
#PBS -l select=19:ncpus=120:mpiprocs=2:ngpus=0:mem=200GB
#PBS -l walltime=06:00:00
#PBS -A NAML0001
#PBS -q main
#PBS -l job_priority=regular
#PBS -j oe
#PBS -k eod
module load conda
conda activate hcredit
cd ..
python -u applications/scaler.py -c config/crossformer.yml -p 60 -t 20 -o /glade/derecho/scratch/dgagne/credit_scalers/
