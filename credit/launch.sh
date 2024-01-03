#!/bin/bash
    #PBS -A NAML0001
    #PBS -N out
    #PBS -l walltime=00:10:00
    #PBS -l select=16:ncpus=64:ngpus=4:mem=480GB
    #PBS -q preempt
    #PBS -j oe
    #PBS -k eod

    # Load modules
    module purge
    module load nvhpc cuda cray-mpich conda 
    conda activate holodec

    # Get a list of allocated nodes
    nodes=( $( cat $PBS_NODEFILE ) )
    head_node=${nodes[0]}
    head_node_ip=$(ssh $head_node hostname -i | awk '{print $1}')

    # Export environment variables
    export LSCRATCH=/glade/derecho/scratch/schreck/
    export LOGLEVEL=INFO
    export NCCL_DEBUG=INFO

    # Print the results
    echo "Number of nodes: 16"
    echo "Number of GPUs per node: 4"
    echo "Total number of GPUs: 64"

    # Log in to WandB if needed
    # wandb login 02d2b1af00b5df901cb2bee071872de774781520
    pip install .

    # Launch MPIs
    CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63" mpiexec -n 16 --ppn 1 --cpu-bind none torchrun --nnodes=16 --nproc-per-node=4 --rdzv-backend=c10d --rdzv-endpoint=$head_node_ip /glade/work/schreck/repos/global/miles-credit/credit/pbs.py -c ../config/vit2d.yml
        