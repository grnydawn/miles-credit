# useful commands

MASTER_ADDR := $(shell hostname)
MASTER_PORT := 29500

SCRDIR := /lustre/cyclone/nwp501/scratch/grnydawn/miles
#SCRDIR := /lustre/gale/nwp501/scratch/grnydawn/miles

MILESDIR := $(shell pwd)
PRERUN := export MPLCONFIGDIR=${SCRDIR}/tmpdir MASTER_ADDR=${MASTER_ADDR} MASTER_PORT=${MASTER_PORT} && \
			source /autofs/nccs-svm1_proj/cli190/grnydawn/hpc11/miles/venv/bin/activate

.PHONY: doc clean_doc

train_era5_miller:
	${PRERUN} && \
	python applications/train.py -c ./config/miller_era5_xformer.yml

train_mpas_miller:
	${PRERUN} && \
	python applications/train.py -c ./config/miller_mpas_xformer.yml

train_mpas_arch:
	${PRERUN} && \
	python applications/train.py -c ./config/arch_mpas_xformer.yml
