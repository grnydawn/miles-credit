# useful commands

SCRDIR := /lustre/cyclone/nwp501/scratch/grnydawn/miles
MILESDIR := $(shell pwd)
PRERUN := export MPLCONFIGDIR=${SCRDIR}/tmpdir && \
			source /autofs/nccs-svm1_proj/cli190/grnydawn/hpc11/miles/venv/bin/activate

.PHONY: doc clean_doc

train_mpas:
	${PRERUN} && \
	python applications/train.py -c ./config/hpc11_mpas_xformer.yml
