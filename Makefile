# useful commands

venv:
	module load cray-python && \
	python3 -m venv venv

install:
	source ./venv/bin/activate && \
	pip install -e . && \
	pip install torchmetrics metpy

train_fuxi:
	source ./venv/bin/activate && \
	python applications/train.py -c ./config/perlmutter_fuxi.yml

train_xformer:
	source ./venv/bin/activate && \
	python applications/train.py -c ./config/perlmutter_xformer.yml

nsys_fuxi:
	source ./venv/bin/activate && \
	nsys profile -o /pscratch/sd/y/youngsun/temp/nsys_fuxi \
		--cuda-memory-usage=true --force-overwrite=true \
		--trace=cuda,nvtx,osrt \
		python applications/train.py -c ./config/perlmutter_fuxi.yml

nsys_xformer:
	source ./venv/bin/activate && \
	nsys profile -o /pscratch/sd/y/youngsun/temp/nsys_crossformer \
		--cuda-memory-usage=true --force-overwrite=true \
		--trace=cuda,nvtx,osrt \
		python applications/train.py -c ./config/perlmutter_xformer.yml

		#--set=full -c 150 \

ncu_fuxi:
	source ./venv/bin/activate && \
	dcgmi profile --pause && \
	ncu --nvtx --force-overwrite \
		--target-processes all --export fuxi \
		--set=full -c 2000 \
		python applications/train.py -c ./config/perlmutter_fuxi.yml && \
	dcgmi profile --resume

ncu_xformer:
	source ./venv/bin/activate && \
	dcgmi profile --pause && \
	ncu --nvtx --force-overwrite \
		--target-processes all --export crossformer \
		--set=full -c 150 \
		python applications/train.py -c ./config/perlmutter_fuxi.yml && \
	dcgmi profile --resume
