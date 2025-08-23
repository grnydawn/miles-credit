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
