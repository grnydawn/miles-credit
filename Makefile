# useful commands

venv:
	module load cray-python && \
	python3 -m venv venv

install:
	source ./venv/bin/activate && \
	module load rocm/6.4.1 && \
	pip install --upgrade pip && \
	pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm6.4 && \
	pip install -e . && \
	pip install torchmetrics metpy

train_fuxi:
	source ./venv/bin/activate && \
	module load rocm/6.4.1 && \
	pip install --upgrade pip && \
	python applications/train.py -c ./config/frontier_fuxi.yml

train_xformer:
	source ./venv/bin/activate && \
	module load rocm/6.4.1 && \
	python applications/train.py -c ./config/frontier_xformer.yml
