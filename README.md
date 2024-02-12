# NSF NCAR MILES Community Runnable Earth Digital Intelligence Twin (CREDIT)

## About
CREDIT is a package to train and run neural networks
that can emulate full NWP models by predicting
the next state of the atmosphere given the current state.

## Installation
Clone from miles-credit github page:
```bash
git clone git@github.com:NCAR/miles-credit.git
cd miles-credit
```

Install dependencies using environment.yml file:
```bash
mamba env create -f environment.yml
conda activate credit
```

To enable GPU support, install pytorch-cuda:
```bash
mamba install pytorch-cuda=12.1 -c pytorch -c nvidia
```

Install miles-credit with the following command:
```bash
pip install .
```
