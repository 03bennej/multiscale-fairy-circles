cuda_version = cupy-cuda12x  # specify cupy version here

## Build virtual environment and install packages
venv: venv/bin/activate # Create virtual environment and download required packages from requirements.txt
venv/bin/activate:
	python -m pip install -U setuptools pip
	python -m venv venv
	. venv/bin/activate; pip install --upgrade pip; pip install -r requirements.txt; pip install -e .;
	.venv/vin/activate; pip install $(cuda_version)