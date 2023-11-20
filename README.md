<div align="center">
  <h1>Supporting code for the paper "Phenotypic plasticity: A missing element in the theory of vegetation pattern formation"</h1>
  <img src="https://github.com/03bennej/multiscale-fairy-circles/blob/main/images/stability.png" width="350"> 
  
  <img src="https://github.com/03bennej/multiscale-fairy-circles/blob/main/images/fc_120.png" width="275"> <img src="https://github.com/03bennej/multiscale-fairy-circles/blob/main/images/fc_129.png" width="275">
</div>

Part of the code in this repo is intended to be accelerated on GPUs. We used [CuPy](https://cupy.dev/) which requires an NVIDIA CUDA GPU and CUDA Toolkit. For the specific installation requirements for your system, see the up-to-date installation instructions: [https://docs.cupy.dev/en/stable/install.html](https://docs.cupy.dev/en/stable/install.html), in particular, how to install the correct version of CUDA Toolkit for your system. 

Once all CuPy dependencies have been installed, modify the `cuda_version` in the Makefile and run `make venv` to build the virtual environment. 

## Notebooks

We provided the basic code used to produce our results in notebook form:

- `notebooks/simulation.ipynb`: numerical simulations of the integro-pde model described in the paper.
- `notebooks/linear_stability_analysis.ipynb`: linear stability analysis of the homoegeneous steady states associated with the integro-pde model.

