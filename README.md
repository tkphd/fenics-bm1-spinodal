# FEniCS & PFHub BM 1: Spinodal Decomposition

This repository presents an implementation of the PFHub [Spinodal Decomposition
Benchmark]( https://pages.nist.gov/pfhub/benchmarks/benchmark1.ipynb/) using
FEniCS. The basic code for this implementation is lifted from the Cahn-Hilliard
example provided by the [FEniCS Docs](
https://fenicsproject.org/olddocs/dolfin/latest/python/demos/cahn-hilliard/demo_cahn-hilliard.py.html).

I use Singularity to run the latest FEniCS build from [DockerHub](
https://hub.docker.com/r/fenics/stable) inside a Conda environment.
The [`Makefile`](Makefile) reflects this workflow.

* Install Miniconda and Mamba (better dependency resolver)
  ```bash
  $ curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
  $ chmod +x Miniconda3-latest-Linux-x86_64.sh
  $ ./Miniconda3-latest-Linux-x86_64.sh
  «agree to EULA, specify install dir, and prepare your shell»
  $ conda install mamba -n base -c conda-forge
  ```
* Create and activate an environment containing Singularity
  ```bash
  $ mamba create -n sing python=3 singularity
  $ conda activate sing
  ```
* Move to the benchmark directory and run the benchmark
  ```bash
  $ cd ~/path/to/fenics-bm1-spinodal
  $ make
  ```
