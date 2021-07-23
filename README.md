# FEniCS & PFHub BM 1: Spinodal Decomposition

This repository presents an implementation of the PFHub [Spinodal Decomposition
Benchmark][bm1spec] using FEniCS. The basic code for this implementation is
lifted from the Cahn-Hilliard example provided by the [FEniCS Docs][fenics].

Implemented by Trevor Keller (@tkphd, NIST) with substantial advice &
optimization from Nana Ofori-Opoku (@noforiopoku, CNL). 
Per [17 USC §105]( LICENSE.md), this work of a Federal employee is not subject
to copyright protection within the United States of America. Elsewhere,
consider it dedicated to the [public domain (CC0)][cc0].

I use Singularity to run the latest FEniCS build from [DockerHub][docker]
inside a Conda environment. The [`Makefile`](Makefile) reflects this workflow.

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

<!--References-->
bm1spec: https://pages.nist.gov/pfhub/benchmarks/benchmark1.ipynb/
cc0: https://creativecommons.org/publicdomain/zero/1.0/
docker: https://hub.docker.com/r/fenics/stable
fenics: https://fenicsproject.org/olddocs/dolfin/latest/python/demos/cahn-hilliard/demo_cahn-hilliard.py.html
