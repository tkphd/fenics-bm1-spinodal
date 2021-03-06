# FEniCS & PFHub BM 1: Spinodal Decomposition

This repository presents an implementation of the PFHub [Spinodal Decomposition
Benchmark][bm1spec] using FEniCS. The basic code for this implementation is
lifted from the Cahn-Hilliard example provided by the [FEniCS Docs][fenics].

Implemented by Trevor Keller (@tkphd, NIST) with substantial advice &
optimization from Nana Ofori-Opoku (@noforiopoku, CNL). 
Per [17 USC Β§105]( LICENSE.md), this work of a Federal employee is not subject
to copyright protection within the United States of America. Elsewhere,
consider it dedicated to the [public domain (CC0)][cc0].

## Usage

I use Singularity to run the latest FEniCS build from [DockerHub][docker]
inside a Conda environment. The [`Makefile`](Makefile) reflects this workflow.

* Install Miniconda and Mamba (better dependency resolver)
  ```bash
  $ curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
  $ chmod +x Miniconda3-latest-Linux-x86_64.sh
  $ ./Miniconda3-latest-Linux-x86_64.sh
  Β«agree to EULA, specify install dir, and prepare your shellΒ»
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

## Formulation

Per the [spec][bm1spec], the Hemlholtz free energy functional for the system is

```
π = β«{πΉ + Β½π|βπ|Β²}dπ
πΉ = π(π - πΌ)Β²(π½ - π)Β²
```

Then `π = ππΉ/ππ‘`, and the split form of the equation of motion (eliminating the
biharmonic operator) is

```
ππ/ππ‘= ββπβπ
π = π - πβΒ²π
```

<!--References-->
[bm1spec]: https://pages.nist.gov/pfhub/benchmarks/benchmark1.ipynb/
[cc0]: https://creativecommons.org/publicdomain/zero/1.0/
[docker]: https://hub.docker.com/r/fenics/stable
[fenics]: https://fenicsproject.org/olddocs/dolfin/latest/python/demos/cahn-hilliard/demo_cahn-hilliard.py.html
