# GNU Makefile for PFHub BM1
# Recommended for use with a Conda environment for Singularity with Python 3

# Cluster Settings

MPI = mpirun
PY3 = python3
RANKS = 4

# Container Settings

IMAGE = fenics/stable
NAME = pfhub

# Make Targets

all: fenics-bm-1b.xdmf
.PHONY: all clean instance list shell spinodal stop watch

fenics-bm-1b.xdmf: spinodal.py
	make instance
	make spinodal
	make stop

clean:
	rm -vf *spinodal.h5 *spinodal.xdmf *spinodal.log fenics*.csv

instance:
	singularity instance start -H $(PWD) docker://$(IMAGE) $(NAME)

list:
	singularity instance list

shell:
	singularity exec instance://$(NAME) bash --init-file .singular-prompt

spinodal: spinodal.py
	singularity exec instance://$(NAME) $(MPI) -np $(RANKS) $(PY3) -u spinodal.py

stop:
	singularity instance stop $(NAME)

watch:
	singularity exec instance://$(NAME) bash -c "watch cat fenics-bm-1b.csv"
