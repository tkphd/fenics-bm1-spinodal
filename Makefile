# GNU Makefile for PFHub BM1
# Recommended for use with a Conda environment for Singularity with Python 3

# Cluster Settings

MPI = mpirun
PY3 = python3
RANKS = 4

# Container Settings

IMAGE = quay.io/fenicsproject/stable
NAME = pfhub

# Make Targets

all: fenics-bm-1b.xdmf
.PHONY: all clean shell start stop watch

fenics-bm-1b.xdmf: spinodal.py
	singularity exec instance://$(NAME) $(MPI) -np $(RANKS) $(PY3) -u spinodal.py

clean:
	rm -vf *.csv *.h5 *.log *.xdmf

list:
	singularity instance list

shell:
	singularity exec instance://$(NAME) bash --init-file .singular-prompt

start:
	singularity instance start -H $(PWD) docker://$(IMAGE) $(NAME)

stop:
	singularity instance stop $(NAME)

watch:
	singularity exec instance://$(NAME) bash -c "watch cat fenics-bm-1b.csv"
