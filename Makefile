# GNU Makefile for PFHub BM1
# Recommended for use with a Conda environment for Singularity with Python 3

# Cluster Settings

MPI = mpirun
PY3 = python3
RANKS = 4
WRK = workspace/cff295345c25cacade5b041b11edb5c6

# Container Settings

IMAGE = quay.io/fenicsproject/stable
NAME = pfhub

# Make Targets

all: spinodal
.PHONY: all clean format lint shell singodal spinodal start stop watch

spinodal: spinodal.py
	$(MPI) -np $(RANKS) $(PY3) -u spinodal.py 100000

singodal: spinodal.py
	singularity exec instance://$(NAME) $(MPI) -np $(RANKS) $(PY3) -u spinodal.py 100000

clean:
	rm -vf $(WRK)/*.csv $(WRK)/*.h5 $(WRK)/*.log $(WRK)/*.xdmf

format: spinodal.py
	yapf --style=.style.yapf -i $<

lint: spinodal.py
	pycodestyle --ignore=E128,E221,E402,W503 $<

list:
	singularity instance list

shell:
	singularity exec instance://$(NAME) bash --init-file .singular-prompt

start:
	singularity instance start -H $(PWD) docker://$(IMAGE) $(NAME)

stop:
	singularity instance stop $(NAME)

watch:
	watch singularity exec instance://$(NAME) "tail -n 40 fenics-bm-1b.csv | column -s, -t"
