# PFHub BM1 using FEniCS in a container

IMAGE = tkphd/fenics-petsc-signac
WRK = /root/shared
NP = 4

all: spinodal
.PHONY: all format init lint shell singodal spinodal start stop watch

spinodal: spinodal.py
	mpirun -np $(NP) python3 -u spinodal.py 100000

init: init.py
	docker run --rm -ti -v $(PWD):$(WRK) -w $(WRK) $(IMAGE) \
		python3 init.py

# singodal: spinodal.py
# 	singularity exec instance://pfhub mpirun -np $(NP) python3 -u spinodal.py 100000

# format: spinodal.py
# 	yapf --style=.style.yapf -i $<

# lint: spinodal.py
# 	pycodestyle --ignore=E128,E221,E402,W503 $<

# list:
# 	singularity instance list

# shell:
# 	singularity exec instance://pfhub bash --init-file .singular-prompt

# start:
# 	singularity instance start -H $(PWD) docker://$(IMAGE) pfhub

# stop:
# 	singularity instance stop pfhub

# watch:
# 	watch singularity exec instance://pfhub "tail -n 40 fenics-bm-1b.csv | column -s, -t"
