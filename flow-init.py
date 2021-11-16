# p V = N kT

# Input: p, N, kT
# Output: V, runtime, ...

import signac

# https://docs.signac.io/projects/core/en/latest/api.html#the-project
project = signac.init_project("pfhub", workspace="bm1b") # --> signac.rc

sp = {
    # Discretization parameters
    "cells": 200, # cells
    "mindt": 0.1, # minimum (initial) timestep
    "crank": 0.5, # Crank-Nicolson parameter

    # Finite element parameters
    "poly": 1,
    "quad": 2,
    "diag": "crossed",
    "solver": "snes",
}

for n in (200, 250, 320):
    sp["cells"] = n
    dx = 200. / n
    sp["mindt"] = dx**4 / 10  # with M = 5 and kappa = 2, Mk = 10
    for c in (0.5, 1):
        sp["crank_nicolson"] = c
        for d in ("right", "crossed"):
            sp["diagonal"] = d
            for p in (1, 2):
                sp["poly"] = p
                for q in (p+1, p+2):
                    sp["quad"] = q
                    # Get a job handle associated with a state point
                    # https://docs.signac.io/projects/core/en/latest/api.html#the-job-class
                    job = project.open_job(sp) # --> $handle
                    job.init() # --> workspace/$handle
