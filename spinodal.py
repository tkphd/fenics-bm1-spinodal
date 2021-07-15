# -*- coding: utf-8 -*-
"""
PFHub Benchmark 1: Spinodal Decomposition
Implemented using FEniCS by Trevor Keller (@tkphd, <trevor.keller@nist.gov>)

    𝓕 = ∫{𝐹 + ½⋅𝜅⋅|∇𝑐|²}⋅d𝛀
    𝐹 = 𝜌⋅(𝑐 - 𝛼)²⋅(𝛽 - 𝑐)²

    𝜕𝑐/𝜕𝑡= ∇⋅{𝑀 ∇(𝑓 - 𝜅∇²𝑐)}

    𝛀 = [0,200) × [0,200)

Endpoint detection based on Δ𝜇 is borrowed from @smondal44,
<https://github.com/smondal44/spinodal-decomposition>

Usage:  mpirun -np 4 --mca opal_cuda_support 0 python -u spinodal.py
"""

from mpi4py import MPI
epoch = MPI.Wtime()

import csv
from datetime import timedelta
import gc
import numpy as np
import queue

from dolfinx import Form, Function, FunctionSpace, NewtonSolver, RectangleMesh
from dolfinx import fem, log
from dolfinx.cpp.mesh import CellType
from dolfinx.fem.problem import NonlinearProblem
from dolfinx.fem.assemble import assemble_matrix, assemble_scalar, assemble_vector
from dolfinx.io import XDMFFile
from os import path
from petsc4py import PETSc
from sys import argv
from ufl import FiniteElement, Measure, TestFunctions, TrialFunction
from ufl import derivative, diff, grad, inner, split, variable
from ufl import dx as Δ𝑥

# Model parameters
𝜅 = 2  # gradient energy coefficient
𝜌 = 5  # well height
𝛼 = 0.3  # eqm composition of phase 1
𝛽 = 0.7  # eqm composition of phase 2
𝜁 = 0.5  # system composition
𝑀 = 5  # interface mobility
𝜀 = 0.01  # noise amplitude

# Discretization parameters
𝑊 = 200  # width
𝑁 = 400  # cells
𝑡 = 0.0  # simulation time
Δ𝑡 = 0.125  # timestep
𝑇 = 1e6  # simulation timeout

if len(argv) == 2:
    if np.isfinite(int(argv[1])):
        𝑇 = int(argv[1])

p_deg = 2  # element/polynomial degree
q_deg = 4  # quadrature_degree

# Output -- check if there's already data here
log.set_output_file("fenics-spinodal.log")
bm1_log = "fenics-bm-1b.csv"
xdmf_file = "fenics-bm-1b.xdmf"
resuming = path.isfile(xdmf_file)

COMM = MPI.COMM_WORLD
rank = MPI.COMM_WORLD.Get_rank()

class CahnHilliardEquation(NonlinearProblem):
    def __init__(self, F, x):
        NonlinearProblem.__init__(self, F, x)

    def form(self, x):
        x.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

    def matrix(self):
        return fem.create_matrix(self.a)

    def vector(self):
        return fem.create_vector(self.L)

def print0(s):
    if rank == 0:
        print(s)

𝛀 = RectangleMesh(
    COMM,
    [np.array([0, 0, 0]), np.array([𝑊, 𝑊, 0])],
    [𝑁, 𝑁],
    CellType.triangle,
    diagonal="crossed",
)

COMM = 𝛀.mpi_comm()

LE = FiniteElement("Lagrange", 𝛀.ufl_cell(), p_deg)

# Create the function space from both the mesh and the element
FS = FunctionSpace(𝛀, LE * LE)

# Build the solution, trial, and test functions
𝒖 = Function(FS)  # current solution
𝒖0 = Function(FS)  # previous solution
d𝒖 = TrialFunction(FS)
𝑞, 𝑣 = TestFunctions(FS)

# Mixed functions
𝑐, 𝜇 = split(𝒖)  # references to components of 𝒖 for clear, direct access
d𝑐, d𝜇 = split(d𝒖)
𝑏, 𝜆 = split(𝒖0)  # 𝑏, 𝜆 are the previous values for 𝑐, 𝜇

𝑐 = variable(𝑐)
𝐹 = 𝜌 * (𝑐 - 𝛼) ** 2 * (𝛽 - 𝑐) ** 2
𝑓 = diff(𝐹, 𝑐)

# === Weak Form ===

# Half-stepping parameter for Crank-Nicolson
𝜃 = 0.5  # Crank-Nicolson parameter
𝜇_mid = (1 - 𝜃) * 𝜆 + 𝜃 * 𝜇

# Time discretization in UFL syntax
# (𝑏 is the previous timestep)
L0 = inner(𝑐, 𝑞) * Δ𝑥 - inner(𝑏, 𝑞) * Δ𝑥 + Δ𝑡 * inner(grad(𝜇_mid), grad(𝑞)) * Δ𝑥
L1 = inner(𝜇, 𝑣) * Δ𝑥 - inner(𝑓, 𝑣) * Δ𝑥 - 𝜅 * inner(grad(𝑐), grad(𝑣)) * Δ𝑥
𝐿 = L0 + L1

# === Solver ===

problem = CahnHilliardEquation(𝐿, 𝒖)
solver = NewtonSolver(COMM, problem)
solver.setF(problem.F, problem.vector())
solver.setJ(problem.J, problem.matrix())
solver.convergence_criterion = "incremental"
solver.rtol = 1e-4
solver.atol = 1e-8

# PETSc options
opts = PETSc.Options()
opts["quadrature_degree"] = q_deg
opts["optimize"] = True
opts["cpp_optimize"] = True
opts["representation"] = "uflacs"
opts["linear_algebra_backend"] = "PETSc"

ksp = solver.krylov_solver
ksprefix = ksp.getOptionsPrefix()
#opts[f"{ksprefix}ksp_type"] = "preonly"
opts[f"{ksprefix}pc_type"] = "lu" # Jacobi SOR
opts[f"{ksprefix}pc_factor_mat_solver_type"] = "mumps"
ksp.setFromOptions()

# === Initial Conditions ===

with 𝒖.vector.localForm() as x:
    x.set(0.0)

noisy = lambda x: 𝜁 + 𝜀 * (
    np.cos(0.105 * x[0]) * np.cos(0.11 * x[1])
    + (np.cos(0.13 * x[0]) * np.cos(0.087 * x[1])) ** 2
    + np.cos(0.025 * x[0] - 0.15 * x[1]) * np.cos(0.07 * x[0] - 0.02 * x[1])
)

𝒖.sub(0).interpolate(noisy)

with XDMFFile(COMM, xdmf_file, "w") as xdmf:
    try:
        xdmf.write_mesh(𝛀)
    except IOError as e:
        MPI.Abort(e)

𝒖.vector.copy(result=𝒖0.vector)
𝒖0.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

# === TIMESTEPPING ===

# Enqueue output timestamps
io_q = queue.Queue()

for t_out in np.arange(0, 1, Δ𝑡):
    io_q.put(t_out)
for n in np.arange(0, 7):
    for m in np.arange(1, 10):
        t_out = m * 10.0 ** n
        if t_out <= 𝑇:
            io_q.put(t_out)

start = MPI.Wtime()


def crunch_the_numbers(𝛀, 𝑡, 𝑐, 𝜇, 𝜆, r, τ):
    𝒎 = assemble_scalar(𝑐 * Δ𝑥)
    𝓕 = assemble_scalar(
        𝜌 * (𝑐 - 𝛼) ** 2 * (𝛽 - 𝑐) ** 2 * Δ𝑥 + 0.5 * 𝜅 * inner(grad(𝑐), grad(𝑐)) * Δ𝑥
    )
    𝜂 = assemble_scalar(np.abs(𝜇 - 𝜆) * Δ𝑥)
    𝑛 = COMM.allreduce(len(𝛀.geometry.x), op=MPI.SUM)

    𝐜 = COMM.allreduce(𝒎 / 𝑛, op=MPI.SUM)
    𝐅 = COMM.allreduce(𝓕, op=MPI.SUM)
    𝛈 = COMM.allreduce(𝜂 / 𝑛, op=MPI.SUM)
    𝐫 = COMM.allreduce(r, op=MPI.MAX)
    𝛕 = MPI.Wtime() - τ

    return (𝑡, 𝐜, 𝐅, 𝛈, 𝐫, 𝛕)


def write_csv_header(filename):
	if rank == 0:
	    with open(filename, mode="w") as nrg_file:
                header = [
                    "time",
                    "composition",
                    "free_energy",
                    "driving_force",
                    "iterations",
                    "runtime",
                ]

                try:
                    io = csv.writer(nrg_file)
                    io.writerow(header)
                except IOError as e:
                    MPI.Abort(e)

def write_csv_summary(filename, summary):
	if rank == 0:
	    with open(filename, mode="a") as nrg_file:
                try:
                    io = csv.writer(nrg_file)
                    io.writerow(summary)
                except IOError as e:
                    MPI.Abort(e)

write_csv_header(bm1_log)
write_csv_summary(bm1_log, crunch_the_numbers(𝛀, 𝑡, 𝑐, 𝜇, 𝜆, 0, start))

Δ𝜇 = 1.0
io_t = io_q.get()

print0("[{}] Next summary at 𝑡={}".format(
        timedelta(seconds=(MPI.Wtime() - epoch)), io_t)
)

while (Δ𝜇 > 1e-8) and (𝑡 < 𝑇):
    𝑡 += Δ𝑡
    r = solver.solve(𝒖)[0]
    𝒖.vector.copy(result=𝒖0.vector)

    if np.isclose(𝑡, io_t) or 𝑡 > io_t:

        with XDMFFile(COMM, xdmf_file, "a") as xdmf:
            try:
                xdmf.write_function(𝒖.sub(0), 𝑡)
            except IOError as e:
                MPI.Abort(e)

        write_csv_summary(bm1_log, crunch_the_numbers(𝛀, 𝑡, 𝑐, 𝜇, 𝜆, r, start))

        io_t = io_q.get()

        print0("[{}] Next summary at 𝑡={}".format(
                timedelta(seconds=(MPI.Wtime() - epoch)), io_t)
        )

        gc.collect()

write_csv_summary(bm1_log, crunch_the_numbers(𝛀, 𝑡, 𝑐, 𝜇, 𝜆, r, start))

print0("Finished simulation after {} s.".format(MPI.Wtime() - epoch))
