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
import random
import queue

from os import path
from petsc4py import PETSc
from sys import argv

from dolfin import Function, FunctionSpace, LogLevel, MixedElement, NewtonSolver, NonlinearProblem, Point, RectangleMesh, UserExpression, XDMFFile
from dolfin import FiniteElement, TestFunctions, TrialFunction
from dolfin import assemble, parameters, cos, derivative, diff, dot, grad, set_log_level
from dolfin import cos, derivative, diff, dot, grad, project, split, variable
from dolfin import dx as Δ𝑥

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
if (len(argv) == 2) and (np.isfinite(int(argv[1]))):
    𝑇 = int(argv[1])

# Output -- check if there's already data here
bm1_log = "fenics-bm-1b.csv"
xdmf_file = "fenics-bm-1b.xdmf"
field_names = ("𝑐", "𝜇")

COMM = MPI.COMM_WORLD
rank = MPI.COMM_WORLD.Get_rank()
set_log_level(LogLevel.ERROR)

poly_deg = 2  # polynomial degree
quad_deg = 4  # quadrature degree

parameters["form_compiler"]["optimize"] = True
parameters["form_compiler"]["cpp_optimize"] = True
parameters["form_compiler"]["representation"] = "uflacs"


class CahnHilliardEquation(NonlinearProblem):
    def __init__(self, a, L):
        NonlinearProblem.__init__(self)
        self.a = a
        self.L = L

    def F(self, b, x):
        assemble(self.L, tensor=b)

    def J(self, A, x):
        assemble(self.a, tensor=A)


class InitialConditions(UserExpression):
    def eval(self, values, x):
        values[0] = 𝜁 + 𝜀 * (
            cos(0.105 * x[0]) * cos(0.11 * x[1]) +
            (cos(0.13 * x[0]) * cos(0.087 * x[1]))**2 +
            cos(0.025 * x[0] - 0.15 * x[1]) * cos(0.07 * x[0] - 0.02 * x[1]))
        values[1] = 0.0

    def value_shape(self):
        return (2, )


def crunch_the_numbers(𝛀, 𝑡, 𝑐, 𝜇, 𝜆, r, τ):
    𝒎 = assemble(𝑐 * Δ𝑥)
    𝓕 = assemble(𝜌 * (𝑐 - 𝛼)**2 * (𝛽 - 𝑐)**2 * Δ𝑥) \
      + assemble(0.5 * 𝜅 * dot(grad(𝑐), grad(𝑐)) * Δ𝑥)
    𝜂 = assemble(np.abs(𝜇 - 𝜆) * Δ𝑥)
    𝑛 = COMM.allreduce(len(𝛀.coordinates()), op=MPI.SUM)

    𝐜 = COMM.allreduce(𝒎 / 𝑛, op=MPI.SUM)
    𝐅 = COMM.allreduce(𝓕, op=MPI.SUM)
    𝛈 = COMM.allreduce(𝜂 / 𝑛, op=MPI.SUM)
    𝐫 = COMM.allreduce(r, op=MPI.MAX)
    𝛕 = MPI.Wtime() - τ

    return (𝑡, 𝐜, 𝐅, 𝛈, 𝐫, 𝛕)


def print0(s):
    if rank == 0:
        print(s)

def set_file_params(file):
    file.parameters["flush_output"] = True
    file.parameters["rewrite_function_mesh"] = False
    file.parameters["functions_share_mesh"] = True


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

# Define domain and finite element
𝛀 = RectangleMesh(COMM, Point([0, 0]), Point([𝑊, 𝑊]), 𝑁, 𝑁, diagonal="crossed")
𝓟 = FiniteElement("Lagrange", 𝛀.ufl_cell(), poly_deg)

# Create the function space from both the mesh and the element
# 𝕊 = FunctionSpace(𝛀, 𝓟 * 𝓟)
𝕊 = FunctionSpace(𝛀, MixedElement([𝓟, 𝓟]))

# Build the solution, trial, and test functions
𝒖 = Function(𝕊)  # current solution
𝒐 = Function(𝕊)  # old (previous) solution
d𝒖 = TrialFunction(𝕊)
𝑞, 𝑣 = TestFunctions(𝕊)

# Mixed functions
𝑐, 𝜇 = split(𝒖)  # references to components of 𝒖 for clear, direct access
d𝑐, d𝜇 = split(d𝒖)
𝑏, 𝜆 = split(𝒐)  # 𝑏, 𝜆 are the previous values for 𝑐, 𝜇

𝑐 = variable(𝑐)
𝜇 = variable(𝜇)

𝐹 = 𝜌 * (𝑐 - 𝛼)**2 * (𝛽 - 𝑐)**2
𝑓 = diff(𝐹, 𝑐)

# === Weak Form ===

# Half-stepping parameter for Crank-Nicolson
𝜃 = 0.5  # Crank-Nicolson parameter
𝜇_mid = (1 - 𝜃) * 𝜆 + 𝜃 * 𝜇

# Time discretization in UFL syntax
𝐿0 = 𝑐 * 𝑞 * Δ𝑥 - 𝑏 * 𝑞 * Δ𝑥 + Δ𝑡 * dot(grad(𝜇_mid), grad(𝑞)) * Δ𝑥
𝐿1 = 𝜇 * 𝑣 * Δ𝑥 - 𝑓 * 𝑣 * Δ𝑥 - 𝜅 * dot(grad(𝑐), grad(𝑣)) * Δ𝑥

𝐿 = 𝐿0 + 𝐿1
𝐽 = derivative(𝐿, 𝒖, d𝒖)

# === Solver ===

problem = CahnHilliardEquation(𝐽, 𝐿)
solver = NewtonSolver(COMM)

solver.parameters["linear_solver"] = "lu"
solver.parameters["convergence_criterion"] = "incremental"
solver.parameters["relative_tolerance"] = 1e-4

# PETSc options
opts = PETSc.Options()
opts["optimize"] = True
opts["cpp_optimize"] = True
opts["representation"] = "uflacs"
opts["linear_algebra_backend"] = "PETSc"

# Krylov preconditioner options
solver.parameters["krylov_solver"]["absolute_tolerance"] = 1e-14
solver.parameters["krylov_solver"]["relative_tolerance"] = 1e-8

# === Initial Conditions ===

𝒊 = InitialConditions()
𝒖.interpolate(𝒊)
𝒐.interpolate(𝒊)

with XDMFFile(COMM, xdmf_file) as xdmf:
    set_file_params(xdmf)
    # write mesh
    try:
        xdmf.write(𝛀)
    except IOError as e:
        MPI.Abort(e)
    # write initial condition
    for i, f in enumerate(𝒖.split()):
        try:
            f.rename(field_names[i], field_names[i])
            xdmf.write(f, 0.0)
        except IOError as e:
            MPI.Abort(e)

# === TIMESTEPPING ===

# Enqueue output timestamps
io_q = queue.Queue()

for t_out in np.arange(0, 1, Δ𝑡):
    io_q.put(t_out)
for n in np.arange(0, 7):
    for m in np.arange(1, 10):
        t_out = m * 10.0**n
        if t_out <= 𝑇:
            io_q.put(t_out)

start = MPI.Wtime()

write_csv_header(bm1_log)
write_csv_summary(bm1_log, crunch_the_numbers(𝛀, 𝑡, 𝑐, 𝜇, 𝜆, 0, start))

Δ𝜇 = 1.0
io_t = io_q.get()

print0("[{}] Next summary at 𝑡={}".format(
    timedelta(seconds=(MPI.Wtime() - epoch)), io_t))

converged = True

while (converged) and (Δ𝜇 > 1e-8) and (𝑡 < 𝑇):
    𝑡 += Δ𝑡
    𝒐.vector()[:] = 𝒖.vector()

    i, converged = solver.solve(problem, 𝒖.vector())

    if np.isclose(𝑡, io_t) or 𝑡 > io_t:
        with XDMFFile(COMM, xdmf_file) as xdmf:
            set_file_params(xdmf)
            for i, f in enumerate(𝒖.split()):
                try:
                    f.rename(field_names[i], field_names[i])
                    xdmf.write(f, 𝑡)
                except IOError as e:
                    MPI.Abort(e)

        write_csv_summary(bm1_log, crunch_the_numbers(𝛀, 𝑡, 𝑐, 𝜇, 𝜆, i, start))

        io_t = io_q.get()

        print0("[{}] Next summary at 𝑡={}".format(
            timedelta(seconds=(MPI.Wtime() - epoch)), io_t))

        gc.collect()

print0("Finished simulation after {} s.".format(MPI.Wtime() - epoch))
