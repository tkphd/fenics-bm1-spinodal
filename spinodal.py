"""
PFHub Benchmark 1: Spinodal Decomposition
Implemented using FEniCS by Trevor Keller (@tkphd, <trevor.keller@nist.gov>)
with substantial optimization from Nana Ofori-Opoku (@noforiopoku)

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

from os import getpid, path
from petsc4py import PETSc
from sys import argv

from dolfin import FiniteElement, FunctionSpace, MixedElement
from dolfin import Function, TestFunctions, TrialFunction
from dolfin import LagrangeInterpolator, NewtonSolver, NonlinearProblem
from dolfin import Mesh, Point, RectangleMesh, UserExpression
from dolfin import XDMFFile
from dolfin import LogLevel, set_log_level
from dolfin import cos, derivative, grad, inner, sin, variable
from dolfin import assemble, parameters, split
from dolfin import dx as Δ𝑥

from ufl import replace

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
𝑁 = 200  # cells
𝑡 = 0.0  # simulation time
Δ𝑡 = 0.125  # timestep
𝜃 = 0.5  # Crank-Nicolson parameter
𝑇 = 1e6  # simulation timeout
poly_deg = 1  # polynomial degree, adds degrees of freedom
quad_deg = 2  # quadrature degree, at least 2 poly_deg
field_names = ("𝑐", "𝜇")

# Read runtime from command line
if (len(argv) == 2) and (np.isfinite(int(argv[1]))):
    𝑇 = int(argv[1])

# Output -- check if there's already data here
bm1_log = "fenics-bm-1b.csv"
bm1_viz = "fenics-bm-1b.xdmf"
bm1_chk = "checkpoint.xdmf"

COMM = MPI.COMM_WORLD
rank = MPI.COMM_WORLD.Get_rank()
set_log_level(LogLevel.ERROR)


def weak_form(𝒖, 𝒐, ℝ, 𝛀, 𝐸):
    # Define the 𝑐 function based on the real space
    𝑝, 𝑞 = TestFunctions(ℝ)
    𝑐, 𝜇 = split(𝒖)  # references to components of 𝒖 for clear, direct access
    𝑏, 𝜆 = split(𝒐)  # 𝑏, 𝜆 are the previous values for 𝑐, 𝜇
    𝜇𝜃 = (1 - 𝜃) * 𝜆 + 𝜃 * 𝜇  # Crank-Nicolson mid-step solution

    𝑪 = inner(𝑐 - 𝑏, 𝑝) * Δ𝑥 + Δ𝑡 * 𝑀 * inner(grad(𝜇𝜃), grad(𝑝)) * Δ𝑥

    # Define the 𝜇 function based on the virtual space
    𝕍 = FunctionSpace(𝛀, 𝐸)
    𝒗 = Function(𝕍)
    𝒙 = variable(𝒗)

    𝐹 = 𝜌 * (𝒙 - 𝛼)**2 * (𝛽 - 𝒙)**2 + 0.5 * 𝜅 * inner(grad(𝒙), grad(𝒙))
    𝑓 = replace(derivative(𝐹, 𝒗, 𝑞), {𝒗: 𝑐})

    𝑼 = (𝜇 * 𝑞 - 𝑓) * Δ𝑥

    𝐹 = replace(𝐹, {𝒗: 𝑐})
    𝐿 = 𝑪 + 𝑼

    return 𝐹, 𝐿


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
        cA = cos(0.105 * x[0]) * cos(0.110 * x[1])
        cB = (cos(0.130 * x[0]) * cos(0.087 * x[1]))**2
        cC = (cos(0.025 * x[0] - 0.150 * x[1])
              * cos(0.070 * x[0] - 0.020 * x[1]))

        values[0] = 𝜁 + 𝜀 * (cA + cB + cC)

        uA = 𝜀 * 𝜅 * (-0.0095 * sin(0.025 * x[0] - 0.15 * x[1])
                      * sin(0.07 * x[0] - 0.02 * x[1])
                      + 0.023125 * cos(0.105 * x[0]) * cos(0.11 * x[1])
                      + 0.097876 * cos(0.13 * x[0])**2 * cos(0.087 * x[1])**2
                      - 0.015138 * cos(0.13 * x[0])**2
                      - 0.0338 * cos(0.087 * x[1])**2
                      + 0.028425 * cos(0.025 * x[0] - 0.15 * x[1])
                      * cos(0.07 * x[0] - 0.02 * x[1]))
        uB = 2 * 𝜌 * (-𝛼 + 𝜀
                      * (cos(0.105 * x[0]) * cos(0.11 * x[1])
                         + cos(0.13 * x[0])**2 * cos(0.087 * x[1])**2
                         + cos(0.025 * x[0] - 0.15 * x[1])
                         * cos(0.07 * x[0] - 0.02 * x[1])) + 𝜁)**2
        uC = (𝜁 - 𝛽 + 𝜀
              * (cos(0.105 * x[0]) * cos(0.11 * x[1])
                 + cos(0.13 * x[0])**2 * cos(0.087 * x[1])**2
                 + cos(0.025 * x[0] - 0.15 * x[1])
                 * cos(0.07 * x[0] - 0.02 * x[1])))
        uD = 2 * 𝜌 * (𝜁 - 𝛼 + 𝜀
                      * (cos(0.105 * x[0]) * cos(0.11 * x[1])
                         + cos(0.13 * x[0])**2 * cos(0.087 * x[1])**2
                         + cos(0.025 * x[0] - 0.15 * x[1])
                      * cos(0.07 * x[0] - 0.02 * x[1])))
        uE = (𝜁 - 𝛽 + 𝜀
              * (cos(0.105 * x[0]) * cos(0.11 * x[1])
                 + cos(0.13 * x[0])**2 * cos(0.087 * x[1])**2
                 + cos(0.025 * x[0] - 0.15 * x[1])
                 * cos(0.07 * x[0] - 0.02 * x[1])))**2

        values[1] = uA + uB * uC + uD * uE

    def value_shape(self):
        return (2, )


def crunch_the_numbers(𝛀, 𝑡, 𝑐, 𝐹, 𝜇, 𝜆, i, 𝜈, τ):
    𝑛 = len(𝛀.coordinates())
    𝐦 = assemble(𝑐 * Δ𝑥) / 𝑊**2
    𝐅 = assemble(𝐹 * Δ𝑥)
    𝛈 = assemble(np.abs(𝜇 - 𝜆) / 𝑛 * Δ𝑥)
    𝐢 = COMM.allreduce(i, op=MPI.MAX)
    𝛎 = COMM.allreduce(𝜈, op=MPI.MIN)
    𝛕 = MPI.Wtime() - τ

    pid = getpid()
    status = open("/proc/%d/status" % pid).read()

    mem_now = int(status.split("VmSize:")[1].split("kB")[0]) / 1024.
    mem_max = int(status.split("VmPeak:")[1].split("kB")[0]) / 1024.

    mem_now = COMM.allreduce(mem_now, op=MPI.SUM)
    mem_max = COMM.allreduce(mem_max, op=MPI.SUM)

    return (𝑡, 𝐦, 𝐅, 𝛈, 𝐢, 𝛎, 𝛕, mem_now, mem_max)


def guesstimate(rate, t_now, t_nxt):
    est_nxt = timedelta(seconds=int((viz_t - 𝑡) / (Δ𝑡 * rate)))
    est_all = timedelta(seconds=int((𝑇 - 𝑡) / (Δ𝑡 * rate)))
    return (est_nxt, est_all)


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
                "time", "composition", "free_energy", "driving_force", "its",
                "sim_rate", "runtime", "memory", "max_mem"
            ]

            try:
                io = csv.writer(nrg_file)
                io.writerow(header)
            except IOError as e:
                MPI.Abort(e)


def write_csv(filename, summary):
    if rank == 0:
        with open(filename, mode="a") as nrg_file:
            try:
                io = csv.writer(nrg_file)
                io.writerow(summary)
            except IOError as e:
                MPI.Abort(e)


def write_viz(xdmf, 𝛀, 𝒖, 𝑡=0.0):
    try:
        if np.isclose(0, 𝑡):
            xdmf.write(𝛀)
        for n, f in enumerate(𝒖.split()):
            f.rename(field_names[n], field_names[n])
            xdmf.write(f, 𝑡)
            xdmf.close()
    except IOError as e:
        MPI.Abort(e)


resuming = path.exists(bm1_chk)
if resuming:
    print0("Resuming simulation from {}.".format(bm1_chk))

# Define domain and finite element
𝛀 = RectangleMesh(COMM, Point([0, 0]), Point([𝑊, 𝑊]), 𝑁, 𝑁, diagonal="crossed")
𝓟 = FiniteElement("Lagrange", 𝛀.ufl_cell(), poly_deg)
𝐸 = MixedElement([𝓟, 𝓟])

# Create the function space from both the mesh and the element
𝕊 = FunctionSpace(𝛀, 𝐸)
d𝒖 = TrialFunction(𝕊)

# Build the solution, trial, and test functions
𝒖 = Function(𝕊)  # current solution
𝒐 = Function(𝕊)  # old (previous) solution
𝑐, 𝜇 = split(𝒖)  # references to components of 𝒖 for clear, direct access
𝑏, 𝜆 = split(𝒐)  # 𝑏, 𝜆 are the previous values for 𝑐, 𝜇

if resuming:
    with XDMFFile(COMM, bm1_chk) as chk:
        chk.read_checkpoint(𝒖, "u")

        attr = chk.attributes("u")
        𝑡 = attr["time"]
        Δ𝑡 = attr["timestep"]

# === Weak Form ===
𝐹, 𝐿 = weak_form(𝒖, 𝒐, 𝕊, 𝛀, 𝓟)
𝑱 = derivative(𝐿, 𝒖, d𝒖)

# === Solver ===

problem = CahnHilliardEquation(𝑱, 𝐿)
solver = NewtonSolver(COMM)

solver.parameters["linear_solver"] = "lu"
solver.parameters["relative_tolerance"] = 1e-3
solver.parameters["absolute_tolerance"] = 1e-7
solver.parameters["convergence_criterion"] = "incremental"
solver.parameters["error_on_nonconvergence"] = True

parameters["linear_algebra_backend"] = "PETSc"
parameters["form_compiler"]["optimize"] = True
parameters["form_compiler"]["cpp_optimize"] = True
parameters["form_compiler"]["representation"] = "uflacs"
parameters["form_compiler"]["quadrature_degree"] = quad_deg

# === Initial Conditions ===

xdmf = XDMFFile(COMM, bm1_viz)
set_file_params(xdmf)

if not resuming:
    𝒊 = InitialConditions(degree=poly_deg)
    LagrangeInterpolator.interpolate(𝒖, 𝒊)
    LagrangeInterpolator.interpolate(𝒐, 𝒊)

    write_viz(xdmf, 𝛀, 𝒖)

# Enqueue output timestamps
viz_q = queue.Queue()
nrg_q = queue.Queue()

for t_out in (1, 2, 5):
    if 𝑡 < t_out:
        viz_q.put(int(t_out))
        nrg_q.put(int(t_out))
for n in np.arange(1, 7):
    step = min(int(10**n), 1000)
    for t_out in np.arange(10**n, 10 * 10**n, step):
        if 𝑡 < t_out and t_out <= 𝑇:
            viz_q.put(int(t_out))
            for k in (-1, 0, 1):
                t_nrg = t_out + k
                if 𝑡 < t_nrg and t_nrg <= 𝑇:
                    nrg_q.put(int(t_nrg))

viz_t = viz_q.get()
nrg_t = nrg_q.get()

# === TIMESTEPPING ===

start = MPI.Wtime()

# Guess initial rate based on 4-core CPU
rate = 0.5 * (𝑁 / 400)**2 * (4.0 / COMM.Get_size())

if not resuming:
    write_csv_header(bm1_log)
    write_csv(bm1_log,
              crunch_the_numbers(𝛀, 𝑡, 𝑐, 𝐹, 𝜇, 𝜆, 0, rate, start))

print0("[{}] Simulation started.".format(
    timedelta(seconds=int((MPI.Wtime() - epoch)))))

est_t, all_t = guesstimate(rate, 𝑡, viz_t)
print0("[{}] ETA: 𝑡={} in {}, 𝑡={} in {}".format(
    timedelta(seconds=int((MPI.Wtime() - epoch))),
    viz_t, est_t, 𝑇, all_t))

nits = 0
itime = MPI.Wtime()

# Main time-stepping loop
while (𝑡 < 𝑇):
    𝒐.assign(𝒖)
    its, converged = solver.solve(problem, 𝒖.vector())

    𝑡 += Δ𝑡
    nits += 1

    if not converged:
        MPI.Abort("Failed to converge!")

    if np.isclose(𝑡, nrg_t) or 𝑡 > nrg_t:
        # write free energy summary
        rate = float(nits) / (MPI.Wtime() - itime)
        write_csv(bm1_log,
                  crunch_the_numbers(𝛀, 𝑡, 𝑐, 𝐹, 𝜇, 𝜆, its, rate, start))

        if not nrg_q.empty():
            nrg_t = nrg_q.get()

    if np.isclose(𝑡, viz_t) or 𝑡 > viz_t:
        write_viz(xdmf, 𝛀, 𝒖, 𝑡)
        with XDMFFile(COMM, bm1_chk) as chk:
            chk.write_checkpoint(𝒖, "u", 𝑡)

            attr = chk.attributes("u")
            attr["time"] = 𝑡
            attr["timestep"] = Δ𝑡

        if not viz_q.empty():
            viz_t = viz_q.get()
            est_t, all_t = guesstimate(rate, 𝑡, viz_t)
            print0("[{}] ETA: 𝑡={} in {}, 𝑡={} in {}".format(
                timedelta(seconds=int((MPI.Wtime() - epoch))),
                viz_t, est_t, 𝑇, all_t))

        gc.collect()
        nits = 0
        itime = MPI.Wtime()

xdmf.close()
print0("[{}] Simulation complete.".format(
    timedelta(seconds=int((MPI.Wtime() - epoch)))))
