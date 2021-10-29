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

Tunable Parameters:

* Mesh resolution (reciprocal) = {200, 150, 100, 50}
* Stability criterion [Timestep = {2, 4, 6, 8} × 10⁻⁶]
* Polynomial degree = {1, 2, 3}
* Relative error = {0.01, 1, 10, 100} × 10⁻⁶
"""

from mpi4py import MPI

epoch = MPI.Wtime()
COMM = MPI.COMM_WORLD
rank = MPI.COMM_WORLD.Get_rank()
mpi_root = (rank == 0)

import csv
from datetime import timedelta
import gc
import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
import pandas as pd
import queue
import signac

from os import getpid, path
from petsc4py import PETSc
from sys import argv

from dolfin import FiniteElement, FunctionSpace, MixedElement
from dolfin import Function, TestFunctions, TrialFunction
from dolfin import LagrangeInterpolator, NewtonSolver, NonlinearProblem
from dolfin import CellType, Mesh, Point, RectangleMesh, UserExpression
from dolfin import HDF5File, XDMFFile
from dolfin import LogLevel, set_log_level
from dolfin import cos, derivative, grad, inner, sin, variable
from dolfin import assemble, parameters, split
from dolfin import dx as Δ𝑥

from ufl import replace

style.use("seaborn")

if mpi_root:
    # Register this simulation with Signac
    project = signac.init_project("fenics-cahn-hilliard")
else:
    project = None

# Establish "State Point" for this simulation

def set_state(P, S):
    for k, v in P.items():
        try:
            p = v.items()
            S[k] = {}
            for k1, v1 in p:
                S[k][k1] = v1
        except AttributeError:
            S[k] = v


def set_solver(solverType, state):
    if solverType == "newton":
        solver = NewtonSolver()
        prm = solver.parameters

        prm["linear_solver"] = "gmres"
        prm["relative_tolerance"] = 1E-4
        prm["absolute_tolerance"] = 1E-8
        prm["convergence_criterion"] = "incremental"  # residual, incremental
        prm["maximum_iterations"] = 20
    elif solverType == "snes":
        solver = PETScSNESSolver()
        prm = solver.parameters

        prm["method"] = "newtonls"
        prm["linear_solver"] = "gmres"
    else:
        raise ValueError("Unrecognized solver type {}".format(solverType))

    state["solver"]["name"] = solverType
    set_state(state["solver"]["parameters"], prm)
    return solver


poly_deg = 1  # polynomial degree, adds degrees of freedom
quad_deg = 1 + poly_deg  # quadrature degree

state = {
    "solver": {
        "name": "newton",
        "parameters": {}
    },
    "linalg": {
        "name": "petsc",
        "parameters": {}
    },
    "dolfin": {
        "poly_deg": poly_deg,
        "quad_deg": quad_deg,
        "parameters": {}
    },
}

solver = set_solver("newton", state)

if mpi_root:
    sigjob = project.open_job(state)
else:
    project = None

if mpi_root:
    # Model parameters
    𝜅 = sigjob.doc.grad_nrg_coeff  = 2  # gradient energy coefficient
    𝜌 = sigjob.doc.well_height     = 5  # well height
    𝛼 = sigjob.doc.a_composition   = 0.3  # eqm composition of phase 1
    𝛽 = sigjob.doc.b_composition   = 0.7  # eqm composition of phase 2
    𝜁 = sigjob.doc.avg_composition = 0.5  # system composition
    𝑀 = sigjob.doc.mobility        = 5  # interface mobility
    𝜀 = sigjob.doc.noise_amplitude = 0.01  # noise amplitude

    # Discretization parameters
    𝑊 = sigjob.doc.domain_width   = 200  # width
    𝑁 = sigjob.doc.domain_cells   = 200  # cells
    Δ𝑡= sigjob.doc.initial_dt     = 0.125  # timestep
    𝜃 = sigjob.doc.crank_nicolson = 0.5  # Crank-Nicolson parameter
    𝑇 = sigjob.doc.runtime_target = 1e6  # simulation timeout

    # Read runtime from command line
    if (len(argv) == 2) and (np.isfinite(int(argv[1]))):
        𝑇 = int(argv[1])

    bm1_log = sigjob.fn("bm1b.csv")
    bm1_viz = sigjob.fn("bm1b.xdmf")
    bm1_chk = sigjob.fn("checkpoint.h5")
else:
    𝜅 = 𝜌 = 𝛼 = 𝛽 = 𝜁 = 𝑀 = 𝜀 = 𝑊 = 𝑁 = Δ𝑡 = 𝜃 = 𝑇 = 0.
    bm1_log = bm1_viz = bm1_chk = ""

𝜅 = COMM.bcast(𝜅)
𝜌 = COMM.bcast(𝜌)
𝛼 = COMM.bcast(𝛼)
𝛽 = COMM.bcast(𝛽)
𝜁 = COMM.bcast(𝜁)
𝑀 = COMM.bcast(𝑀)
𝜀 = COMM.bcast(𝜀)

𝑊 = COMM.bcast(𝑊)
𝑁 = COMM.bcast(𝑁)
Δ𝑡= COMM.bcast(Δ𝑡)
𝜃 = COMM.bcast(𝜃)
𝑇 = COMM.bcast(𝑇)

bm1_log = COMM.bcast(bm1_log)
bm1_viz = COMM.bcast(bm1_viz)
bm1_chk = COMM.bcast(bm1_chk)

resuming = path.exists(bm1_chk)

Δ0 = Δ𝑡   # initial timestep
Δτ = 0  # runtime offset for resumed simulation

set_log_level(LogLevel.ERROR)

viz_file = XDMFFile(COMM, bm1_viz)
viz_file.parameters["flush_output"] = True
viz_file.parameters["rewrite_function_mesh"] = False
viz_file.parameters["functions_share_mesh"] = True


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


def print0(s):
    if rank == 0:
        print(s)


def adapt_timestep(𝑡, Δ𝑡, its):
    dt_max = 10.0
    growth = 1.2
    decay = 0.5
    dt = Δ𝑡

    if (its < 3):
        dt = min(dt_max, growth * dt)
        print0("  𝑡 = {}: Δ𝑡 = {:.4f} ⤴ {:.4f}".format(𝑡, Δ𝑡, dt))

    if (its > 5):
        dt = max(Δ0, decay * dt)
        print0("  𝑡 = {}: Δ𝑡 = {:.4f} ⤵ {:.4f}".format(𝑡, Δ𝑡, dt))

    return dt


def timestep(t, dt0):
    # Interpolate exp [0, 1] onto timestep [Δ0, 8]
    x0 = 1.0
    x1 = np.exp(1)

    y0 = Δ0
    y1 = 8.0

    m = (y1 - y0) / (x1 - x0)

    x = np.exp(t / 1e6)
    y = y0 + m * (x - x0)

    # Alias timestep to multiples of Δ0, and cap at 1.0
    dt = min(1.0, Δ0 * np.floor(y / Δ0))
    changed = (not np.isclose(dt, dt0))

    if changed:
        print0("  𝑡 = {}: Δ𝑡 ⤴ {}".format(𝑡, Δ𝑡))
    return dt


def crunch_the_numbers(𝛀, 𝑡, 𝑐, 𝐹, 𝜇, 𝜆, i, 𝜈, τ):
    𝑛 = len(𝛀.coordinates())
    𝐦 = assemble(𝑐 * Δ𝑥) / 𝑊**2
    𝐅 = assemble(𝐹 * Δ𝑥)
    𝛈 = assemble(np.abs(𝜇 - 𝜆) / 𝑛 * Δ𝑥)
    𝐢 = COMM.allreduce(i, op=MPI.MAX)
    𝛎 = COMM.allreduce(𝜈, op=MPI.MIN)
    𝛕 = MPI.Wtime() - τ + Δτ

    pid = getpid()
    status = open("/proc/%d/status" % pid).read()

    mem = int(status.split("VmSize:")[1].split("kB")[0]) / 1024.0

    return (𝑡, 𝐦, 𝐅, 𝛈, 𝐢, 𝛎, 𝛕, mem)


def guesstimate(rate, t_now, t_nxt):
    est_nxt = timedelta(seconds=int((t_nxt - t_now) / (Δ𝑡 * rate)))
    est_all = timedelta(seconds=int((𝑇 - t_now) / (Δ𝑡 * rate)))
    return (est_nxt, est_all)


def write_csv_header(filename):
    if rank == 0:
        with open(filename, mode="w") as nrg_file:
            header = [
                "time", "composition", "free_energy", "driving_force",
                "its", "sim_rate", "runtime", "memory"
            ]

            io = csv.writer(nrg_file)
            io.writerow(header)


def write_csv(filename, summary):
    if rank == 0:
        with open(filename, mode="a") as nrg_file:
            io = csv.writer(nrg_file)
            io.writerow(summary)


def runtime_offset(filename):
    rto = 0.0
    if rank == 0:
        with open(filename, mode="r") as nrg_file:
            try:
                io = csv.reader(nrg_file)
                for row in io:
                    _, _, _, _, _, _, rto, _ = row
            except IOError as e:
                MPI.Abort(e)
    rto = COMM.bcast(float(rto))
    return rto


def write_viz(viz_file, u, t=0):
    for n, field in enumerate(u.split()):
        field.rename(field_names[n], field_names[n])
        viz_file.write(field, t)

    viz_file.close()


# Define domain and finite element
#   RectangleMesh.create([Point(0, 0), Point(Nx, Ny)], [nx, ny], cell_type, cell_diag)
𝛀 = RectangleMesh.create(COMM, [Point([0, 0]), Point([𝑊, 𝑊])], [𝑁, 𝑁], CellType.Type.triangle, "crossed")
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
field_names = ("𝑐", "𝜇")

# === Weak Form ===

𝐹, 𝐿 = weak_form(𝒖, 𝒐, 𝕊, 𝛀, 𝓟)
𝑱 = derivative(𝐿, 𝒖, d𝒖)
problem = CahnHilliardEquation(𝑱, 𝐿)

if mpi_root:
    sigjob.doc.weak_form = "{}".format(𝐹).replace("var0(f_7[0])", "𝜑")

# === Dolfin Parameters ===

parameters["linear_algebra_backend"] = "PETSc"
parameters["form_compiler"]["optimize"] = True
parameters["form_compiler"]["cpp_optimize"] = True
parameters["form_compiler"]["representation"] = "uflacs"
parameters["form_compiler"]["quadrature_degree"] = quad_deg

# === Initial Conditions ===

𝑡 = 0.0  # simulation time

if not resuming:
    viz_file.write(𝛀)
    𝒊 = InitialConditions(degree=poly_deg)
    LagrangeInterpolator.interpolate(𝒖, 𝒊)
    LagrangeInterpolator.interpolate(𝒐, 𝒊)

    write_viz(viz_file, 𝒖)
else:
    if resuming:
        print0("Resuming simulation from {}:".format(bm1_chk))
    Δτ = runtime_offset(bm1_log)
    with HDF5File(COMM, bm1_chk, "r") as chk:
        chk.read(𝒖, "/field")

        attr = chk.attributes("/field")
        𝑡 = attr["time"]
        Δ𝑡 = attr["timestep"]

    print0("  𝑡 = {} and Δ𝑡 = {}".format(𝑡, Δ𝑡))


# Enqueue output timestamps
io_q = queue.Queue()

for t_out in (1, 2, 5):
    if 𝑡 < t_out:
        io_q.put(int(t_out))
for n in np.arange(1, 7):
    step = min(int(10**n), 1000)
    for t_out in np.arange(10**n, 10 * 10**n, step):
        if 𝑡 < t_out and t_out <= 𝑇:
            io_q.put(int(t_out))

io_t = io_q.get()

# === TIMESTEPPING ===

start = MPI.Wtime()

# Guess initial rate based on 4-core CPU
rate = 0.5 * (400. / 𝑁)**2 * (COMM.Get_size() / 4)

if not resuming:
    write_csv_header(bm1_log)
    write_csv(bm1_log,
              crunch_the_numbers(𝛀, 𝑡, 𝑐, 𝐹, 𝜇, 𝜆, 0, rate, start))

print0("[{}] Timestepping {}.".format(
    timedelta(seconds=int((MPI.Wtime() - epoch))),
    "resumed" if resuming else "started"))

est_t, all_t = guesstimate(rate, 𝑡, io_t)
print0("[{}] ETA: 𝑡={} in {}, 𝑡={} in {}".format(
    timedelta(seconds=int((MPI.Wtime() - epoch))),
    io_t, est_t, 𝑇, all_t))

nits = 0
adapt_t = 10
itime = MPI.Wtime()

# Main time-stepping loop
while (𝑡 < 𝑇):
    𝒐.assign(𝒖)
    its, converged = solver.solve(problem, 𝒖.vector())

    𝑡 += Δ𝑡
    nits += 1

    if not converged:
        MPI.Abort("Failed to converge!")

    if np.isclose(𝑡, io_t) or 𝑡 > io_t:
        # write free energy summary
        rate = float(nits) / (MPI.Wtime() - itime)
        write_csv(bm1_log,
                  crunch_the_numbers(𝛀, 𝑡, 𝑐, 𝐹, 𝜇, 𝜆, its, rate, start))
        # write visualization slice
        write_viz(viz_file, 𝒖, 𝑡)
        # write checkpoint
        with HDF5File(COMM, bm1_chk, "w") as chk:
            chk.write(𝒖, "/field")

            attr = chk.attributes("/field")
            attr["time"] = 𝑡
            attr["timestep"] = Δ𝑡
            attr["runtime"] = MPI.Wtime() - start

        if not io_q.empty():
            io_t = io_q.get()
            est_t, all_t = guesstimate(rate, 𝑡, io_t)
            print0("[{}] ETA: 𝑡={} in {}, 𝑡={} in {}".format(
                timedelta(seconds=int((MPI.Wtime() - epoch))),
                io_t, est_t, 𝑇, all_t))

        nits = 0
        itime = MPI.Wtime()

    if np.isclose(𝑡, adapt_t) or 𝑡 > adapt_t:
        Δ𝑡 = timestep(𝑡, Δ𝑡)
        adapt_t += 100

    gc.collect()

viz_file.close()
print0("[{}] Simulation complete.".format(
    timedelta(seconds=int((MPI.Wtime() - epoch)))))

# === Plot Energy vs. Sim Time ===

if mpi_root:
    data = pd.read_csv(bm1_log)

    plt.figure(figsize=(10,8))
    plt.title("FEniCS BM1b Free Energy")
    plt.xlabel("Time (a.u.)")
    plt.ylabel(u"Energy Density (J/m³)")

    plt.semilogx(data["time"], data["free_energy"])
    plt.savefig(sigjob.fn("bm1b-energy.png"), dpi=400, bbox_inches="tight")
    plt.close()
