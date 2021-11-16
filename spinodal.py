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

* Mesh resolution (reciprocal)
* Stability criterion (timestep)
* Polynomial degree
* Relative error
"""

import csv
import gc
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import queue
import traceback

from datetime import timedelta
from flow import FlowProject
from mpi4py import MPI
from matplotlib import style
from os import getpid, path
from petsc4py import PETSc
from sys import argv

from dolfin import FiniteElement, FunctionSpace, MixedElement
from dolfin import Function, TestFunctions, TrialFunction
from dolfin import LagrangeInterpolator, NonlinearProblem
from dolfin import NewtonSolver, PETScSNESSolver
from dolfin import CellType, Mesh, Point, RectangleMesh, UserExpression
from dolfin import HDF5File, XDMFFile
from dolfin import LogLevel, set_log_level
from dolfin import cos, derivative, grad, inner, sin, variable
from dolfin import assemble, parameters, split
from dolfin import dx as Δ𝑥
from ufl import replace

# === Parallel environment ===
epoch = MPI.Wtime()
COMM = MPI.COMM_WORLD
rank = COMM.Get_rank()
mpi_root = (rank == 0)

style.use("seaborn")

# === Benchmark 1 Parameters ===
𝑊 = 200  # domain width & length
𝜅 = 2    # gradient energy coefficient
𝜌 = 5    # well height
𝛼 = 0.3  # eqm composition of phase 1
𝛽 = 0.7  # eqm composition of phase 2
𝜁 = 0.5  # system composition
𝑀 = 5    # interface mobility
𝜀 = 0.01 # noise amplitude
𝑇 = 5e4  # simulation target

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
    𝛕 = MPI.Wtime() - τ + Δτ

    pid = getpid()
    status = open("/proc/%d/status" % pid).read()

    mem = int(status.split("VmSize:")[1].split("kB")[0]) / 1024.0

    return (𝑡, 𝐦, 𝐅, 𝛈, 𝐢, 𝛎, 𝛕, mem)


def guesstimate(rate, t_now, t_nxt):
    est_nxt = timedelta(seconds=int((t_nxt - t_now) / (Δ𝑡 * rate)))
    est_all = timedelta(seconds=int((𝑇 - t_now) / (Δ𝑡 * rate)))
    return (est_nxt, est_all)


def print0(filename, io_str):
    if mpi_root:
        with open(filename, mode="a") as fh:
            print(io_str, file=fh)


def runtime_offset(filename):
    rto = 0.0
    if mpi_root:
        with open(filename, mode="r") as nrg_file:
            try:
                io = csv.reader(nrg_file)
                for row in io:
                    _, _, _, _, _, _, rto, _ = row
            except IOError as e:
                MPI.Abort(e)
    rto = COMM.bcast(float(rto))
    return rto


def set_solver(solverType):
    if solverType == "newton":
        solver = NewtonSolver()
        prm = solver.parameters
        prm["linear_solver"] = "gmres"
        prm["convergence_criterion"] = "incremental"
    elif solverType == "snes":
        solver = PETScSNESSolver()
        prm = solver.parameters
        prm["linear_solver"] = "gmres"
        prm["method"] = "newtonls"
    else:
        raise ValueError("Unrecognized solver type {}".format(solverType))

    return solver


def timestep(t, dt0):
    # Interpolate exp [1, e] onto timestep [Δ0, 8]
    x1 = np.exp(1)

    y0 = Δ0
    y1 = 8.0

    m = (y1 - y0) / (x1 - 1)

    x = np.exp(t / 1e6)
    y = y0 + m * (x - 1)

    # Alias timestep to multiples of Δ0, and cap at 1.0
    dt = min(1.0, Δ0 * np.floor(y / Δ0))
    changed = (not np.isclose(dt, dt0))

    if changed:
        print0(log_file, "  𝑡 = {}: Δ𝑡 ⤴ {}".format(𝑡, Δ𝑡))
    return dt


def weak_form(𝒖, 𝒐, ℝ, 𝛀, 𝐸, 𝜃, d𝒖):
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
    𝑱 = derivative(𝐿, 𝒖, d𝒖)

    return 𝐹, 𝐿, 𝑱


def write_csv_header(filename):
    if mpi_root:
        with open(filename, mode="w") as nrg_file:
            header = [
                "time", "composition", "free_energy", "driving_force",
                "its", "sim_rate", "runtime", "memory"
            ]

            io = csv.writer(nrg_file)
            io.writerow(header)


def write_csv(filename, summary):
    if mpi_root:
        with open(filename, mode="a") as nrg_file:
            io = csv.writer(nrg_file)
            io.writerow(summary)
    return summary


def write_viz(viz_file, u, t=0):
    for n, field in enumerate(u.split()):
        field.rename(field_names[n], field_names[n])
        viz_file.write(field, t)


@FlowProject.label
def victory(job):
    if "simtime" in job.doc.keys():
        return (job.doc.simtime >= 𝑇)
    elif "error" in job.doc.keys():
        return True
    else:
        return False

@FlowProject.operation
@FlowProject.post(victory)
def spinodal(job):
    try:
        # Discretization parameters
        𝑁 = job.sp.cells
        𝜃 = job.sp.crank
        Δ0 = job.sp.mindt # initial timestep
        resuming = ("simtime" in job.doc.keys())
        set_log_level(LogLevel.ERROR)

        chk_name = job.fn("checkpoint.h5")
        csv_name = job.fn("bm1b.csv")
        log_name = job.fn("bm1b.log")
        png_name = job.fn("bm1b.png")
        viz_name = job.fn("bm1b.xdmf")

        𝛀 = RectangleMesh.create(COMM, [Point(0, 0), Point(𝑊, 𝑊)],
                                 [𝑁, 𝑁], CellType.Type.triangle, job.sp.diag)
        𝓟 = FiniteElement("Lagrange", 𝛀.ufl_cell(), job.sp.poly)
        𝐸 = MixedElement([𝓟, 𝓟])

        # Big important data file
        viz_file = XDMFFile(𝛀.mpi_comm(), viz_name)
        Δτ = 0  # runtime offset for resumed simulation
        𝑡  = 0  # simulation time

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
        𝐹, 𝐿, 𝑱 = weak_form(𝒖, 𝒐, 𝕊, 𝛀, 𝓟, 𝜃, d𝒖)
        problem = CahnHilliardEquation(𝑱, 𝐿)

        if mpi_root:
            job.doc.weak_form = "{}".format(𝐹).replace("var0(f_7[0])", "𝜑")

        # === Dolfin Parameters ===
        parameters["linear_algebra_backend"] = "PETSc"
        parameters["form_compiler"]["representation"] = "uflacs"
        parameters["form_compiler"]["quadrature_degree"] = job.sp.quad

        solver = set_solver(job.sp.solver)

        # === Initial Conditions ===

        if not resuming:
            𝒊 = InitialConditions(degree=job.sp.poly)
            LagrangeInterpolator.interpolate(𝒖, 𝒊)
            LagrangeInterpolator.interpolate(𝒐, 𝒊)

            viz_file.parameters["flush_output"] = True
            viz_file.parameters["rewrite_function_mesh"] = False
            viz_file.parameters["functions_share_mesh"] = True

            viz_file.write(𝛀)
            write_viz(viz_file, 𝒖)
        else:
            Δτ = runtime_offset(csv_name)
            𝑡 = job.doc.simtime
            Δ𝑡 = job.doc.timestep
            print0(log_file, "Resuming simulation from 𝑡 = {} with Δ𝑡 = {}".format(𝑡, Δ𝑡))
            with HDF5File(𝛀.mpi_comm(), chk_name, "r") as chk:
                chk.read(𝒖, "/field")

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
            write_csv_header(csv_name)
            write_csv(csv_name,
                      crunch_the_numbers(𝛀, 𝑡, 𝑐, 𝐹, 𝜇, 𝜆, 0, rate, start))

        print0(log_file, "[{}] Timestepping {}.".format(
            timedelta(seconds=int((MPI.Wtime() - epoch))),
            "resumed" if resuming else "started"))

        est_t, all_t = guesstimate(rate, 𝑡, io_t)
        print0(log_file, "[{}] ETA: 𝑡={} in {}, 𝑡={} in {}".format(
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
                summary = write_csv(csv_name,
                                    crunch_the_numbers(𝛀, 𝑡, 𝑐, 𝐹, 𝜇, 𝜆, its, rate, start))
                𝐅 = summary[2]
                𝛕 = summary[6]
                # write visualization slice
                write_viz(viz_file, 𝒖, 𝑡)
                # write checkpoint
                with HDF5File(𝛀.mpi_comm(), chk_name, "w") as chk:
                    chk.write(𝒖, "/field")
                if mpi_root:
                    job.doc.simtime = 𝑡
                    job.doc.free_energy = 𝐅
                    job.doc.timestep = Δ𝑡
                    job.doc.runtime = 𝛕

                    # plot Energy vs. Sim Time ===
                    data = pd.read_csv(csv_name)
                    plt.figure(figsize=(10,8))
                    plt.xlabel("Time (a.u.)")
                    plt.ylabel(u"Free Energy")
                    plt.semilogx(data["time"], data["free_energy"])
                    a, b = plt.xlim()
                    plt.xlim([a, np.ceil(np.log10(𝑇))])
                    plt.savefig(png_name, dpi=400, bbox_inches="tight")
                    plt.close()
                if not io_q.empty():
                    io_t = io_q.get()
                    est_t, all_t = guesstimate(rate, 𝑡, io_t)
                    print0(log_file, "[{}] ETA: 𝑡={} in {}, 𝑡={} in {}".format(
                        timedelta(seconds=int((MPI.Wtime() - epoch))),
                        io_t, est_t, 𝑇, all_t))

                nits = 0
                itime = MPI.Wtime()

            if 𝑡 >= adapt_t:
                Δ𝑡 = timestep(𝑡, Δ𝑡)
                adapt_t += 10

            gc.collect()

        print0(log_file, "[{}] Simulation complete.".format(
            timedelta(seconds=int((MPI.Wtime() - epoch)))))
    except:
        if mpi_root:
            job.doc.traceback = traceback.format_exc()
            print(job.doc.traceback)

        print0(log_file, "[{}] Simulation failure.".format(
            timedelta(seconds=int((MPI.Wtime() - epoch)))))
    finally:
        viz_file.close()

if __name__ == "__main__":
    FlowProject().main()
