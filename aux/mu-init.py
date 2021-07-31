"""
Ask SymPy to generate the initial condition for 𝜇 using the synthetic
initial condition for 𝑐 to improve convergence of the first few steps.
"""

from sympy import Symbol, cos, diff, init_printing, pprint, simplify, symbols
from sympy.abc import alpha, beta, epsilon, kappa, mu, rho
from sympy.physics.vector import ReferenceFrame, divergence, gradient
init_printing()

R = ReferenceFrame("R")
c, x, y = symbols("𝑐 𝑥 𝑦")
alpha, beta, epsilon, kappa, rho, zeta = symbols("𝛼 𝛽 𝜀 𝜅 𝜌 𝜁")

F = rho * (c - alpha)**2 * (beta - c)**2
f = diff(F, c)

pprint("{} = {}".format(Symbol("F"), F))
pprint("{} = {}".format(Symbol("f"), f))

c0 = zeta + epsilon * (
    cos(0.105 * R[0]) * cos(0.110 * R[1])
    + (cos(0.130 * R[0]) * cos(0.087 * R[1]))**2
    + cos(0.025 * R[0] - 0.150 * R[1]) * cos(0.070 * R[0] - 0.020 * R[1])
)

pprint("{} = {}".format(Symbol("c"), c0.subs({R[0]: x, R[1]: y})))

u0 = f.subs(c, c0) - kappa * divergence(gradient(c0, R), R)
pprint("{} = {}".format(Symbol("u"), simplify(u0.subs({R[0]: x, R[1]: y}))))
