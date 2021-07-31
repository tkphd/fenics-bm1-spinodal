# Plot the projected timestep for an "adaptive" PDE solver

import numpy as np
from matplotlib.pyplot import legend, savefig, xlim, ylim
from matplotlib.pyplot import semilogx as plot

𝑒 = np.exp(1)
dt_lim = (0.125, 8.0)
t_lim = (1e3, 1e6)

def linterp(x, y0=dt_lim[0], y1=dt_lim[1]):
    m = y1 - y0
    return m * x + y0

def expterp(z, y0=dt_lim[0], y1=dt_lim[1]):
    x = np.exp(z)
    m = (y1 - y0) / (𝑒 - 1)
    return y0 + m * (x - 1)

def alias_dt(x, dt0=dt_lim[0]):
    return dt0 * (np.floor(x / dt0))

def alias_exp(x):
    return 2 ** (np.floor(np.log2(x)))

t = np.logspace(3, 6, 401)

u = np.minimum(1, expterp((t / 1e6) ** 0.75))
v = np.minimum(1, expterp((t / 1e6)))
w = np.minimum(1, expterp((t / 1e6) ** 2))

x = alias_dt(u)
y = alias_dt(v)
z = alias_dt(w)

plot(t, u)
plot(t, x, label="1l")

plot(t, v)
plot(t, y, label="1e")

plot(t, w)
plot(t, z, label="2e")

ylim([0, 1])

legend(loc="upper left")
savefig("timestep.png", dpi=400, bbox_inches="tight")
