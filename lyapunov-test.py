from scipy.integrate import RK45
import numpy as np
import matplotlib.pyplot as plt


def build_vanderpol_oscillator(a=1, b=1, o=0.45, x0=1, v0=1, t0=0, t_bound=1e+5):
    def vanderpol(t, y):
        ydot = np.empty_like(y)
        ydot[0] = y[1]
        ydot[1] = a * np.sin(o * t) - b * (y[0] ** 2 - 1) * y[1] - y[0]
        return ydot

    return RK45(vanderpol, t0, [x0, v0], t_bound=t_bound, max_step=1)

def build_fc_oscillator05(x0=0, v0=0, t0=0, t_bound=1e+5):
    d = None
    def oscillator(t, y):
        ydot = np.empty_like(y) if d is None else d 
        ydot[0] = y[1]
        ydot[1] = np.sin(2 * t) - np.power(y[0], 7)
        return ydot

    return RK45(oscillator, t0, [x0, v0], t_bound=t_bound, max_step=1e-2)

vs = []
xs = []
aa = []

solver = build_fc_oscillator05()

for i in range(1000):
    solver.step()

xb0 = solver.y + np.random.randn(*solver.y.shape) * 1e-14
solver2 = build_fc_oscillator05(xb0[0], xb0[1], solver.t)
d0 = np.sqrt((solver.y - xb0) ** 2)

lsum = 0

timesteps = 10000
for i in range(timesteps):
    solver.step()
    solver2.step()

    d1 = np.sqrt((solver.y - solver2.y) ** 2)
    xbi = solver2.y
    xb0 = solver.y + d0 * (solver2.y - solver.y) / d1
    solver2.y = xb0
    lsum = lsum + np.log(d1 / d0)

largest = np.max(lsum / timesteps)
print(largest)