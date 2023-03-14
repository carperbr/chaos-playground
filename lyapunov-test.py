import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from systems import van_der_pol_oscillator, forced_conservative_nonlinear_oscillator05, forced_van_der_pol_oscillator, build_system

def calculate_lyapunov_exponent(params, state, t0, system):
    solver = build_system(system, params, state, t0)

    dt0 = 0
    while dt0 < 1000:
        solver.step()
        dt0 = dt0 + solver.step_size

    pstate = np.copy(solver.y)
    pstate[0] = pstate[0] + np.random.randn() * 1e-12
    solver2 = build_system(system, params, pstate, solver.t)
    lsum = 0

    dt = 0
    dt2 = 0
    timesteps = 100000
    for i in tqdm(range(timesteps)):
        solver.step()
        solver2.step()

        v = solver2.y - solver.y
        d = np.linalg.norm(v)
        rs = 1 / np.sqrt(d)
        solver2.y = solver.y + rs * (solver2.y - solver.y)
        lsum = lsum + np.log(d)

        r = (np.cos((v[0]) * np.pi) + 1) / 2.0
        g = (np.cos((v[1]) * np.pi) + 1) / 2.0
        b = (np.sin((v[0]) * np.pi) + 1) / 4.0 + (np.sin((v[1]) * np.pi) + 1) / 4.0

        plt.plot([solver.y[0], solver.y_old[0]], [solver.y[1], solver.y_old[1]], linewidth=0.25, color=(r,g,b), alpha=0.2)

        dt = dt + solver.step_size
        dt2 = dt2 + solver2.step_size

    l = lsum / dt
    plt.savefig(f'tmp{np.random.randn()}.png', dpi=512)
    
    return np.max(l)

print(calculate_lyapunov_exponent([], [0, 0], 0, forced_conservative_nonlinear_oscillator05))