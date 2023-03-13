import numpy as np
from scipy.integrate import RK45

def van_der_pol_oscillator(params):
    def system(t, y):
        h = np.empty_like(y)
        h[0] = y[1]
        h[1] = params[0] * np.sin(params[2] * t) - params[1] * (y[0] ** 2 - 1) * y[1] - y[0]
        return h
    
    return system

def forced_conservative_nonlinear_oscillator05(params):
    def system(t, y):
        h = np.empty_like(y)
        h[0] = y[1]
        h[1] = np.sin(2*t) - np.power(y[0], 7)
        return h
    
    return system

def forced_van_der_pol_oscillator(params):
    def system(t, y):
        h = np.empty_like(y)
        h[0] = y[1]
        h[1] = 5 * np.sin(0.8 * t) - 4 * y[1] * np.cos(y[0]) - 2 * y[0]
        return h
    
    return system

def build_system(system, params=[], state=[], t0=0, t_bound=1e+5, max_step=1e-2):
    return RK45(system(params), t0, state, t_bound=t_bound, max_step=max_step)