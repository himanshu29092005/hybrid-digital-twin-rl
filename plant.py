"""Electric water-tank heater. State T (C), input u in [0,1] (heater duty), disturbance Tamb (C).
   'Physics' model = idealised first-principles ODE. 'Real' plant = same physics with unmodelled effects
   (efficiency loss, temperature-dependent losses, sensor noise) - this gap is what the ML residual learns."""
import numpy as np

DT = 30.0                      # s per step
C, K, EFF, PMAX = 6.0e4, 20.0, 0.90, 2000.0   # J/K, W/K, -, W

def phys_step(T, u, Tamb):
    """Energy balance C dT/dt = eff*P*u - k (T - Tamb), explicit Euler."""
    return T + DT * (EFF * PMAX * u - K * (T - Tamb)) / C

def real_step(T, u, Tamb, rng=None):
    eff = 0.80 - 0.0008 * (T - 20)                 # heater efficiency degrades when hot
    k = K * (1 + 0.012 * (T - Tamb))               # nonlinear (convective) losses
    Tn = T + DT * (eff * PMAX * u - k * (T - Tamb)) / C
    return Tn + (rng.normal(0, 0.02) if rng is not None else 0.0)

def ambient(t):
    return 20 + 5 * np.sin(2 * np.pi * t * DT / 86400 * 6)   # fast-ish drift for demo
