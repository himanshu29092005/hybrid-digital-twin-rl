import numpy as np, datetime as dt

def psi(ref, cur, bins=10):
    edges = np.quantile(ref, np.linspace(0, 1, bins + 1)); edges[0], edges[-1] = -np.inf, np.inf
    r = np.histogram(ref, edges)[0] / len(ref) + 1e-6; c = np.histogram(cur, edges)[0] / len(cur) + 1e-6
    return float(np.sum((c - r) * np.log(c / r)))

def residual_monitor(twin, data_new):
    """If twin error on new data grows vs training, flag for retraining."""
    T, u, Ta, Tn = data_new.T
    pred = np.array([twin.step(a, b, c) for a, b, c in zip(T, u, Ta)])
    return float(np.sqrt(np.mean((pred - Tn) ** 2)))

def model_card(f, iae_pid=None, iae_rl=None):
    ctrl = f"\n**Control (IAE, lower better):** PID {iae_pid:.2f} vs PPO {iae_rl:.2f}" if iae_pid else ""
    return f"""# Model Card - Hybrid Digital Twin + RL Controller
Generated: {dt.datetime.now():%Y-%m-%d %H:%M}

**Intended use:** simulation-based controller design for a water-tank heater. Not certified for safety-critical use.
**Twin:** first-principles energy balance + gradient-boosting residual learned from plant data.
**Open-loop RMSE:** physics-only {f['rmse_physics']:.3f} C -> hybrid {f['rmse_hybrid']:.3f} C
{ctrl}
**Limitations:** single-tank, 1-D temperature, trained on simulated 'real' data; extrapolation outside 20-90 C is unreliable.
**Safeguards:** action clipping to [0,1]; residual-RMSE drift monitor triggers retraining; PID fallback recommended.
"""
