import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from plant import phys_step, real_step, ambient

def collect_data(n=6000, seed=0):
    """Excite the real plant with random hold-and-switch heater inputs (system identification data)."""
    rng = np.random.default_rng(seed)
    T, rows, u = 20.0, [], 0.5
    for t in range(n):
        if t % 20 == 0: u = rng.uniform(0, 1)
        Ta = ambient(t)
        Tn = real_step(T, u, Ta, rng)
        rows.append((T, u, Ta, Tn)); T = Tn
        if T > 90: T = 25.0
    return np.array(rows)

class HybridTwin:
    """T_next = physics(T,u,Tamb) + ML_residual(T,u,Tamb)."""
    def __init__(self): self.res = GradientBoostingRegressor(n_estimators=200, max_depth=3, random_state=0)
    def fit(self, data):
        T, u, Ta, Tn = data.T
        self.res.fit(np.c_[T, u, Ta], Tn - phys_step(T, u, Ta)); return self
    def step(self, T, u, Ta):
        return phys_step(T, u, Ta) + float(self.res.predict([[T, u, Ta]])[0])

def rollout(step_fn, u_seq, T0=20.0):
    T, out = T0, []
    for t, u in enumerate(u_seq):
        T = step_fn(T, u, ambient(t)); out.append(T)
    return np.array(out)

def fidelity(twin, seed=1, n=400):
    """Open-loop rollout error vs the real plant on unseen inputs."""
    rng = np.random.default_rng(seed)
    u = np.repeat(rng.uniform(0, 1, n // 20), 20)
    real = rollout(lambda T, u, Ta: real_step(T, u, Ta), u)
    phys = rollout(phys_step, u)
    hyb = rollout(twin.step, u)
    rm = lambda a: float(np.sqrt(np.mean((a - real) ** 2)))
    return {"real": real, "physics": phys, "hybrid": hyb, "rmse_physics": rm(phys), "rmse_hybrid": rm(hyb)}
