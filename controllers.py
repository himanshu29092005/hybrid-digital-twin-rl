import numpy as np
from env import TankEnv

class PID:
    def __init__(self, kp=0.25, ki=0.004, kd=0.0):
        self.kp, self.ki, self.kd = kp, ki, kd; self.i = 0.0; self.prev = None
    def act(self, T, sp):
        e = sp - T
        d = 0.0 if self.prev is None else e - self.prev
        u_raw = self.kp * e + self.ki * self.i + self.kd * d
        u = float(np.clip(u_raw, 0, 1))
        if u == u_raw: self.i += e                       # anti-windup: integrate only when unsaturated
        self.prev = e; return u

def evaluate(policy, sp=50.0, horizon=200, step_fn=None):
    """policy(obs, env) -> u. Returns time series + IAE + energy."""
    env = TankEnv(step_fn, horizon); obs, _ = env.reset(options={"sp": sp})
    Ts, us, done = [], [], False
    while not done:
        u = policy(obs, env); obs, _, term, trunc, _ = env.step(np.array([u]))
        Ts.append(env.T); us.append(u); done = term or trunc
    Ts, us = np.array(Ts), np.array(us)
    return {"T": Ts, "u": us, "IAE": float(np.mean(np.abs(sp - Ts))), "energy": float(us.sum())}

def pid_policy():
    pid = PID(); return lambda obs, env: pid.act(env.T, env.sp)

def rl_policy(model):
    return lambda obs, env: float(np.clip(model.predict(obs, deterministic=True)[0][0], 0, 1))
