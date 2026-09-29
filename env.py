import numpy as np, gymnasium as gym
from gymnasium import spaces
from plant import real_step, phys_step, ambient

class TankEnv(gym.Env):
    """Track a temperature setpoint. `step_fn(T,u,Tamb)` = the dynamics (real plant, physics or hybrid twin)."""
    def __init__(self, step_fn=None, horizon=200):
        self.step_fn = step_fn or (lambda T, u, Ta: real_step(T, u, Ta))
        self.horizon = horizon
        self.action_space = spaces.Box(0.0, 1.0, (1,), np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, (4,), np.float32)

    def _obs(self):
        return np.array([(self.T - 40) / 20, (self.sp - 40) / 20, (ambient(self.t) - 20) / 5,
                         (self.sp - self.T) / 20], np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.t, self.T = 0, ambient(0)
        self.sp = float(self.np_random.uniform(35, 60)) if not options or "sp" not in options else options["sp"]
        return self._obs(), {}

    def step(self, a):
        u = float(np.clip(a[0], 0, 1))
        self.T = self.step_fn(self.T, u, ambient(self.t)); self.t += 1
        r = -abs(self.sp - self.T) / 10 - 0.02 * u          # tracking error + energy cost
        return self._obs(), r, False, self.t >= self.horizon, {}
