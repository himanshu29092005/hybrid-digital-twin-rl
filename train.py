"""python train.py  -> fits hybrid twin, trains PPO on the TWIN, saves models, evaluates on REAL plant."""
import joblib, numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from twin import collect_data, HybridTwin, fidelity
from env import TankEnv
from controllers import evaluate, pid_policy, rl_policy

data = collect_data(); twin = HybridTwin().fit(data); joblib.dump(twin, "twin.joblib")
f = fidelity(twin); print(f"Open-loop RMSE  physics: {f['rmse_physics']:.3f}  hybrid: {f['rmse_hybrid']:.3f}")

env = make_vec_env(lambda: TankEnv(twin.step), n_envs=4)      # train in the digital twin (safe & cheap)
model = PPO("MlpPolicy", env, learning_rate=3e-4, n_steps=512, batch_size=256, verbose=0, seed=0)
model.learn(total_timesteps=150_000); model.save("ppo_tank")

for sp in (40, 50, 58):                                        # deploy on the real plant
    p, r = evaluate(pid_policy(), sp), evaluate(rl_policy(model), sp)
    print(f"sp={sp}  PID IAE={p['IAE']:.2f} E={p['energy']:.0f} | PPO IAE={r['IAE']:.2f} E={r['energy']:.0f}")
