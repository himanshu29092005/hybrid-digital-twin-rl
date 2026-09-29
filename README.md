# Hybrid Digital Twin (Physics + ML) with RL Controller

> A physics + machine-learning digital twin of a water-tank heater, used as a simulator to train a PPO reinforcement-learning controller that is benchmarked against a tuned PID.

![Python](https://img.shields.io/badge/python-3.10+-blue)
![Stable-Baselines3](https://img.shields.io/badge/RL-Stable--Baselines3-green)
![Streamlit](https://img.shields.io/badge/dashboard-Streamlit-red)

## Overview
Pure physics models miss real-world effects such as efficiency loss and nonlinear heat losses. Pure ML models need lots of data and don't respect the physics. This project combines both:

1. **Physics model**: a first-principles energy balance of a heated water tank.
2. **Residual ML model**: a gradient-boosting model learns what the physics misses from plant data, giving a **hybrid digital twin**.
3. **RL controller**: a PPO agent (Stable-Baselines3, Gymnasium env) is trained *entirely inside the twin*, then deployed on the "real" plant.
4. **Benchmark**: PPO is compared with a tuned PID controller on tracking error (IAE) and energy use.
5. **Dashboard and governance**: a Streamlit app for PID vs PPO comparison, plus a model card and a residual drift monitor.

## Key Results
| Metric | Result |
|--------|--------|
| Open-loop RMSE, physics-only | 11.248 °C |
| Open-loop RMSE, hybrid twin | **0.146 °C** (~98.7% lower) |
| PPO vs PID tracking (IAE) | 9.3% better at 50 °C, 7.3% better at 58 °C, 5.8% worse at 40 °C |
| Energy use | Comparable to PID (within ~4%) |

### Digital twin accuracy (open-loop RMSE)
| Model | RMSE (°C) |
|-------|-----------|
| Physics-only | 11.248 |
| Hybrid (physics + ML residual) | **0.146** |

### Controller benchmark (PID vs PPO trained inside the twin)
| Setpoint | PID IAE | PPO IAE | PID energy | PPO energy |
|----------|---------|---------|------------|------------|
| 40 °C | **1.55** | 1.64 | 70 | **69** |
| 50 °C | 3.66 | **3.32** | **113** | 115 |
| 58 °C | 6.20 | **5.75** | **147** | 152 |

*IAE = Integral of Absolute Error (lower is better). Bold = better.*

**Takeaway:** PPO, trained only on the twin, transfers to the real plant and performs on par with a tuned PID. It tracks better at higher setpoints, is slightly worse at 40 °C, and uses similar energy overall.

## Architecture
```
plant.py  ->  twin.py  ->  env.py  ->  train.py  ->  app.py
(physics +    (residual    (Gymnasium  (PPO          (Streamlit
 real plant)   ML)          env)        training)     dashboard)
```

| File | Purpose |
|------|---------|
| `plant.py` | Physics model and the simulated "real" plant |
| `twin.py` | Gradient-boosting residual model, hybrid twin |
| `env.py` | Gymnasium environment built on the twin |
| `controllers.py` | PID baseline controller |
| `train.py` | Fits the twin, trains PPO, prints the benchmark |
| `governance.py` | Model card and residual drift monitor |
| `app.py` | Streamlit dashboard |

## Quick Start
```bash
pip install -r requirements.txt
python train.py          # fits twin, trains PPO (~3-5 min on CPU), prints results
streamlit run app.py     # opens the dashboard
```

## Screenshots
<!-- Add dashboard screenshot here -->
<!-- ![Dashboard](screenshots/dashboard.png) -->

## Tech Stack
Python, NumPy, scikit-learn, Stable-Baselines3, Gymnasium, PyTorch, Streamlit

## Roadmap
- Swap the plant for a Simulink / Amesim model or a PyBaMM battery model
- Add an MPC baseline
- Add sensor-fault scenarios
