# Hybrid Digital Twin (Physics + ML) with RL Controller
A water-tank heater is modelled with a **first-principles energy balance**; a **gradient-boosting residual model** learns
what the physics misses (efficiency loss, nonlinear losses) from plant data -> **hybrid digital twin**.
A **PPO** agent (Stable-Baselines3, Gymnasium env) is trained *inside the twin* and deployed on the "real" plant,
benchmarked against a tuned **PID**. **Streamlit** dashboard + governance (model card, residual drift monitor).

## Run
```bash
pip install -r requirements.txt
python train.py         # fits twin, trains PPO (~3-5 min CPU), prints PID vs PPO table
streamlit run app.py
```
## Architecture
`plant.py` (physics + real plant) -> `twin.py` (residual ML) -> `env.py` (Gym) -> `train.py` (PPO) -> `app.py` (dashboard)

## Results
Paste output of `train.py` here (open-loop RMSE physics vs hybrid; IAE/energy PID vs PPO).

## Extending
Swap the plant for a Simulink/Amesim model or PyBaMM battery model; add MPC baseline; add sensor-fault scenarios.
