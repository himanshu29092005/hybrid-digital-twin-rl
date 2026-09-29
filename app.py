import os, joblib, numpy as np, pandas as pd, streamlit as st
from twin import collect_data, HybridTwin, fidelity
from controllers import evaluate, pid_policy, rl_policy
from governance import model_card, residual_monitor

st.set_page_config(page_title="Hybrid Digital Twin + RL", layout="wide")
st.title("Hybrid Digital Twin (Physics + ML) with RL Controller")

@st.cache_resource
def get_twin():
    if os.path.exists("twin.joblib"): return joblib.load("twin.joblib")
    return HybridTwin().fit(collect_data())

twin = get_twin(); f = fidelity(twin)
t1, t2, t3 = st.tabs(["Twin fidelity", "PID vs RL", "Governance"])
with t1:
    c1, c2 = st.columns(2)
    c1.metric("Physics-only RMSE (C)", f"{f['rmse_physics']:.3f}")
    c2.metric("Hybrid RMSE (C)", f"{f['rmse_hybrid']:.3f}", f"{f['rmse_hybrid']-f['rmse_physics']:.3f}")
    st.line_chart(pd.DataFrame({k: f[k] for k in ("real", "physics", "hybrid")}))
with t2:
    sp = st.slider("Setpoint (C)", 35, 60, 50)
    step = st.radio("Run controllers on", ["Real plant", "Hybrid twin"], horizontal=True)
    fn = None if step == "Real plant" else twin.step
    p = evaluate(pid_policy(), sp, step_fn=fn)
    cols = {"PID": p["T"], "Setpoint": np.full(len(p["T"]), sp)}
    res = {"PID": (p["IAE"], p["energy"])}
    if os.path.exists("ppo_tank.zip"):
        from stable_baselines3 import PPO
        r = evaluate(rl_policy(PPO.load("ppo_tank")), sp, step_fn=fn)
        cols["PPO"] = r["T"]; res["PPO"] = (r["IAE"], r["energy"])
    else: st.info("Run `python train.py` to train the PPO agent, then reload.")
    st.line_chart(pd.DataFrame(cols))
    st.dataframe(pd.DataFrame(res, index=["IAE (C)", "Energy (duty-steps)"]).T)
with t3:
    new = collect_data(1500, seed=42)
    rm = residual_monitor(twin, new)
    st.metric("Twin RMSE on fresh data (C)", f"{rm:.3f}", "OK" if rm < 0.5 else "retrain twin")
    card = model_card(f, *(res["PID"][0], res["PPO"][0]) if "PPO" in res else ())
    st.markdown(card); st.download_button("Download model card", card, "MODEL_CARD.md")
