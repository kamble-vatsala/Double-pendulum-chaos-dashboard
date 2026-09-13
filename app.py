import streamlit as st
import numpy as np
from scipy.integrate import solve_ivp
import plotly.graph_objects as go
import sqlite3
import pandas as pd
import os
import time as time_module
from dotenv import load_dotenv
from google import genai

load_dotenv()

st.set_page_config(page_title="Double Pendulum Chaos Dashboard", layout="wide")
st.title("🌀 Double Pendulum Chaos Simulation")
st.caption("Numerical ODE Solving | Lyapunov Chaos Analysis | Monte Carlo | ML Surrogate Model")

g = 9.81

def equations_of_motion(t, state, L1, L2, m1, m2):
    theta1, z1, theta2, z2 = state
    delta = theta1 - theta2
    denom1 = L1 * (2*m1 + m2 - m2*np.cos(2*delta))
    denom2 = L2 * (2*m1 + m2 - m2*np.cos(2*delta))
    dtheta1_dt = z1
    dz1_dt = (-g*(2*m1+m2)*np.sin(theta1) - m2*g*np.sin(theta1-2*theta2)
              - 2*np.sin(delta)*m2*(z2**2*L2 + z1**2*L1*np.cos(delta))) / denom1
    dtheta2_dt = z2
    dz2_dt = (2*np.sin(delta)*(z1**2*L1*(m1+m2) + g*(m1+m2)*np.cos(theta1)
              + z2**2*L2*m2*np.cos(delta))) / denom2
    return [dtheta1_dt, dz1_dt, dtheta2_dt, dz2_dt]

def compute_positions(theta1, theta2, L1, L2):
    x1, y1 = L1*np.sin(theta1), -L1*np.cos(theta1)
    x2 = x1 + L2*np.sin(theta2)
    y2 = y1 - L2*np.cos(theta2)
    return x1, y1, x2, y2

# ==================== SIDEBAR: Interactive Controls ====================
st.sidebar.header("Pendulum Parameters")
theta1_deg = st.sidebar.slider("Initial Angle 1 (°)", 0, 180, 90)
theta2_deg = st.sidebar.slider("Initial Angle 2 (°)", 0, 180, 90)
L1 = st.sidebar.slider("Rod 1 Length (m)", 0.5, 2.0, 1.0)
L2 = st.sidebar.slider("Rod 2 Length (m)", 0.5, 2.0, 1.0)
m1 = st.sidebar.slider("Mass 1 (kg)", 0.5, 3.0, 1.0)
m2 = st.sidebar.slider("Mass 2 (kg)", 0.5, 3.0, 1.0)
duration = st.sidebar.slider("Simulation Duration (s)", 5, 20, 10)

# ==================== SECTION 1: Live Animated Simulation ====================
st.subheader("🎬 Live Pendulum Simulation")

theta1_0 = np.radians(theta1_deg)
theta2_0 = np.radians(theta2_deg)
n_frames = 150
t_eval = np.linspace(0, duration, n_frames)

sol = solve_ivp(equations_of_motion, (0, duration), [theta1_0, 0, theta2_0, 0],
                 t_eval=t_eval, args=(L1, L2, m1, m2), method="RK45", rtol=1e-8, atol=1e-8)
x1, y1, x2, y2 = compute_positions(sol.y[0], sol.y[2], L1, L2)

max_reach = L1 + L2 + 0.3

frames = []
for i in range(n_frames):
    frames.append(go.Frame(
        data=[
            go.Scatter(x=[0, x1[i], x2[i]], y=[0, y1[i], y2[i]],
                       mode="lines+markers", line=dict(color="gray", width=2),
                       marker=dict(size=[8, 14, 14], color=["black", "royalblue", "crimson"])),
            go.Scatter(x=x2[:i+1], y=y2[:i+1], mode="lines",
                       line=dict(color="crimson", width=1), opacity=0.4)
        ],
        name=str(i)
    ))

fig_anim = go.Figure(
    data=[
        go.Scatter(x=[0, x1[0], x2[0]], y=[0, y1[0], y2[0]], mode="lines+markers",
                   line=dict(color="gray", width=2), marker=dict(size=[8, 14, 14], color=["black", "royalblue", "crimson"])),
        go.Scatter(x=[x2[0]], y=[y2[0]], mode="lines", line=dict(color="crimson", width=1))
    ],
    frames=frames
)
fig_anim.update_layout(
    xaxis=dict(range=[-max_reach, max_reach], zeroline=False),
    yaxis=dict(range=[-max_reach, max_reach], zeroline=False, scaleanchor="x"),
    height=550, showlegend=False,
    updatemenus=[dict(type="buttons", buttons=[
        dict(label="▶ Play", method="animate", args=[None, {"frame": {"duration": 40, "redraw": True}, "fromcurrent": True}]),
        dict(label="⏸ Pause", method="animate", args=[[None], {"frame": {"duration": 0}, "mode": "immediate"}])
    ])],
    sliders=[dict(steps=[dict(method="animate", args=[[str(i)], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate"}],
                              label="") for i in range(n_frames)], len=1.0)]
)
st.plotly_chart(fig_anim, use_container_width=True)
st.caption("Press ▶ Play to watch the pendulum swing. The red trail shows the path of the second bob.")

# ==================== SECTION 2: Project Summary ====================
st.subheader("Project Findings Summary")
conn = sqlite3.connect("data/pendulum.db")
summary = pd.read_sql("SELECT * FROM project_summary", conn)
conn.close()
st.dataframe(summary, use_container_width=True)

st.info(
    "**Key finding:** This system is provably chaotic (Lyapunov exponent λ ≈ 0.253/s, positive = chaos confirmed), "
    "meaning predictions become meaningless within ~4 seconds even with near-perfect initial measurements. "
    "The ML surrogate model still adds real value: it's 257x faster than numerical simulation for large batches, "
    "and accurate for short-to-medium time horizons - but its error grows with prediction distance, directly "
    "reflecting the same chaos the Lyapunov analysis quantified."
)

# ==================== SECTION 3: Chaos Comparison Tool ====================
st.subheader("🔬 Chaos Demonstration: Compare Two Nearly Identical Starting Angles")
perturbation = st.slider("Perturbation in Angle 1 (degrees)", 0.0001, 1.0, 0.01, format="%.4f")

theta1_0_b = theta1_0 + np.radians(perturbation)
sol_b = solve_ivp(equations_of_motion, (0, duration), [theta1_0_b, 0, theta2_0, 0],
                   t_eval=t_eval, args=(L1, L2, m1, m2), method="RK45", rtol=1e-8, atol=1e-8)
_, _, x2_b, y2_b = compute_positions(sol_b.y[0], sol_b.y[2], L1, L2)

fig_compare = go.Figure()
fig_compare.add_trace(go.Scatter(x=x2, y=y2, name="Original", line=dict(color="royalblue")))
fig_compare.add_trace(go.Scatter(x=x2_b, y=y2_b, name=f"+{perturbation}° perturbed", line=dict(color="crimson")))
fig_compare.update_layout(title="Two Trajectories, Nearly Identical Start", height=450, yaxis=dict(scaleanchor="x"))
st.plotly_chart(fig_compare, use_container_width=True)

divergence = np.sqrt((x2-x2_b)**2 + (y2-y2_b)**2)
st.caption(f"Final separation between the two trajectories: {divergence[-1]:.3f} m (started {perturbation}° apart)")

# ==================== SECTION 4: AI Summary ====================
st.subheader("🤖 AI-Generated Project Summary")

if st.button("Generate AI Summary"):
    with st.spinner("Generating summary..."):
        api_key = os.getenv("GEMINI_API_KEY") or st.secrets.get("GEMINI_API_KEY")
        client = genai.Client(api_key=api_key)

        prompt = f"""You are a physics educator. Write a concise 4-5 sentence plain-English summary of this
double pendulum chaos project for a non-technical reader. Cover: (1) what makes this system chaotic,
(2) the Lyapunov time and what it practically means, (3) how the ML surrogate model performed and its speed advantage,
(4) why the surrogate's accuracy gets worse for longer time predictions.

Lyapunov exponent: 0.253/s, Lyapunov time: 3.95 seconds
Monte Carlo: 500 simulations, uncertainty becomes significant around t=15s
ML surrogate: 4.2cm error at 0-2s, 7.5cm error at 8-10s, 257x faster than numerical solving at scale
"""
        models_to_try = ["gemini-2.0-flash-lite", "gemini-flash-latest"]
        success = False
        for model_name in models_to_try:
            for attempt in range(2):
                try:
                    response = client.models.generate_content(model=model_name, contents=prompt)
                    st.success(response.text)
                    st.caption(f"(Generated using {model_name})")
                    success = True
                    break
                except Exception:
                    if attempt == 0:
                        time_module.sleep(3)
                    continue
            if success:
                break
        if not success:
            st.warning("Google's free-tier AI models are experiencing high demand right now. Please try again in a few minutes.")