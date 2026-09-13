# 🌀 Double Pendulum Chaos Simulation

A physics simulation and machine learning project exploring chaos theory through the classic double pendulum system — numerical differential equation solving, Lyapunov chaos quantification, Monte Carlo uncertainty analysis, and a neural network surrogate model, tied together in a live interactive dashboard.

**🔗 Live app:** https://double-pendulum-chaos-dashboard-vatsala.streamlit.app/

## Overview

A double pendulum (a pendulum attached to the end of another pendulum) is one of the simplest systems in physics that exhibits true chaos — deterministic motion that becomes practically unpredictable due to extreme sensitivity to initial conditions. This project simulates it numerically, mathematically quantifies its chaos, and tests whether machine learning can approximate it faster than solving the physics directly.

## Tech Stack

- **Numerical Simulation:** SciPy (`solve_ivp`, Runge-Kutta 4/5)
- **Physics:** Lagrangian mechanics, energy conservation validation
- **Machine Learning:** PyTorch (feedforward neural network surrogate model)
- **Dashboard:** Streamlit, Plotly (animated, interactive)
- **AI Integration:** Google Gemini API

## Pipeline

1. **Simulation** — Derived and numerically solved the double pendulum's equations of motion via Runge-Kutta integration, validated by confirming energy conservation (relative drift of 3.3×10⁻⁹, essentially perfect).
2. **Lyapunov Chaos Analysis** — Simulated two trajectories differing by just 10⁻⁶ radians and measured their exponential divergence, calculating a Lyapunov exponent of 0.253/s and a Lyapunov time of ~3.95 seconds.
3. **Monte Carlo Sensitivity** — Ran a 500-simulation ensemble with randomized starting angles to visualize how positional uncertainty grows from near-zero to significant within about 15 seconds.
4. **ML Surrogate Model** — Trained a PyTorch neural network to predict pendulum position directly from (starting angle, time), achieving 1.7-7.5cm accuracy and a 257x speedup over numerical solving at batch scale.
5. **Interactive Dashboard** — Built a live-animated, slider-controlled Streamlit app, including a tool to interactively compare two near-identical trajectories and watch chaos unfold in real time, plus an AI-generated plain-English summary via the Gemini API.

## Key Findings

- **Chaos, quantified, not just observed.** A positive Lyapunov exponent (0.253/s) mathematically confirms the system is chaotic, and the Lyapunov time (~3.95s) gives a concrete, physically meaningful answer to "how long can this be predicted?" — even with near-perfect knowledge of the starting angle, useful prediction is gone within about 4 seconds.
- **The ML surrogate model's accuracy degrades with prediction horizon** (4.2cm error at 0-2s, growing to 7.5cm at 8-10s) — not due to a modeling flaw, but as a direct, expected consequence of the same chaos quantified by the Lyapunov analysis. No amount of additional training data can fix this; it's a fundamental limit of the physical system, not the model.
- **The surrogate still adds real practical value** despite this limit: a 257x speedup over numerical integration at the batch scale needed for Monte Carlo-style analysis (150,000 evaluations), making it genuinely useful for short-to-medium horizon predictions or applications needing many fast approximate simulations.
- **Two independent methods (pairwise Lyapunov divergence and ensemble Monte Carlo spread) both confirm chaos**, using different metrics and thresholds — their exact crossing times differ (as expected, since they measure different things), but both tell the same underlying story.

## Run Locally

```bash
git clone https://github.com/kamble-vatsala/Double-pendulum-chaos-dashboard
cd double-pendulum-chaos
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
streamlit run app.py
```

You'll need a free Gemini API key from [Google AI Studio](https://aistudio.google.com) — add it to a `.env` file as `GEMINI_API_KEY=your_key_here`.

## Project Structure

```
├── simulate_pendulum.py        # Core ODE simulation + energy conservation check
├── lyapunov_analysis.py         # Chaos quantification via Lyapunov exponent
├── monte_carlo_pendulum.py      # 500-simulation ensemble sensitivity analysis
├── train_surrogate.py           # PyTorch neural network surrogate model
├── save_pendulum_summary.py     # Saves project findings summary to the database
├── app.py                       # Interactive Streamlit dashboard
├── requirements.txt
└── data/pendulum.db              # SQLite database (simulations, analysis results)
```