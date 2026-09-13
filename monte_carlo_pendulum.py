import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import sqlite3
import pandas as pd

g = 9.81
L1, L2 = 1.0, 1.0
m1, m2 = 1.0, 1.0

def equations_of_motion(t, state):
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

def compute_positions(theta1, theta2):
    x1, y1 = L1*np.sin(theta1), -L1*np.cos(theta1)
    x2 = x1 + L2*np.sin(theta2)
    y2 = y1 - L2*np.cos(theta2)
    return x2, y2

# ---- Ensemble simulation ----
n_simulations = 500       # 500 pendulums, each with a slightly different starting angle
t_span = (0, 15)
n_points = 300
t_eval = np.linspace(*t_span, n_points)

np.random.seed(42)
# Realistic "measurement uncertainty": starting angle known only to within ~0.01 degrees
initial_angles = np.pi/2 + np.random.normal(0, 0.0001, n_simulations)

all_x2 = np.zeros((n_simulations, n_points))
all_y2 = np.zeros((n_simulations, n_points))

print(f"Running {n_simulations} simulations...")
for i, theta1_0 in enumerate(initial_angles):
    sol = solve_ivp(equations_of_motion, t_span, [theta1_0, 0, np.pi/2, 0],
                     t_eval=t_eval, method="RK45", rtol=1e-8, atol=1e-8)
    x2, y2 = compute_positions(sol.y[0], sol.y[2])
    all_x2[i], all_y2[i] = x2, y2
    if (i+1) % 100 == 0:
        print(f"  {i+1}/{n_simulations} done")

# ---- Quantify spread (uncertainty) over time ----
position_std = np.sqrt(all_x2.std(axis=0)**2 + all_y2.std(axis=0)**2)

# Find when the ensemble "loses predictability" - spread exceeds 10% of pendulum's max reach
max_reach = L1 + L2
threshold = 0.1 * max_reach
unpredictable_idx = np.argmax(position_std > threshold)
unpredictable_time = t_eval[unpredictable_idx] if position_std[unpredictable_idx] > threshold else None

print(f"\nEnsemble spread starts near 0, grows to {position_std[-1]:.3f} m by t={t_eval[-1]:.0f}s")
if unpredictable_time:
    print(f"Position becomes 'unpredictable' (spread > 10% of max reach) at t ≈ {unpredictable_time:.2f}s")

# ---- Save summary and plot ----
conn = sqlite3.connect("data/pendulum.db")
pd.DataFrame({"time": t_eval, "position_std": position_std}).to_sql(
    "monte_carlo_spread", conn, if_exists="replace", index=False)
conn.close()

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Left: fan of trajectories
for i in range(0, n_simulations, 10):  # plot every 10th for clarity
    axes[0].plot(all_x2[i], all_y2[i], alpha=0.1, color="steelblue", linewidth=0.5)
axes[0].set_title(f"{n_simulations} Trajectories from Near-Identical Starting Angles")
axes[0].set_xlabel("x (m)"); axes[0].set_ylabel("y (m)")
axes[0].axis("equal")

# Right: uncertainty growth over time
axes[1].plot(t_eval, position_std, color="darkred")
axes[1].axhline(threshold, color="gray", linestyle="--", label="10% of max reach")
axes[1].set_title("Growth of Positional Uncertainty Over Time")
axes[1].set_xlabel("Time (s)"); axes[1].set_ylabel("Std Dev of Position (m)")
axes[1].legend()

plt.tight_layout()
plt.savefig("data/monte_carlo_plot.png", dpi=150)
print("\nSaved plot to data/monte_carlo_plot.png")