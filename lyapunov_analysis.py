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

def run_simulation(theta1_0, t_span=(0, 20), n_points=2000):
    t_eval = np.linspace(*t_span, n_points)
    sol = solve_ivp(equations_of_motion, t_span, [theta1_0, 0, np.pi/2, 0],
                     t_eval=t_eval, method="RK45", rtol=1e-10, atol=1e-10)
    return sol.t, sol.y

# ---- Two near-identical starting conditions ----
epsilon = 1e-6   # tiny perturbation: 0.000057 degrees
t, state_a = run_simulation(np.pi/2)
_, state_b = run_simulation(np.pi/2 + epsilon)

# ---- Measure divergence over time (Euclidean distance in state space) ----
diff = state_a - state_b
distance = np.sqrt(np.sum(diff**2, axis=0))
distance[distance == 0] = 1e-16  # avoid log(0)

# ---- Estimate Lyapunov exponent: fit a line to log(distance) vs time ----
# In a chaotic system, distance grows exponentially: d(t) = d(0) * e^(lambda*t)
# So log(d(t)) is LINEAR in t, and the slope IS the Lyapunov exponent
log_distance = np.log(distance)

# Fit only the early-to-mid portion, before the system saturates (distance can't exceed
# the pendulum's physical size, so exponential growth can't continue forever)
fit_mask = (t > 0.5) & (t < 8) & (distance < 1.0)
lyapunov_exponent = np.polyfit(t[fit_mask], log_distance[fit_mask], 1)[0]

print(f"Initial perturbation: {epsilon:.2e} radians")
print(f"Estimated Lyapunov exponent: {lyapunov_exponent:.4f} per second")
print(f"Lyapunov time (predictability horizon): {1/lyapunov_exponent:.2f} seconds")
print("(A positive Lyapunov exponent confirms chaos - the system loses predictability exponentially fast)")

# ---- Save and plot ----
conn = sqlite3.connect("data/pendulum.db")
pd.DataFrame({"time": t, "distance": distance, "log_distance": log_distance}).to_sql(
    "lyapunov_divergence", conn, if_exists="replace", index=False)
conn.close()

plt.figure(figsize=(10, 5))
plt.plot(t, log_distance, label="log(distance) between trajectories")
plt.plot(t[fit_mask], np.polyval([lyapunov_exponent, np.polyfit(t[fit_mask], log_distance[fit_mask], 1)[1]], t[fit_mask]),
          "r--", label=f"Linear fit (slope = λ = {lyapunov_exponent:.3f})")
plt.xlabel("Time (s)")
plt.ylabel("log(distance between trajectories)")
plt.title("Exponential Divergence of Nearby Trajectories (Chaos)")
plt.legend()
plt.savefig("data/lyapunov_plot.png", dpi=150)
print("\nSaved plot to data/lyapunov_plot.png")