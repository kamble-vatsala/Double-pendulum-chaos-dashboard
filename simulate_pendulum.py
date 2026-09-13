import numpy as np
from scipy.integrate import solve_ivp
import sqlite3
import matplotlib.pyplot as plt

# ---- Physical parameters ----
g = 9.81       # gravity (m/s^2)
L1, L2 = 1.0, 1.0    # rod lengths (m)
m1, m2 = 1.0, 1.0    # bob masses (kg)

# ---- Initial conditions ----
theta1_0 = np.pi / 2    # 90 degrees from vertical
theta2_0 = np.pi / 2
z1_0 = 0.0               # initial angular velocities
z2_0 = 0.0

def equations_of_motion(t, state):
    """Standard double pendulum equations of motion, derived via Lagrangian mechanics."""
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
    """Convert angles to (x, y) Cartesian positions for both bobs."""
    x1 = L1 * np.sin(theta1)
    y1 = -L1 * np.cos(theta1)
    x2 = x1 + L2 * np.sin(theta2)
    y2 = y1 - L2 * np.cos(theta2)
    return x1, y1, x2, y2

def total_energy(theta1, z1, theta2, z2):
    """Total mechanical energy (kinetic + potential) - should stay constant if simulation is accurate."""
    x1, y1, x2, y2 = compute_positions(theta1, theta2)
    v1_sq = L1**2 * z1**2
    v2_sq = L1**2*z1**2 + L2**2*z2**2 + 2*L1*L2*z1*z2*np.cos(theta1-theta2)
    KE = 0.5*m1*v1_sq + 0.5*m2*v2_sq
    PE = m1*g*y1 + m2*g*y2
    return KE + PE

# ---- Solve the ODE ----
t_span = (0, 20)             # simulate 20 seconds
t_eval = np.linspace(*t_span, 2000)   # 2000 time steps for smooth output

solution = solve_ivp(
    equations_of_motion, t_span, [theta1_0, z1_0, theta2_0, z2_0],
    t_eval=t_eval, method="RK45", rtol=1e-10, atol=1e-10
)

theta1, z1, theta2, z2 = solution.y
x1, y1, x2, y2 = compute_positions(theta1, theta2)
energy = total_energy(theta1, z1, theta2, z2)

# ---- Sanity check: energy conservation ----
energy_drift = (energy.max() - energy.min()) / abs(energy[0])
print(f"Initial energy: {energy[0]:.6f} J")
print(f"Energy drift over simulation: {energy_drift:.2e} (should be very close to 0)")

# Better energy check: compare drift to a meaningful physical scale,
# not the (misleadingly near-zero) initial energy
characteristic_energy = (m1 + m2) * g * (L1 + L2)  # max possible potential energy scale
absolute_drift = energy.max() - energy.min()
relative_drift_fixed = absolute_drift / characteristic_energy

print(f"\nAbsolute energy drift: {absolute_drift:.2e} J")
print(f"Relative drift (vs characteristic energy scale): {relative_drift_fixed:.2e}")

# ---- Save to SQLite ----
conn = sqlite3.connect("data/pendulum.db")
import pandas as pd
df = pd.DataFrame({
    "time": solution.t, "theta1": theta1, "z1": z1, "theta2": theta2, "z2": z2,
    "x1": x1, "y1": y1, "x2": x2, "y2": y2, "energy": energy
})
df.to_sql("simulation_run1", conn, if_exists="replace", index=False)
conn.close()

# ---- Plot the trajectory of the second bob (the classic chaotic pattern) ----
plt.figure(figsize=(8, 8))
plt.plot(x2, y2, linewidth=0.5, color="darkblue")
plt.title("Double Pendulum: Path Traced by the Second Bob")
plt.xlabel("x (m)")
plt.ylabel("y (m)")
plt.axis("equal")
plt.savefig("data/trajectory_plot.png", dpi=150)
print("\nSaved trajectory plot to data/trajectory_plot.png")
print(f"Simulation saved to data/pendulum.db ({len(df)} time steps)")