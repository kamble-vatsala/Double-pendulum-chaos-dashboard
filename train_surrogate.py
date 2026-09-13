import numpy as np
from scipy.integrate import solve_ivp
import torch
import torch.nn as nn
import time
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

# ---- Generate training data: many (initial_angle, time) -> (x2, y2) pairs ----
print("Generating training data...")
n_ics = 300                             # 300 different starting angles
t_span = (0, 10)
n_time_points = 100

np.random.seed(42)
initial_angles = np.random.uniform(np.pi/2 - 0.5, np.pi/2 + 0.5, n_ics)
t_eval = np.linspace(*t_span, n_time_points)

X_data, y_data = [], []
for theta1_0 in initial_angles:
    sol = solve_ivp(equations_of_motion, t_span, [theta1_0, 0, np.pi/2, 0],
                     t_eval=t_eval, method="RK45", rtol=1e-8, atol=1e-8)
    x2, y2 = compute_positions(sol.y[0], sol.y[2])
    for i, t in enumerate(t_eval):
        X_data.append([theta1_0, t])
        y_data.append([x2[i], y2[i]])

X_data = np.array(X_data)
y_data = np.array(y_data)
print(f"Generated {len(X_data)} training examples from {n_ics} simulations")

# ---- Normalize inputs/outputs (neural nets train much better on similar-scale numbers) ----
X_mean, X_std = X_data.mean(axis=0), X_data.std(axis=0)
y_mean, y_std = y_data.mean(axis=0), y_data.std(axis=0)
X_norm = (X_data - X_mean) / X_std
y_norm = (y_data - y_mean) / y_std

# Time-based-ish split: hold out 20% of (angle, time) combos at random for testing
split_idx = np.random.permutation(len(X_norm))
n_test = int(0.2 * len(X_norm))
test_idx, train_idx = split_idx[:n_test], split_idx[n_test:]

X_train = torch.tensor(X_norm[train_idx], dtype=torch.float32)
y_train = torch.tensor(y_norm[train_idx], dtype=torch.float32)
X_test = torch.tensor(X_norm[test_idx], dtype=torch.float32)
y_test_norm = y_norm[test_idx]
t_test = X_data[test_idx, 1]  # actual (un-normalized) time values, for later analysis

# ---- Simple feedforward network ----
class SurrogateModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 2)
        )
    def forward(self, x):
        return self.net(x)

model = SurrogateModel()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

print("\nTraining surrogate model...")
epochs = 300
batch_size = 256
n_samples = len(X_train)

for epoch in range(epochs):
    # Shuffle every epoch - important for proper mini-batch training
    permutation = torch.randperm(n_samples)
    epoch_loss = 0
    n_batches = 0

    for i in range(0, n_samples, batch_size):
        idx = permutation[i:i+batch_size]
        batch_X, batch_y = X_train[idx], y_train[idx]

        optimizer.zero_grad()
        preds = model(batch_X)
        loss = criterion(preds, batch_y)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        n_batches += 1

    if (epoch+1) % 50 == 0:
        print(f"Epoch {epoch+1}/{epochs} - Avg Loss: {epoch_loss/n_batches:.6f}")

# ---- Evaluate: accuracy vs how far into the future we're predicting ----
model.eval()
with torch.no_grad():
    preds_norm = model(X_test).numpy()
preds = preds_norm * y_std + y_mean
actual = y_test_norm * y_std + y_mean

errors = np.sqrt(np.sum((preds - actual)**2, axis=1))  # position error in meters, per prediction

# Bucket errors by how far into the future they were predicting
time_buckets = [(0, 2), (2, 4), (4, 6), (6, 8), (8, 10)]
print("\n=== Surrogate Accuracy by Prediction Horizon ===")
for lo, hi in time_buckets:
    mask = (t_test >= lo) & (t_test < hi)
    if mask.sum() > 0:
        print(f"t = {lo}-{hi}s: Mean position error = {errors[mask].mean():.4f} m  (n={mask.sum()})")

# ---- Speed comparison: surrogate vs numerical solver ----
# ---- Speed comparison: realistic scale, matching Task 3's Monte Carlo workload ----
n_ics_speed = 500
n_time_speed = 300
t_speed = np.linspace(0, 10, n_time_speed)
test_angles_speed = np.random.uniform(np.pi/2 - 0.5, np.pi/2 + 0.5, n_ics_speed)

# Numerical: must solve each initial condition separately (sequential)
start = time.time()
for theta1_0 in test_angles_speed:
    _ = solve_ivp(equations_of_motion, (0, 10), [theta1_0, 0, np.pi/2, 0],
                   t_eval=t_speed, method="RK45", rtol=1e-8, atol=1e-8)
numerical_time = time.time() - start

# Surrogate: one big batched forward pass covering ALL (angle, time) combinations at once
batch_inputs = np.array([[theta1_0, t] for theta1_0 in test_angles_speed for t in t_speed])
batch_inputs_norm = (batch_inputs - X_mean) / X_std
batch_tensor = torch.tensor(batch_inputs_norm, dtype=torch.float32)

model.eval()
with torch.no_grad():
    _ = model(batch_tensor)  # warmup call, excluded from timing
    start = time.time()
    _ = model(batch_tensor)
    surrogate_time = time.time() - start

print(f"\n=== Speed Comparison ({n_ics_speed} simulations x {n_time_speed} time points = {n_ics_speed*n_time_speed:,} evaluations) ===")
print(f"Numerical ODE solver ({n_ics_speed} sequential solves): {numerical_time:.2f} s")
print(f"ML surrogate (1 batched forward pass):    {surrogate_time:.4f} s")
print(f"Speedup: {numerical_time/surrogate_time:.0f}x")

# Save model weights and results summary for the dashboard
torch.save(model.state_dict(), "data/surrogate_model.pt")
np.save("data/norm_params.npy", {"X_mean": X_mean, "X_std": X_std, "y_mean": y_mean, "y_std": y_std})
print("\nModel saved to data/surrogate_model.pt")