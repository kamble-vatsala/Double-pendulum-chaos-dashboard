import sqlite3
import pandas as pd

conn = sqlite3.connect("data/pendulum.db")

summary_data = [
    {"Task": "Energy Conservation", "Metric": "Relative Drift", "Value": "3.30e-09"},
    {"Task": "Lyapunov Analysis", "Metric": "Exponent (λ)", "Value": "0.253 /s"},
    {"Task": "Lyapunov Analysis", "Metric": "Lyapunov Time", "Value": "3.95 s"},
    {"Task": "Monte Carlo", "Metric": "Simulations Run", "Value": "500"},
    {"Task": "Monte Carlo", "Metric": "Unpredictable At", "Value": "~14.95 s"},
    {"Task": "ML Surrogate", "Metric": "Error (0-2s)", "Value": "4.2 cm"},
    {"Task": "ML Surrogate", "Metric": "Error (8-10s)", "Value": "7.5 cm"},
    {"Task": "ML Surrogate", "Metric": "Speedup vs Numerical", "Value": "257x"},
]

pd.DataFrame(summary_data).to_sql("project_summary", conn, if_exists="replace", index=False)
conn.close()
print("Summary saved.")