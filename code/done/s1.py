import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from evidently.report import Report
from evidently.metrics import DataDriftTable

# Generate synthetic time-series data to simulate data drift
def generate_data(seed, size=500):
    np.random.seed(seed)
    return pd.DataFrame({
        'timestamp': pd.date_range(start='2024-01-01', periods=size, freq='D'),
        'traffic_delay': np.random.normal(loc=30, scale=5, size=size),  # Simulating traffic delays
        'weather_condition': np.random.normal(loc=15, scale=3, size=size) # Simulating weather conditions
    })

# Simulate past data (reference) and new incoming data (current)
reference_data = generate_data(seed=42)  # Historical data (before expansion)
current_data = generate_data(seed=99)    # New data (after expansion, with drift)

# Visualize feature distributions over time to show how they evolve
plt.figure(figsize=(12, 5))

# Plotting traffic_delay distribution for both reference and current data
plt.hist(reference_data['traffic_delay'], bins=30, alpha=0.5, label='Reference Data', color='blue')
plt.hist(current_data['traffic_delay'], bins=30, alpha=0.5, label='Current Data', color='red')

plt.legend()
plt.xlabel("Traffic Delay (minutes)")
plt.ylabel("Frequency")
plt.title("Traffic Delay Distribution Over Time - Data Drift Example")
plt.show()

# Use Evidently to generate a data drift report
report = Report(metrics=[DataDriftTable()])
report.run(reference_data=reference_data, current_data=current_data)

# Save the drift report as an HTML file for inspection
report_path = 'data_drift_report.html'
report.save_html(report_path)

print(f"Data drift report saved to {report_path}")
