import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from evidently.report import Report
from evidently.metrics import DataDriftTable

def generate_data(seed, size=500):
    np.random.seed(seed)
    return pd.DataFrame({
        'timestamp': pd.date_range(start='2024-01-01', periods=size, freq='D'),
        'traffic_delay': np.random.normal(loc=30, scale=5, size=size),
        'weather_condition': np.random.normal(loc=15, scale=3, size=size)
    })

reference_data = generate_data(seed=42)
current_data = generate_data(seed=99)

plt.figure(figsize=(12, 5))

plt.hist(reference_data['traffic_delay'], bins=30, alpha=0.5, label='Reference Data', color='blue')
plt.hist(current_data['traffic_delay'], bins=30, alpha=0.5, label='Current Data', color='red')

plt.legend()
plt.xlabel("Traffic Delay (minutes)")
plt.ylabel("Frequency")
plt.title("Traffic Delay Distribution Over Time - Data Drift Example")
plt.show()

report =  Report(metrics=[DataDriftTable()])
report.run(reference_data=reference_data, current_data=current_data)

report_path = 'data_drift_report.html'
report.save_html(report_path)

print(f"Data drift report saved to {report_path}")