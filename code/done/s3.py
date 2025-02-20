import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from alibi_detect.cd import TabularDrift
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Generate synthetic time-series data with traffic and weather changes (simulating concept drift)
def generate_synthetic_data(seed, size=500):
    np.random.seed(seed)
    time = pd.date_range('2024-01-01', periods=size, freq='D')
    
    # Feature 1: Traffic speed in km/h (e.g., average speed on a given route)
    traffic_speed = np.random.normal(loc=60, scale=10, size=size)
    
    # Feature 2: Weather conditions (scaled from 0 - 100, where 100 is ideal weather for delivery)
    weather_condition = np.random.normal(loc=75, scale=15, size=size)
    
    # Simulate drift after 250 days (e.g., new weather patterns or traffic congestion)
    traffic_speed[250:] -= np.random.normal(loc=10, scale=5, size=size-250)  # Decrease traffic speed
    weather_condition[250:] -= np.random.normal(loc=10, scale=5, size=size-250)  # Worse weather conditions
    
    # Target: Delivery success (1 = on-time delivery, 0 = delay)
    # A simple threshold model where better traffic and weather means higher chance of on-time delivery
    target = (traffic_speed + weather_condition) > 130  # Arbitrary threshold for on-time delivery
    
    data = pd.DataFrame({'timestamp': time, 'traffic_speed': traffic_speed, 'weather_condition': weather_condition, 'target': target})
    return data

# Split data into reference (past) and current (new) data
reference_data = generate_synthetic_data(seed=42, size=500)
current_data = generate_synthetic_data(seed=99, size=500)

# Visualize traffic speed and weather condition distributions before and after drift
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.hist(reference_data['traffic_speed'], bins=30, alpha=0.5, label='Reference', color='blue')
plt.title('Traffic Speed Distribution - Reference')
plt.subplot(1, 2, 2)
plt.hist(current_data['traffic_speed'], bins=30, alpha=0.5, label='Current', color='red')
plt.title('Traffic Speed Distribution - Current')
plt.legend()
plt.show()

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.hist(reference_data['weather_condition'], bins=30, alpha=0.5, label='Reference', color='blue')
plt.title('Weather Condition Distribution - Reference')
plt.subplot(1, 2, 2)
plt.hist(current_data['weather_condition'], bins=30, alpha=0.5, label='Current', color='red')
plt.title('Weather Condition Distribution - Current')
plt.legend()
plt.show()

# Extract features and standardize for drift detection
reference_features = reference_data[['traffic_speed', 'weather_condition']].values
current_features = current_data[['traffic_speed', 'weather_condition']].values
scaler = StandardScaler()
reference_features_scaled = scaler.fit_transform(reference_features)
current_features_scaled = scaler.transform(current_features)

# Detect drift using alibi-detect's TabularDrift
detector = TabularDrift(x_ref=reference_features_scaled, p_val=0.05)
drift_result = detector.predict(current_features_scaled)

# Print drift detection result
if drift_result['data']['is_drift']:
    print("Concept Drift Detected: The distribution of features has changed significantly!")
else:
    print("No drift detected: The feature distributions are consistent.")

# Now, let's simulate model performance drift using logistic regression
X_ref, y_ref = reference_data[['traffic_speed', 'weather_condition']], reference_data['target']
X_curr, y_curr = current_data[['traffic_speed', 'weather_condition']], current_data['target']

# Train a logistic regression model on reference data
model = LogisticRegression()
model.fit(X_ref, y_ref)

# Evaluate on both reference and current data
y_pred_ref = model.predict(X_ref)
y_pred_curr = model.predict(X_curr)

# Calculate accuracy on both reference and current data
accuracy_ref = accuracy_score(y_ref, y_pred_ref)
accuracy_curr = accuracy_score(y_curr, y_pred_curr)

print(f"Accuracy on Reference Data: {accuracy_ref:.4f}")
print(f"Accuracy on Current Data: {accuracy_curr:.4f}")

# If the accuracy has decreased significantly, it indicates potential concept drift
if accuracy_curr < accuracy_ref - 0.05:  # Threshold for significant accuracy drop
    print("Performance drift detected: Model accuracy has decreased.")
else:
    print("No performance drift detected: Model accuracy is consistent.")
