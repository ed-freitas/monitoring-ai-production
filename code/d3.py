import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from alibi_detect.cd import TabularDrift
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def generate_synthetic_data(seed, size=500):
    np.random.seed(seed)
    time = pd.date_range('2024-01-01', periods=size, freq='D')
    
    traffic_speed = np.random.normal(loc=60, scale=10, size=size)   
    weather_condition = np.random.normal(loc=75, scale=15, size=size)
    traffic_speed[250:] -= np.random.normal(loc=10, scale=5, size=size-250)
    weather_condition[250:] -= np.random.normal(loc=10, scale=5, size=size-250)
    target = (traffic_speed + weather_condition) > 130
    
    data = pd.DataFrame({'timestamp': time, 'traffic_speed': traffic_speed, 'weather_condition': weather_condition, 'target': target})
    return data

reference_data = generate_synthetic_data(seed=42, size=500)
current_data = generate_synthetic_data(seed=99, size=500)

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.hist(reference_data['traffic_speed'], bins=30, alpha=0.5, label='Reference', color='blue')
plt.title('Traffic Speed Distribution - Reference')
plt.subplot(1, 2, 2)
plt.hist(current_data['traffic_speed'], bins=30, alpha=0.5, label='Current', color='red')
plt.title('Traffic Speed Distribution - Current')
plt.legend()
plt.show()

reference_features = reference_data[['traffic_speed', 'weather_condition']].values
current_features = current_data[['traffic_speed', 'weather_condition']].values
scaler = StandardScaler()
reference_features_scaled = scaler.fit_transform(reference_features)
current_features_scaled = scaler.transform(current_features)

detector = TabularDrift(x_ref=reference_features_scaled, p_val=0.05)
drift_result = detector.predict(current_features_scaled)

if drift_result['data']['is_drift']:
    print("Concept Drift Detected: The distribution of features has changed significantly!")
else:
    print("No drift detected: The feature distributions are consistent.")

X_ref, y_ref = reference_data[['traffic_speed', 'weather_condition']], reference_data['target']
X_curr, y_curr = current_data[['traffic_speed', 'weather_condition']], current_data['target']

model = LogisticRegression()
model.fit(X_ref, y_ref)

y_pred_ref = model.predict(X_ref)
y_pred_curr = model.predict(X_curr)

accuracy_ref = accuracy_score(y_ref, y_pred_ref)
accuracy_curr = accuracy_score(y_curr, y_pred_curr)

print(f"Accuracy on Reference Data: {accuracy_ref:.4f}")
print(f"Accuracy on Current Data: {accuracy_curr:.4f}")

if accuracy_curr < accuracy_ref - 0.05:
    print("Performance drift detected: Model accuracy has decreased.")
else:
    print("No performance drift detected: Model accuracy is consistent.")