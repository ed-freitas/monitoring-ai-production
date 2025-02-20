import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from scipy.stats import ks_2samp
import os
import time

# Simulate data generation (traffic speed and weather conditions)
def generate_synthetic_data(seed, size=500):
    np.random.seed(seed)
    time = pd.date_range('2024-01-01', periods=size, freq='D')
    
    # Feature 1: Traffic speed in km/h (e.g., average speed on a given route)
    traffic_speed = np.random.normal(loc=60, scale=10, size=size)
    
    # Feature 2: Weather conditions (scaled from 0 - 100, where 100 is ideal weather for delivery)
    weather_condition = np.random.normal(loc=75, scale=15, size=size)
    
    # Simulate drift after 250 days (e.g., new traffic patterns or weather conditions)
    traffic_speed[250:] -= np.random.normal(loc=20, scale=10, size=size-250)  # Stronger degradation of traffic speed
    weather_condition[250:] -= np.random.normal(loc=20, scale=10, size=size-250)  # Worse weather conditions
    
    # Target: Delivery success (1 = on-time delivery, 0 = delay)
    target = (traffic_speed + weather_condition) > 130  # Arbitrary threshold for on-time delivery
    
    data = pd.DataFrame({'timestamp': time, 'traffic_speed': traffic_speed, 'weather_condition': weather_condition, 'target': target})
    return data

# Function to train and log a model
def train_and_log_model(model, model_name, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # Calculate evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    
    # Log the model with feature names by using pandas DataFrame
    input_example = X_train.iloc[0].to_dict()  # Convert the first row of X_train to a dict for input example
    
    # Log the model and its performance using MLflow
    with mlflow.start_run():
        mlflow.log_param("model_name", model_name)
        mlflow.log_param("accuracy", accuracy)
        mlflow.log_param("precision", precision)
        mlflow.log_param("recall", recall)
        mlflow.sklearn.log_model(model, model_name, input_example=input_example)  # Pass input_example here
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
    
    return accuracy, precision, recall

# Perform a statistical test to detect drift using Kolmogorov-Smirnov (K-S) test
def ks_test(ref_data, curr_data, feature):
    # Perform K-S test to compare the distributions of a feature in reference vs current data
    stat, p_value = ks_2samp(ref_data[feature], curr_data[feature])
    return p_value

# Function to manage versioning and trigger retraining based on performance drift
def retrain_if_needed(reference_data, current_data, model_version='v1'):
    # Split data into features and target
    X_ref = reference_data[['traffic_speed', 'weather_condition']]
    y_ref = reference_data['target']
    
    X_curr = current_data[['traffic_speed', 'weather_condition']]
    y_curr = current_data['target']
    
    # Standardize the features
    scaler = StandardScaler()
    X_ref_scaled = scaler.fit_transform(X_ref)
    X_curr_scaled = scaler.transform(X_curr)
    
    # Convert back to pandas DataFrame with feature names for training
    X_ref_scaled_df = pd.DataFrame(X_ref_scaled, columns=X_ref.columns)
    X_curr_scaled_df = pd.DataFrame(X_curr_scaled, columns=X_curr.columns)
    
    # Train a baseline model (Version 1)
    baseline_model = LogisticRegression()
    accuracy_ref, precision_ref, recall_ref = train_and_log_model(baseline_model, model_version, X_ref_scaled_df, y_ref, X_ref_scaled_df, y_ref)
    
    # Introduce stronger performance degradation on current data (e.g., simulate a lower accuracy on the current dataset)
    current_data_degraded = current_data.copy()
    
    # Intentionally degrade traffic_speed significantly (simulate performance drift)
    current_data_degraded['traffic_speed'] *= 0.2  # Stronger degradation of traffic speed feature
    X_curr_degraded = current_data_degraded[['traffic_speed', 'weather_condition']]
    y_curr_degraded = current_data_degraded['target']
    
    # Train on degraded current data
    accuracy_curr, precision_curr, recall_curr = train_and_log_model(baseline_model, model_version, X_curr_scaled_df, y_curr_degraded, X_curr_scaled_df, y_curr_degraded)
    
    # Perform K-S Test to check for drift in features
    p_value_traffic_speed = ks_test(reference_data, current_data, 'traffic_speed')
    p_value_weather_condition = ks_test(reference_data, current_data, 'weather_condition')
    
    print(f"Reference Accuracy: {accuracy_ref:.4f}")
    print(f"Current Accuracy: {accuracy_curr:.4f}")
    print(f"Precision (Reference): {precision_ref:.4f}, Precision (Current): {precision_curr:.4f}")
    print(f"Recall (Reference): {recall_ref:.4f}, Recall (Current): {recall_curr:.4f}")
    print(f"K-S Test p-value for traffic_speed: {p_value_traffic_speed:.4f}")
    print(f"K-S Test p-value for weather_condition: {p_value_weather_condition:.4f}")
    
    # If performance has degraded or there is significant feature drift, retrain the model
    if accuracy_curr < accuracy_ref - 0.05 or p_value_traffic_speed < 0.05 or p_value_weather_condition < 0.05:
        print("Performance drift detected or significant feature drift. Retraining model...")
        # Save the old model version
        old_model_dir = f'models/{model_version}_v1'
        if not os.path.exists(old_model_dir):
            os.makedirs(old_model_dir)
        mlflow.pyfunc.save_model(old_model_dir)
        
        # Train a new model with updated data
        model_version = f'v2'
        retrained_model = LogisticRegression()
        retrained_accuracy, retrained_precision, retrained_recall = train_and_log_model(retrained_model, model_version, X_curr_scaled_df, y_curr_degraded, X_curr_scaled_df, y_curr_degraded)
        
        print(f"New model version {model_version} trained with accuracy: {retrained_accuracy}")
    else:
        print(f"No significant drift detected. Continuing with version {model_version}.")

# Main function to run the pipeline
def run_pipeline():
    # Generate reference (old) and current (new) data
    reference_data = generate_synthetic_data(seed=42, size=500)
    current_data = generate_synthetic_data(seed=99, size=500)
    
    # Check and retrain if necessary
    retrain_if_needed(reference_data, current_data, model_version='v1')

# Run the retraining pipeline
run_pipeline()
