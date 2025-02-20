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

def generate_synthetic_data(seed, size=500):
    np.random.seed(seed)
    time = pd.date_range('2024-01-01', periods=size, freq='D')
    traffic_speed = np.random.normal(loc=60, scale=10, size=size)
    weather_condition = np.random.normal(loc=75, scale=15, size=size)
    traffic_speed[250:] -= np.random.normal(loc=20, scale=10, size=size-250)
    weather_condition[250:] -= np.random.normal(loc=20, scale=10, size=size-250)
    target = (traffic_speed + weather_condition) > 130
    
    data = pd.DataFrame({'timestamp': time, 'traffic_speed': traffic_speed, 'weather_condition': weather_condition, 'target': target})
    return data

def train_and_log_model(model, model_name, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    input_example = X_train.iloc[0].to_dict()
    
    with mlflow.start_run():
        mlflow.log_param("model_name", model_name)
        mlflow.log_param("accuracy", accuracy)
        mlflow.log_param("precision", precision)
        mlflow.log_param("recall", recall)
        mlflow.sklearn.log_model(model, model_name, input_example=input_example)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
    
    return accuracy, precision, recall

def ks_test(ref_data, curr_data, feature):
    stat, p_value = ks_2samp(ref_data[feature], curr_data[feature])
    return p_value

def retrain_if_needed(reference_data, current_data, model_version='v1'):
    X_ref = reference_data[['traffic_speed', 'weather_condition']]
    y_ref = reference_data['target']
    
    X_curr = current_data[['traffic_speed', 'weather_condition']]
    y_curr = current_data['target']
  
    scaler = StandardScaler()
    X_ref_scaled = scaler.fit_transform(X_ref)
    X_curr_scaled = scaler.transform(X_curr)
    
    X_ref_scaled_df = pd.DataFrame(X_ref_scaled, columns=X_ref.columns)
    X_curr_scaled_df = pd.DataFrame(X_curr_scaled, columns=X_curr.columns)
    
    baseline_model = LogisticRegression()
    accuracy_ref, precision_ref, recall_ref = train_and_log_model(baseline_model, model_version, X_ref_scaled_df, y_ref, X_ref_scaled_df, y_ref)
    
    current_data_degraded = current_data.copy()
    
    current_data_degraded['traffic_speed'] *= 0.2
    X_curr_degraded = current_data_degraded[['traffic_speed', 'weather_condition']]
    y_curr_degraded = current_data_degraded['target']
    
    accuracy_curr, precision_curr, recall_curr = train_and_log_model(baseline_model, model_version, X_curr_scaled_df, y_curr_degraded, X_curr_scaled_df, y_curr_degraded)
    
    p_value_traffic_speed = ks_test(reference_data, current_data, 'traffic_speed')
    p_value_weather_condition = ks_test(reference_data, current_data, 'weather_condition')
    
    print(f"Reference Accuracy: {accuracy_ref:.4f}")
    print(f"Current Accuracy: {accuracy_curr:.4f}")
    print(f"Precision (Reference): {precision_ref:.4f}, Precision (Current): {precision_curr:.4f}")
    print(f"Recall (Reference): {recall_ref:.4f}, Recall (Current): {recall_curr:.4f}")
    print(f"K-S Test p-value for traffic_speed: {p_value_traffic_speed:.4f}")
    print(f"K-S Test p-value for weather_condition: {p_value_weather_condition:.4f}")
    
    if accuracy_curr < accuracy_ref - 0.05 or p_value_traffic_speed < 0.05 or p_value_weather_condition < 0.05:
        print("Performance drift detected or significant feature drift. Retraining model...")
        old_model_dir = f'models/{model_version}_v1'
        if not os.path.exists(old_model_dir):
            os.makedirs(old_model_dir)
        mlflow.pyfunc.save_model(old_model_dir)
        
        model_version = f'v2'
        retrained_model = LogisticRegression()
        retrained_accuracy, retrained_precision, retrained_recall = train_and_log_model(retrained_model, model_version, X_curr_scaled_df, y_curr_degraded, X_curr_scaled_df, y_curr_degraded)
        
        print(f"New model version {model_version} trained with accuracy: {retrained_accuracy}")
    else:
        print(f"No significant drift detected. Continuing with version {model_version}.")

def run_pipeline():
    reference_data = generate_synthetic_data(seed=42, size=500)
    current_data = generate_synthetic_data(seed=99, size=500)

    retrain_if_needed(reference_data, current_data, model_version='v1')

run_pipeline()