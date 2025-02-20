import pandas as pd
import numpy as np
import time
import psutil
import logging
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.datasets import load_iris

# Set up logging to track real-time metrics
logging.basicConfig(filename='model_performance.log', level=logging.INFO, 
                    format='%(asctime)s - %(message)s')

# Function to track and log model performance metrics
def track_model_performance(model, X_train, X_test, y_train, y_test):
    # Start time to measure latency
    start_time = time.time()
    
    # Train the model
    model.fit(X_train, y_train)
    
    # Predict using the trained model
    y_pred = model.predict(X_test)
    
    # Calculate performance metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    
    # Measure latency
    latency = time.time() - start_time  # Time taken for training and prediction
    
    # Log the metrics
    logging.info(f'Accuracy: {accuracy:.4f}')
    logging.info(f'Precision: {precision:.4f}')
    logging.info(f'Recall: {recall:.4f}')
    logging.info(f'Latency: {latency:.4f} seconds')
    
    # Print the metrics to the console for real-time feedback
    print(f"Model Performance - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, Latency: {latency:.4f} seconds")

# Load dataset (Iris dataset for simplicity)
data = load_iris()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize a Logistic Regression model
model = LogisticRegression(max_iter=200)

# Track and log model performance
track_model_performance(model, X_train, X_test, y_train, y_test)

