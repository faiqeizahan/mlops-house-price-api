import os
import joblib
import mlflow
import mlflow.sklearn

from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

import os

# MinIO Credentials
os.environ["AWS_ACCESS_KEY_ID"] = "minioadmin"
os.environ["AWS_SECRET_ACCESS_KEY"] = "minio123"
os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://127.0.0.1:9000"

# ===========================
# MLflow Configuration

# ===========================

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("House Price Prediction")


# ===========================
# Load Dataset
# ===========================

housing = fetch_california_housing()

X = housing.data
y = housing.target


# ===========================
# Start MLflow Run
# ===========================

with mlflow.start_run():

    # Create Model
    model = LinearRegression()

    # Train Model
    model.fit(X, y)

    # Predict
    predictions = model.predict(X)

    # Calculate Metric
    mse = mean_squared_error(y, predictions)

    # Log Parameters
    mlflow.log_param("model_type", "LinearRegression")

    # Log Metrics
    mlflow.log_metric("mse", mse)

    # Save Model Locally
    os.makedirs("model", exist_ok=True)
    joblib.dump(model, "model/model.pkl")

    # Log Model to MLflow
    mlflow.sklearn.log_model(
        sk_model=model,
        name="house-price-model"
    )

print("======================================")
print("Training completed successfully!")
print(f"Mean Squared Error: {mse}")
print("Model saved to ./model/model.pkl")
print("Logged to MLflow successfully!")
print("======================================")