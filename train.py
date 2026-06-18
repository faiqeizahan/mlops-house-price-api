from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
import joblib
import os

# Load dataset
housing = fetch_california_housing()

X = housing.data
y = housing.target

# Train model
model = LinearRegression()
model.fit(X, y)

# Create model folder
os.makedirs("model", exist_ok=True)

# Save model
joblib.dump(model, "model/model.pkl")

print("Model trained and saved!")