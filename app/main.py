from fastapi import FastAPI
from pydantic import BaseModel

class HouseData(BaseModel):
    MedInc: float
    HouseAge: float
    AveRooms: float
    AveBedrms: float
    Population: float
    AveOccup: float
    Latitude: float
    Longitude: float
import joblib

app = FastAPI()

model = joblib.load("model/model.pkl")


@app.get("/")
def home():
    return {"message": "MLOps API Running"}


@app.post("/predict")
def predict(data: HouseData):

    sample = [[
        data.MedInc,
        data.HouseAge,
        data.AveRooms,
        data.AveBedrms,
        data.Population,
        data.AveOccup,
        data.Latitude,
        data.Longitude
    ]]

    prediction = model.predict(sample)

    return {
        "predicted_house_price": float(prediction[0])
    }