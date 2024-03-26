import pandas as pd
from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
import joblib
import numpy as np

# Define FastAPI app
app = FastAPI()

# Load the machine learning model
model = joblib.load("/Users/zhuyuchen/Desktop/CMU MSBA/Mini 4/Machine Learning For Business Applications/simple_model.pkl")

# Define request body model
class InputFeatures(BaseModel):
    lead_time: int
    arrival_date_year: int
    # Define other features here...

@app.post("/predict/")
async def predict_from_csv(file: UploadFile = File(...)):
    # Read the uploaded CSV file
    contents = await file.read()
    df = pd.read_csv(contents)

    # Data cleaning (if needed)
    # Drop rows with missing values or impute missing values
    # For demonstration, let's assume no data cleaning is needed
    
    # Selecting relevant features
    feature_simple = ['lead_time', 'arrival_date_year']
    df_cleaned = df[feature_simple]

    # Scaling numerical features
    # Scale numerical features if necessary using StandardScaler
    df_scaled = scaler.fit_transform(df_cleaned)

    # Make predictions
    predictions = model.predict(df_scaled)

    # Return predictions
    return {"predictions": predictions.tolist()}

    
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)