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

# Define API endpoint
@app.post("/predict/")
async def predict(features: InputFeatures):
    try:
        # Convert input features to numpy array
        input_data = np.array([[features.lead_time,
                                features.arrival_date_year,
                                # Add other features here...
                                ]])

        # Make predictions
        prediction = model.predict(input_data)
        return {"prediction": prediction.tolist()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)