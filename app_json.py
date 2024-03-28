import io  
import pandas as pd
from fastapi import FastAPI, File, UploadFile
from sklearn.preprocessing import StandardScaler
from pydantic import BaseModel
import joblib
import numpy as np


# Define FastAPI app
app = FastAPI()

# Load the machine learning model
model = joblib.load("/Users/zhuyuchen/Desktop/CMU MSBA/Mini 4/Machine Learning For Business Applications/logistic_regression_model.pkl")
columns = pd.read_csv("/Users/zhuyuchen/Desktop/CMU MSBA/Mini 4/Machine Learning For Business Applications/hotel cancellation/data_columns.csv")

# Define request body model
class InputFeatures(BaseModel):
    hotel: str
    lead_time: int
    arrival_date_year: int
    arrival_date_month: str
    arrival_date_week_number: int
    arrival_date_day_of_month: int
    stays_in_weekend_nights: int
    stays_in_week_nights: int
    adults: int
    children: int
    babies: int
    meal: str
    country: str
    market_segment: str
    distribution_channel: str
    is_repeated_guest: int
    previous_cancellations: int
    previous_bookings_not_canceled: int
    reserved_room_type: str
    assigned_room_type: str
    booking_changes: int
    deposit_type: str
    days_in_waiting_list: int
    customer_type: str
    adr: float
    required_car_parking_spaces: int
    total_of_special_requests: int

class Prediction(BaseModel):
    prediction: bool
    # Define other features here...

# Define API endpoint
@app.post("/predict/")
async def predict(features: InputFeatures):
    input_data = features.dict() # Convert InputFeatures object to dictionary
    df_cleaned = pd.DataFrame([input_data])  # Create DataFrame from dictionary
    # Identify categorical columns
    categorical_columns = ['hotel', 'arrival_date_month', 'meal', 'arrival_date_year', 
                    'arrival_date_week_number', 'arrival_date_day_of_month', 'country', 'market_segment', 
                    'distribution_channel', 'reserved_room_type', 'assigned_room_type',  'deposit_type', 'customer_type']

    # Create dummy variables for the categorical column
    dummy_variables = pd.get_dummies(df_cleaned[categorical_columns], drop_first=True)

    # Concatenate dummy variables with the original DataFrame
    df_cleaned= pd.concat([df_cleaned, dummy_variables], axis=1)

    # Drop the original categorical column if needed
    df_cleaned.drop(categorical_columns, axis=1, inplace=True)

    #  Filling up NA for X with median of the variables
    df_cleaned.fillna(df_cleaned.median(), inplace=True)

    # Merge the DataFrames
    merged_df = pd.concat([df_cleaned, columns], axis=1)
    merged_df = merged_df.loc[:,~merged_df.columns.duplicated()]
    merged_df = merged_df[columns.columns]

    # Fill missing values with 0
    merged_df.fillna(0, inplace=True)

    # Make predictions
    predictions = model.predict(merged_df)

    # Return predictions
    return {"Prediction completed successfully": predictions.tolist()}
    
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)