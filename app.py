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
    lead_time: int
    arrival_date_year: int
    # Define other features here...

@app.post("/predict/")
async def predict_from_csv(file: UploadFile = File(...)):
    # Read the uploaded CSV file
    contents = await file.read()

    # Create a file-like object from the bytes read
    file_like_object = io.BytesIO(contents)

    df = pd.read_csv(file_like_object)

    # Data cleaning (if needed)
    features = ['hotel',
            'lead_time',
            'arrival_date_year',
            'arrival_date_month', 
            'arrival_date_week_number',
            'arrival_date_day_of_month',
            'stays_in_weekend_nights',
            'stays_in_week_nights',
            'adults',
            'children',
            'babies',
            'meal',
            'country',
            'market_segment',
            'distribution_channel',
            'is_repeated_guest',
            'previous_cancellations',
            'previous_bookings_not_canceled',
            'reserved_room_type',
            'assigned_room_type',
            'booking_changes',
            'deposit_type',
            'days_in_waiting_list',
            'customer_type',
            'adr',
            'required_car_parking_spaces',
            'total_of_special_requests' ]
    
    df_cleaned = df[features]
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
    df_cleaned.fillna(df.median(), inplace=True)

    # Scaling numerical features
    # Scale numerical features if necessary using StandardScaler
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df_cleaned)
    df_scaled_df = pd.DataFrame(df_scaled, columns=df_cleaned.columns)

    # Merge the DataFrames
    merged_df = pd.concat([df_scaled_df, columns], axis=1)
    merged_df = merged_df.loc[:,~merged_df.columns.duplicated()]
    merged_df = merged_df[columns.columns]

    # Fill missing values with 0
    merged_df.fillna(0, inplace=True)

    # Make predictions
    predictions = model.predict(merged_df)

    # Return predictions
    return {"predictions": predictions.tolist()}

    
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)