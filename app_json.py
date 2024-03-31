import io  
import pandas as pd
from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
import joblib
from enum import Enum
import numpy as np


# Define FastAPI app
app = FastAPI()

# Load the machine learning model
model = joblib.load("/Users/zhuyuchen/Desktop/CMU MSBA/Mini 4/Machine Learning For Business Applications/logistic_regression_model.pkl")

# Load the column names for variables used in model
columns = pd.read_csv("/Users/zhuyuchen/Desktop/CMU MSBA/Mini 4/Machine Learning For Business Applications/hotel cancellation/data_columns.csv")

# Define classes for categorical variables
class HotelType(str, Enum):
    CityHotel = 'City Hotel'
    ResortHotel = 'Resort Hotel'

class Month(Enum):
    January = 'January'
    February = 'February'
    March = 'March'
    April = 'April'
    May = 'May'
    June = 'June'
    July = 'July'
    August = 'August'
    September = 'September'
    October = 'October'
    November = 'November'
    December = 'December'

class WeekOfYear(Enum):
    pass

class Meal(Enum):
    BB = 'BB'
    FB = 'FB'
    HB = 'HB'
    SC = 'SC'
    NA = 'Undefined'

class Country(Enum):
    pass

class MarketSegment(Enum):
    pass

class Distribution(Enum):
    pass

class RoomType(Enum):
    pass

class DepositType(Enum):
    NoDeposit = 'No Deposit'
    Refundable = 'Refundable'
    NonRefundable = "Non Refund"

class CustomerType(Enum):
    Transient = 'Transient'
    Contract = 'Contract'
    TransientParty = 'Transient-Party'
    Group = 'Group'

for i in range(1, 54):
    setattr(WeekOfYear, f"Week{i}", i)

unique_countries = ['PRT', 'GBR', 'USA', 'ESP', 'IRL', 'FRA', 'ROU', 'NOR', 'OMN',
       'ARG', 'POL', 'DEU', 'BEL', 'CHE', 'CN', 'GRC', 'ITA', 'NLD',
       'DNK', 'RUS', 'SWE', 'AUS', 'EST', 'CZE', 'BRA', 'FIN', 'MOZ',
       'BWA', 'LUX', 'SVN', 'ALB', 'IND', 'CHN', 'MEX', 'MAR', 'UKR',
       'SMR', 'LVA', 'PRI', 'SRB', 'CHL', 'AUT', 'BLR', 'LTU', 'TUR',
       'ZAF', 'AGO', 'ISR', 'CYM', 'ZMB', 'CPV', 'ZWE', 'DZA', 'KOR',
       'CRI', 'HUN', 'ARE', 'TUN', 'JAM', 'HRV', 'HKG', 'IRN', 'GEO',
       'AND', 'GIB', 'URY', 'JEY', 'CAF', 'CYP', 'COL', 'GGY', 'KWT',
       'NGA', 'MDV', 'VEN', 'SVK', 'FJI', 'KAZ', 'PAK', 'IDN', 'LBN',
       'PHL', 'SEN', 'SYC', 'AZE', 'BHR', 'NZL', 'THA', 'DOM', 'MKD',
       'MYS', 'ARM', 'JPN', 'LKA', 'CUB', 'CMR', 'BIH', 'MUS', 'COM',
       'SUR', 'UGA', 'BGR', 'CIV', 'JOR', 'SYR', 'SGP', 'BDI', 'SAU',
       'VNM', 'PLW', 'QAT', 'EGY', 'PER', 'MLT', 'MWI', 'ECU', 'MDG',
       'ISL', 'UZB', 'NPL', 'BHS', 'MAC', 'TGO', 'TWN', 'DJI', 'STP',
       'KNA', 'ETH', 'IRQ', 'HND', 'RWA', 'KHM', 'MCO', 'BGD', 'IMN',
       'TJK', 'NIC', 'BEN', 'VGB', 'TZA', 'GAB', 'GHA', 'TMP', 'GLP',
       'KEN', 'LIE', 'GNB', 'MNE', 'UMI', 'MYT', 'FRO', 'MMR', 'PAN',
       'BFA', 'LBY', 'MLI', 'NAM', 'BOL', 'PRY', 'BRB', 'ABW', 'AIA',
       'SLV', 'DMA', 'PYF', 'GUY', 'LCA', 'ATA', 'GTM', 'ASM', 'MRT',
       'NCL', 'KIR', 'SDN', 'ATF', 'SLE', 'LAO']

for country_code in unique_countries:
    setattr(Country, country_code, country_code)

market_segment = ['Direct', 'Corporate', 'Online TA', 'Offline TA/TO', 
                  'Complementary', 'Groups', 'Undefined', 'Aviation']

for segments in market_segment:
    setattr(MarketSegment, segments, segments)

distribution_channels = ['Direct', 'Corporate', 'TA/TO', 'Undefined', 'GDS']

for channels in distribution_channels:
    setattr(Distribution, channels, channels)

room_types = ['C', 'A', 'D', 'E', 'G', 'F', 'H', 'L', 'P', 'B']

for types in room_types:
    setattr(RoomType, types, types)

# Define request body model
class InputFeatures(BaseModel):
    hotel: HotelType
    lead_time: int
    arrival_date_year: int
    arrival_date_month: Month
    arrival_date_week_number: int
    arrival_date_day_of_month: int
    stays_in_weekend_nights: int
    stays_in_week_nights: int
    adults: int
    children: int
    babies: int
    meal: Meal
    country: Country
    market_segment: MarketSegment
    distribution_channel: Distribution
    is_repeated_guest: int
    previous_cancellations: int
    previous_bookings_not_canceled: int
    reserved_room_type: RoomType
    assigned_room_type: RoomType
    booking_changes: int
    deposit_type: DepositType
    days_in_waiting_list: int
    customer_type: CustomerType
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
    categorical_columns = ['hotel', 'arrival_date_month', 'meal', 
                    'arrival_date_week_number', 'country', 'market_segment', 
                    'distribution_channel', 'reserved_room_type', 'assigned_room_type',  
                    'deposit_type', 'customer_type']

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