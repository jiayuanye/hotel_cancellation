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

def convert_json_to_csv(json_data):
    # Convert JSON to DataFrame
    df = pd.DataFrame(json_data)
    # Assuming the DataFrame is converted to CSV format as a string
    csv_data = df.to_csv(index=False)
    return csv_data

@app.post("/predict/")
async def predict_from_csv(file: UploadFile = File(...)):
    # Read the uploaded file contents
    contents = await file.read()
    
    # Check file extension
    file_extension = file.filename.split(".")[-1]
    
    if file_extension == "csv":
        # Create a file-like object from the bytes read
        file_like_object = io.BytesIO(contents)
        # Read the CSV file into a DataFrame
        df = pd.read_csv(file_like_object)
    
    elif file_extension == "json":
        # Convert JSON to DataFrame
        json_data = json.loads(contents)
        df = convert_json_to_csv(json_data)
    else:
        return {"error": "Unsupported file format. Only CSV and JSON are supported."}

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
    return {"Prediction completed successfully": predictions.tolist()}

    
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)