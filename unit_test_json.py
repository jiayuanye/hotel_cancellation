import pytest
from fastapi.testclient import TestClient
from app_json import app, InputFeatures

client = TestClient(app)

# Define a test case for the /predict/ endpoint
def test_predict_correct():
    # Sample input features for testing
    input_features = {
        "hotel": "City Hotel",
        "lead_time": 0,
        "arrival_date_year": 0,
        "arrival_date_month": "January",
        "arrival_date_week_number": 1,
        "arrival_date_day_of_month": 0,
        "stays_in_weekend_nights": 0,
        "stays_in_week_nights": 0,
        "adults": 0,
        "children": 0,
        "babies": 0,
        "meal": "BB",
        "country": "PRT",
        "market_segment": "Direct",
        "distribution_channel": "Direct",
        "is_repeated_guest": 0,
        "previous_cancellations": 0,
        "previous_bookings_not_canceled": 0,
        "reserved_room_type": "C",
        "assigned_room_type": "C",
        "booking_changes": 0,
        "deposit_type": "No Deposit",
        "days_in_waiting_list": 0,
        "customer_type": "Transient",
        "adr": 0,
        "required_car_parking_spaces": 0,
        "total_of_special_requests": 0
    }
    # Make a POST request to the /predict/ endpoint with the sample input features
    response = client.post("/predict/", json = input_features)
    
    # Check if the response status code is 200 (OK)
    assert response.status_code == 200
    
    # Check if the response contains the prediction and info fields
    assert "prediction" in response.json()
    assert "info" in response.json() 
    
    assert isinstance(response.json()["prediction"], bool)
    assert response.json()["info"] == "Prediction completed successfully"


def test_string_as_integer_error():
    # Sample input features for testing
    input_features = {
        "hotel": 1011001,  # Invalid input causing an error
        "lead_time": 0,
        "arrival_date_year": 0,
        "arrival_date_month": "January",
        "arrival_date_week_number": 1,
        "arrival_date_day_of_month": 0,
        "stays_in_weekend_nights": 0,
        "stays_in_week_nights": 0,
        "adults": 0,
        "children": 0,
        "babies": 0,
        "meal": "BB",
        "country": "PRT",
        "market_segment": "Direct",
        "distribution_channel": "Direct",
        "is_repeated_guest": 0,
        "previous_cancellations": 0,
        "previous_bookings_not_canceled": 0,
        "reserved_room_type": "C",
        "assigned_room_type": "C",
        "booking_changes": 0,
        "deposit_type": "No Deposit",
        "days_in_waiting_list": 0,
        "customer_type": "Transient",
        "adr": 0,
        "required_car_parking_spaces": 0,
        "total_of_special_requests": 0
    }
    # Make a POST request to the /predict/ endpoint with the sample input features
    response = client.post("/predict/", json=input_features)
    
    # Check if the response status code is 422 (Unprocessable Entity)
    assert response.status_code == 422
    
    # Check if the response contains the expected error message
    expected_error_msg = "Input should be a valid string"
    response_json = response.json()
    assert "detail" in response_json
    assert len(response_json["detail"]) == 1
    assert "msg" in response_json["detail"][0]
    assert response_json["detail"][0]["msg"] == expected_error_msg

def test_integer_as_string_error():
    # Sample input features for testing (with lead_time as "zero" instead of 0)
    input_features = {
        "hotel": "City Hotel",
        "lead_time": "zero",
        "arrival_date_year": 2015,
        "arrival_date_month": "July",
        "arrival_date_week_number": 1,
        "arrival_date_day_of_month": 0,
        "stays_in_weekend_nights": 0,
        "stays_in_week_nights": 0,
        "adults": 2,
        "children": 0,
        "babies": 0,
        "meal": "BB",
        "country": "PRT",
        "market_segment": "Direct",
        "distribution_channel": "Direct",
        "is_repeated_guest": 0,
        "previous_cancellations": 0,
        "previous_bookings_not_canceled": 0,
        "reserved_room_type": "C",
        "assigned_room_type": "C",
        "booking_changes": 3,
        "deposit_type": "No Deposit",
        "days_in_waiting_list": 0,
        "customer_type": "Transient",
        "adr": 0,
        "required_car_parking_spaces": 0,
        "total_of_special_requests": 0
    }
    
    # Make a POST request to the /predict/ endpoint with the sample input features
    response = client.post("/predict/", json=input_features)
    
    # Check if the response status code is 422 (Unprocessable Entity)
    assert response.status_code == 422
    
    # Check if the response contains the expected error message
    expected_error_msg = "Input should be a valid integer, unable to parse string as an integer"
    response_json = response.json()
    assert "detail" in response_json
    assert len(response_json["detail"]) == 1
    assert "msg" in response_json["detail"][0]
    assert expected_error_msg in response_json["detail"][0]["msg"]


def test_not_in_category_error():
    # Sample input features for testing (with hotel as "Morgana Hotel" which may not be in category)
    input_features = {
        "hotel": "Morgana Hotel",
        "lead_time": 0,
        "arrival_date_year": 2015,
        "arrival_date_month": "July",
        "arrival_date_week_number": 1,
        "arrival_date_day_of_month": 0,
        "stays_in_weekend_nights": 0,
        "stays_in_week_nights": 0,
        "adults": 2,
        "children": 0,
        "babies": 0,
        "meal": "BB",
        "country": "PRT",
        "market_segment": "Direct",
        "distribution_channel": "Direct",
        "is_repeated_guest": 0,
        "previous_cancellations": 0,
        "previous_bookings_not_canceled": 0,
        "reserved_room_type": "C",
        "assigned_room_type": "C",
        "booking_changes": 3,
        "deposit_type": "No Deposit",
        "days_in_waiting_list": 0,
        "customer_type": "Transient",
        "adr": 0,
        "required_car_parking_spaces": 0,
        "total_of_special_requests": 0
    }
    
    # Make a POST request to the /predict/ endpoint with the sample input features
    response = client.post("/predict/", json=input_features)
    
    # Check if the response status code is 422 (Unprocessable Entity)
    assert response.status_code == 422
    
    # Check if the response contains the expected error message
    expected_error_msg = "Input should be 'City Hotel' or 'Resort Hotel'"
    response_json = response.json()
    assert "detail" in response_json
    assert len(response_json["detail"]) == 1
    assert "msg" in response_json["detail"][0]
    assert expected_error_msg in response_json["detail"][0]["msg"]

def test_missing_field_error():
    # Sample input features for testing (with the "hotel" field missing)
    input_features = {
        "lead_time": 0,
        "arrival_date_year": 2015,
        "arrival_date_month": "July",
        "arrival_date_week_number": 1,
        "arrival_date_day_of_month": 0,
        "stays_in_weekend_nights": 0,
        "stays_in_week_nights": 0,
        "adults": 2,
        "children": 0,
        "babies": 0,
        "meal": "BB",
        "market_segment": "Direct",
        "distribution_channel": "Direct",
        "is_repeated_guest": 0,
        "previous_cancellations": 0,
        "previous_bookings_not_canceled": 0,
        "reserved_room_type": "C",
        "assigned_room_type": "C",
        "booking_changes": 3,
        "deposit_type": "No Deposit",
        "days_in_waiting_list": 0,
        "customer_type": "Transient",
        "adr": 0,
        "required_car_parking_spaces": 0,
        "total_of_special_requests": 0
    }
    
    # Make a POST request to the /predict/ endpoint with the sample input features
    response = client.post("/predict/", json=input_features)
    
    # Check if the response status code is 422 (Unprocessable Entity)
    assert response.status_code == 422
    
    # Check if the response contains the expected error message
    expected_error_msg = "Field required"
    response_json = response.json()
    assert "detail" in response_json
    assert len(response_json["detail"]) == 2
    assert "msg" in response_json["detail"][0]
    assert expected_error_msg in response_json["detail"][0]["msg"]
