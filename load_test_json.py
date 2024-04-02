from locust import HttpUser, task, between


class PredictUser(HttpUser):
    wait_time = between(1, 3)  # wait time between tasks

    @task
    def predict(self):
        # Define sample input features for testing
        input_features = {
            "hotel": "City Hotel",
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
        
        # Make a POST request to the /predict/ endpoint
        self.client.post("/predict/", json=input_features)
