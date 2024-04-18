import gradio as gr
import requests
from gradio.components import Textbox, Number, Dropdown

# Define a placeholder function
# Define function to be called on interface's submission
def predict(*args):
    # Prepare data to send to FastAPI endpoint
    data = {
        "hotel": args[0],
        "lead_time": args[1],
        "arrival_date_year": args[2],
        "arrival_date_month": args[3],
        "arrival_date_week_number": args[4],
        "arrival_date_day_of_month": args[5],
        "stays_in_weekend_nights": args[6],
        "stays_in_week_nights": args[7],
        "adults": args[8],
        "children": args[9],
        "babies": args[10],
        "meal": args[11],
        "country": args[12],
        "market_segment": args[13],
        "distribution_channel": args[14],
        "is_repeated_guest": args[15],
        "previous_cancellations": args[16],
        "previous_bookings_not_canceled": args[17],
        "reserved_room_type": args[18],
        "assigned_room_type": args[19],
        "booking_changes": args[20],
        "deposit_type": args[21],
        "days_in_waiting_list": args[22],
        "customer_type": args[23],
        "adr": args[24],
        "required_car_parking_spaces": args[25],
        "total_of_special_requests": args[26]
    }
    
    # Make a POST request to FastAPI endpoint
    response = requests.post("http://127.0.0.1:8000/predict/", json=data)
    
    # Check if response is successful
    if response.status_code == 200:
        try:
            # Extract prediction result from response
            prediction = response.json()["prediction"]
            return prediction
        except KeyError:
            return "Error: Prediction not found in response"
    else:
        return "Error: Failed to fetch prediction"
    
# Define Gradio Interface
iface = gr.Interface(
    fn=predict,  # Use the placeholder function
    inputs=[
        Dropdown(
            choices=["City Hotel", "Resort Hotel"],
            label="Hotel Type"
        ),
        Number(label="Lead Time"),
        Number(label="Arrival Date Year"),
        Dropdown(
            choices=["January", "February", "March", "April", "May", "June", 
                     "July", "August", "September", "October", "November", "December"],
            label="Arrival Date Month"
        ),
        Number(label="Arrival Date Week Number"),
        Number(label="Arrival Date Day of Month"),
        Number(label="Stays in Weekend Nights"),
        Number(label="Stays in Week Nights"),
        Number(label="Adults"),
        Number(label="Children"),
        Number(label="Babies"),
        Dropdown(
            choices=["BB", "FB", "HB", "SC", "Undefined"],
            label="Meal"
        ),
        Dropdown(choices = ['PRT', 'GBR', 'USA', 'ESP', 'IRL', 'FRA', 'ROU', 'NOR', 'OMN',
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
                                'NCL', 'KIR', 'SDN', 'ATF', 'SLE', 'LAO'],
                 label="Country"),
        Dropdown(
            choices=["Direct", "Corporate", "Online TA", "Offline TA/TO",
                     "Complementary", "Groups", "Undefined", "Aviation"],
            label="Market Segment"
        ),
        Dropdown(
            choices=["Direct", "Corporate", "TA/TO", "Undefined", "GDS"],
            label="Distribution Channel"
        ),
        Number(label="Is Repeated Guest"),
        Number(label="Previous Cancellations"),
        Number(label="Previous Bookings Not Canceled"),
        Dropdown(
            choices=["C", "A", "D", "E", "G", "F", "H", "L", "P", "B"],
            label="Reserved Room Type"
        ),
        Dropdown(
            choices=["C", "A", "D", "E", "G", "F", "H", "L", "P", "B"],
            label="Assigned Room Type"
        ),
        Number(label="Booking Changes"),
        Dropdown(
            choices=["No Deposit", "Refundable", "Non Refund"],
            label="Deposit Type"
        ),
        Number(label="Days in Waiting List"),
        Dropdown(
            choices=["Transient", "Contract", "Transient-Party", "Group"],
            label="Customer Type"
        ),
        Number(label="ADR"),
        Number(label="Required Car Parking Spaces"),
        Number(label="Total of Special Requests")
    ],
    outputs=Textbox(label="Prediction")
)



# Run Gradio Interface
iface.launch()
