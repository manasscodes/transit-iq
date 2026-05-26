import pandas as pd
import joblib

# Load trained model
model = joblib.load(
    "models/transit_iq_classifier.pkl"
)

sample = pd.DataFrame({
    "transport_type": ["Bus"],
    "route_id": ["Route_1"],
    "origin_station": ["Station_1"],
    "destination_station": ["Station_2"],
    "weather_condition": ["Rain"],
    "temperature_C": [18],
    "humidity_percent": [80],
    "wind_speed_kmh": [25],
    "precipitation_mm": [10],
    "event_type": ["Sports"],
    "event_attendance_est": [10000],
    "traffic_congestion_index": [80],
    "holiday": [0],
    "peak_hour": [1],
    "weekday": [2],
    "season": ["Winter"],
    "day": [10],
    "day_of_week": [2],
    "is_weekend": [0],
    "departure_hour": [8],
    "departure_minute": [15],
    "arrival_hour": [9],
    "scheduled_trip_duration": [45]
})

prediction = model.predict(sample)[0]

probability = model.predict_proba(sample)[0][1]

print("Prediction:", prediction)
print("Delay Probability:", round(probability, 3))