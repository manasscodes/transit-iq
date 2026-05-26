import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="TransitIQ",
    page_icon="🚌",
    layout="centered"
)

st.title("🚌 TransitIQ")
st.write("AI-Powered Transit Delay Intelligence")

# Load model safely
try:
    model = joblib.load("models/transit_iq_classifier.pkl")
    st.success("Model loaded successfully")
except Exception as e:
    st.error(f"Model loading failed: {e}")
    st.stop()

transport_type = st.selectbox(
    "Transport Type",
    ["Bus", "Metro", "Train", "Tram"]
)

weather_condition = st.selectbox(
    "Weather Condition",
    ["Clear", "Cloudy", "Fog", "Rain", "Snow", "Storm"]
)

traffic_congestion = st.slider(
    "Traffic Congestion Index",
    0,
    100,
    50
)

if st.button("Predict Delay"):

    sample = pd.DataFrame({
        "transport_type": [transport_type],
        "route_id": ["Route_1"],
        "origin_station": ["Station_1"],
        "destination_station": ["Station_2"],
        "weather_condition": [weather_condition],
        "temperature_C": [20],
        "humidity_percent": [70],
        "wind_speed_kmh": [15],
        "precipitation_mm": [5],
        "event_type": ["No_Event"],
        "event_attendance_est": [0],
        "traffic_congestion_index": [traffic_congestion],
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

    try:
        prediction = model.predict(sample)[0]
        probability = model.predict_proba(sample)[0][1]

        st.metric(
            "Delay Probability",
            f"{probability:.1%}"
        )

        if probability >= 0.75:
            st.error("🔴 High Delay Risk")
        elif probability >= 0.50:
            st.warning("🟡 Medium Delay Risk")
        else:
            st.success("🟢 Low Delay Risk")

    except Exception as e:
        st.error(f"Prediction failed: {e}")