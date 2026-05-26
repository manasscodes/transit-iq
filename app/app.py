import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="TransitIQ",
    page_icon="🚌",
    layout="centered"
)

@st.cache_resource
def load_model():
    return joblib.load("models/transit_iq_classifier.pkl")

model = load_model()

st.title("🚌 TransitIQ")
st.subheader("AI-Powered Public Transport Delay Predictor")

st.write(
    """
Predict the probability that a public transport trip will be delayed
based on weather, traffic, event, and scheduling information.
"""
)

st.markdown("---")

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
    min_value=0,
    max_value=100,
    value=50
)

temperature = st.slider(
    "Temperature (°C)",
    min_value=-5,
    max_value=40,
    value=20
)

trip_duration = st.slider(
    "Scheduled Trip Duration (minutes)",
    min_value=5,
    max_value=120,
    value=45
)

season = st.selectbox(
    "Season",
    ["Spring", "Summer", "Autumn", "Winter"]
)

if st.button("Predict Delay"):

    sample = pd.DataFrame({
        "transport_type": [transport_type],
        "route_id": ["Route_1"],
        "origin_station": ["Station_1"],
        "destination_station": ["Station_2"],
        "weather_condition": [weather_condition],
        "temperature_C": [temperature],
        "humidity_percent": [70],
        "wind_speed_kmh": [15],
        "precipitation_mm": [5],
        "event_type": ["No_Event"],
        "event_attendance_est": [0],
        "traffic_congestion_index": [traffic_congestion],
        "holiday": [0],
        "peak_hour": [1],
        "weekday": [2],
        "season": [season],
        "day": [10],
        "day_of_week": [2],
        "is_weekend": [0],
        "departure_hour": [8],
        "departure_minute": [15],
        "arrival_hour": [9],
        "scheduled_trip_duration": [trip_duration]
    })

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

    st.subheader("Prediction Details")

    st.write(
        f"Model Prediction: {'Delayed' if prediction == 1 else 'On Time'}"
    )

st.markdown("---")

st.caption(
    "TransitIQ • Machine Learning Project • Built with Streamlit & Scikit-Learn"
)