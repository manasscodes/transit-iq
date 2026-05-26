import streamlit as st
import pandas as pd
import joblib

# =====================================================
# CONFIG
# =====================================================

st.set_page_config(
    page_title="TransitIQ",
    page_icon="🚌",
    layout="wide"
)

# =====================================================
# LOAD MODEL
# =====================================================

@st.cache_resource
def load_model():
    return joblib.load("models/transit_iq_classifier.pkl")

model = load_model()

# =====================================================
# HEADER
# =====================================================

st.title("🚌 TransitIQ")

st.markdown("""
### AI-Powered Public Transport Delay Intelligence

Predict the probability of transport delays using:

- 🌦️ Weather conditions
- 🚦 Traffic congestion
- 📅 Scheduling information
- 🎉 Event impact
- 🚌 Transport characteristics
""")

st.markdown("---")

# =====================================================
# SIDEBAR INPUTS
# =====================================================

with st.sidebar:

    st.header("⚙️ Trip Information")

    transport_type = st.selectbox(
        "Transport Type",
        ["Bus", "Metro", "Train", "Tram"]
    )

    weather_condition = st.selectbox(
        "Weather Condition",
        [
            "Clear",
            "Cloudy",
            "Fog",
            "Rain",
            "Snow",
            "Storm"
        ]
    )

    season = st.selectbox(
        "Season",
        [
            "Spring",
            "Summer",
            "Autumn",
            "Winter"
        ]
    )

    st.divider()

    st.header("🌦️ Weather Metrics")

    temperature = st.slider(
        "Temperature (°C)",
        -5,
        40,
        20
    )

    humidity = st.slider(
        "Humidity (%)",
        0,
        100,
        70
    )

    wind_speed = st.slider(
        "Wind Speed (km/h)",
        0,
        60,
        15
    )

    precipitation = st.slider(
        "Precipitation (mm)",
        0,
        20,
        5
    )

    st.divider()

    st.header("🚦 Traffic & Operations")

    traffic_congestion = st.slider(
        "Traffic Congestion Index",
        0,
        100,
        50
    )

    trip_duration = st.slider(
        "Scheduled Trip Duration (mins)",
        5,
        120,
        45
    )

    departure_hour = st.slider(
        "Departure Hour",
        0,
        23,
        8
    )

    peak_hour = st.selectbox(
        "Peak Hour",
        [0, 1],
        format_func=lambda x:
            "Yes" if x == 1 else "No"
    )

    holiday = st.selectbox(
        "Holiday",
        [0, 1],
        format_func=lambda x:
            "Yes" if x == 1 else "No"
    )

    st.divider()

    predict_button = st.button(
        "🔮 Predict Delay",
        use_container_width=True
    )

# =====================================================
# MAIN CONTENT
# =====================================================

col1, col2 = st.columns([2, 1])

with col1:

    st.subheader("📊 Delay Prediction")

    if predict_button:

        sample = pd.DataFrame({
            "transport_type": [transport_type],
            "route_id": ["Route_1"],
            "origin_station": ["Station_1"],
            "destination_station": ["Station_2"],
            "weather_condition": [weather_condition],
            "temperature_C": [temperature],
            "humidity_percent": [humidity],
            "wind_speed_kmh": [wind_speed],
            "precipitation_mm": [precipitation],
            "event_type": ["No_Event"],
            "event_attendance_est": [0],
            "traffic_congestion_index": [traffic_congestion],
            "holiday": [holiday],
            "peak_hour": [peak_hour],
            "weekday": [2],
            "season": [season],
            "day": [10],
            "day_of_week": [2],
            "is_weekend": [0],
            "departure_hour": [departure_hour],
            "departure_minute": [15],
            "arrival_hour": [departure_hour + 1],
            "scheduled_trip_duration": [trip_duration]
        })

        prediction = model.predict(sample)[0]
        probability = model.predict_proba(sample)[0][1]

        st.metric(
            label="Delay Probability",
            value=f"{probability:.1%}"
        )

        if probability >= 0.75:
            st.error("🔴 High Delay Risk")

        elif probability >= 0.50:
            st.warning("🟡 Medium Delay Risk")

        else:
            st.success("🟢 Low Delay Risk")

        st.markdown("### Prediction Result")

        if prediction == 1:
            st.write(
                "The model predicts that this trip is likely to be delayed."
            )
        else:
            st.write(
                "The model predicts that this trip is likely to arrive on time."
            )

with col2:

    st.subheader("📌 Input Summary")

    st.info(
        f"""
Transport: **{transport_type}**

Weather: **{weather_condition}**

Traffic Index: **{traffic_congestion}**

Trip Duration: **{trip_duration} mins**

Peak Hour: **{'Yes' if peak_hour else 'No'}**

Holiday: **{'Yes' if holiday else 'No'}**
"""
    )

# =====================================================
# INSIGHTS SECTION
# =====================================================

st.markdown("---")

st.subheader("📈 Model Insights")

st.write("""
Top factors used by the model include:

- Temperature
- Traffic Congestion
- Precipitation
- Wind Speed
- Humidity
- Scheduled Trip Duration
- Departure Time

These features were identified through Random Forest feature importance analysis.
""")

# =====================================================
# FOOTER
# =====================================================

st.markdown("---")

st.caption(
    "TransitIQ • End-to-End Machine Learning Project • Built with Streamlit & Scikit-Learn"
)