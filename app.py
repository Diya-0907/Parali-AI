import streamlit as st
import joblib
import pandas as pd
import requests

# -------------------------------
# Load Pipeline Model
# -------------------------------

@st.cache_resource
def load_model():
    return joblib.load("reuse_model_pipeline.pkl")

model = load_model()

API_KEY = st.secrets["WEATHER_API_KEY"]

# -------------------------------
# Weather Function
# -------------------------------

def get_weather(city):
    url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={API_KEY}&units=metric"
    response = requests.get(url)

    if response.status_code != 200:
        return None

    data = response.json()

    temperature = data["main"]["temp"]
    humidity = data["main"]["humidity"]
    wind_speed = data["wind"]["speed"]
    rainfall = data.get("rain", {}).get("1h", 0)

    return temperature, humidity, rainfall, wind_speed


# -------------------------------
# UI
# -------------------------------

st.set_page_config(page_title="Parali AI", page_icon="🌾")

st.title("🌾 Parali AI - Smart Reuse Recommendation")
st.markdown("AI-powered stubble reuse decision system")

soil = st.selectbox("Select Soil Type", ["Sandy", "Loamy", "Clay"])

crop = st.text_input("Enter Crop Type (e.g., rice, maize)").lower()

north_cities = [
    "Amritsar", "Ludhiana", "Patiala", "Jalandhar", "Bathinda",   # Punjab
    "Karnal", "Hisar", "Kurukshetra", "Rohtak", "Panipat", "Ambala"  # Haryana
]

city = st.selectbox("Select Your District/City (Punjab & Haryana Only)", north_cities)

moisture = st.number_input("Moisture Level", min_value=0.0)

st.divider()

if st.button("🔍 Predict Reuse Method"):

    weather = get_weather(city)

    if weather is None:
        st.error("⚠️ Invalid city or API issue.")
    else:
        temperature, humidity, rainfall, wind_speed = weather

        st.info(f"🌡 Temperature: {temperature} °C")
        st.info(f"💧 Humidity: {humidity} %")
        st.info(f"🌧 Rainfall: {rainfall} mm")
        st.info(f"🌬 Wind Speed: {wind_speed} m/s")

        # Create dataframe for pipeline
        input_df = pd.DataFrame([{
            "Soil_Type": soil,
            "Crop_Type": crop,
            "Moisture": moisture,
            "temperature": temperature,
            "humidity": humidity,
            "rainfall": rainfall,
            "Wind_Speed": wind_speed
        }])

        prediction = model.predict(input_df)

        st.success(f"✅ Recommended Reuse Method: {prediction[0]}")

        st.markdown("## 💰 Government Incentive & Cost Analysis")

# State Selection (focused on Punjab & Haryana)
state = st.selectbox("Select Your State", ["Punjab", "Haryana"])

# Land Area Input
area = st.number_input(
    "Enter Land Area (in acres)",
    min_value=0.5,
    max_value=100.0,
    value=1.0,
    step=0.5
)

# Government Scheme Data (Sample Structured Dataset)
govt_schemes = {
    "Punjab": {
        "incentive_per_acre": 1500,
        "machine_subsidy_percent": 50,
        "estimated_burning_fine_per_acre": 500
    },
    "Haryana": {
        "incentive_per_acre": 1200,
        "machine_subsidy_percent": 40,
        "estimated_burning_fine_per_acre": 500
    }
}

# Fetch Scheme Details
scheme = govt_schemes[state]

# Calculations
total_incentive = scheme["incentive_per_acre"] * area
burning_penalty_risk = scheme["estimated_burning_fine_per_acre"] * area

# Optional Estimated Reuse Cost (basic assumption)
estimated_reuse_cost_per_acre = 800
total_reuse_cost = estimated_reuse_cost_per_acre * area

# Net Financial Comparison
net_advantage = total_incentive + burning_penalty_risk - total_reuse_cost

# Display Results
st.success(f"✅ Eligible Government Incentive: ₹{total_incentive:,.0f}")
st.warning(f"⚠️ Estimated Burning Fine Risk: ₹{burning_penalty_risk:,.0f}")
st.info(f"♻️ Estimated Reuse Implementation Cost: ₹{total_reuse_cost:,.0f}")

if net_advantage > 0:
    st.success(f"💸 By choosing reuse, you gain approximately ₹{net_advantage:,.0f} compared to burning.")
else:
    st.error(f"⚠️ Reuse may cost approximately ₹{abs(net_advantage):,.0f} more than burning under current assumptions.")