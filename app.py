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

city = st.text_input("Enter Your City")

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