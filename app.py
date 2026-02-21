import streamlit as st
import joblib
import numpy as np
import requests

# -------------------------------
# Load Model
# -------------------------------

@st.cache_resource
def load_model():
    return joblib.load("reuse_model_complete.pkl")

package = load_model()

model = package["model"]
le_soil = package["soil_encoder"]
le_crop = package["crop_encoder"]
le_target = package["target_encoder"]

# -------------------------------
# Get API Key from Secrets
# -------------------------------

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

soil = st.selectbox("Select Soil Type", ["Sandy", "Loamy", "Clay"])
crop = st.text_input("Enter Crop Type").lower()
city = st.text_input("Enter Your City")

moisture = st.number_input("Moisture Level", min_value=0.0)

st.divider()

if st.button("🔍 Predict Reuse Method"):

    weather = get_weather(city)

    if weather is None:
        st.error("⚠️ Invalid city name or API issue.")
    else:
        temperature, humidity, rainfall, wind_speed = weather

        st.info(f"🌡 Temperature: {temperature} °C")
        st.info(f"💧 Humidity: {humidity} %")
        st.info(f"🌧 Rainfall: {rainfall} mm")
        st.info(f"🌬 Wind Speed: {wind_speed} m/s")

        try:
            soil_encoded = le_soil.transform([soil])[0]
            crop_encoded = le_crop.transform([crop])[0]

            input_data = np.array([[soil_encoded,
                                    crop_encoded,
                                    moisture,
                                    temperature,
                                    humidity,
                                    rainfall,
                                    wind_speed]])

            prediction = model.predict(input_data)
            result = le_target.inverse_transform(prediction)

            st.success(f"✅ Recommended Reuse Method: {result[0]}")

        except:
            st.error("❌ Crop type not recognized.")