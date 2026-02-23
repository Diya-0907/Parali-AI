import streamlit as st
import joblib
import pandas as pd
import requests
import tensorflow as tf
import numpy as np
from PIL import Image

# MUST BE FIRST STREAMLIT COMMAND
st.set_page_config(page_title="Parali AI", page_icon="🌾")

# -------------------------------
# Sidebar Navigation
# -------------------------------
option = st.sidebar.selectbox(
    "Select Module",
    ["Disease Detection", "Reuse Recommendation", "Yield Prediction"]
)


class_names = [
    "BrownRust",   # 0
    "Healthy",     # 1
    "LooseSmut",   # 2
    "Mildew",      # 3
    "Septoria",    # 4
    "YellowRust"   # 5
]

# -------------------------
# 3️⃣ IMAGE PREPROCESSING FUNCTION
# -------------------------
def preprocess_image(image):
    image = image.resize((224, 224))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

# -------------------------------
# Load Pipeline Model
# -------------------------------

@st.cache_resource
def load_reuse_model():
    return joblib.load("reuse_model_pipeline.pkl")

reuse_model = load_reuse_model()

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

@st.cache_resource
def load_yield_model():
    return joblib.load("yield_model.pkl")

yield_model = load_yield_model()


@st.cache_resource
def load_image_model():
    base_model = tf.keras.applications.MobileNetV2(
        weights="imagenet",
        include_top=False,
        input_shape=(224, 224, 3)
    )

    base_model.trainable = False

    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(6, activation='softmax')
    ])

    model.load_weights("wheat_fungal.weights.h5")

    return model

image_model = load_image_model()

# -------------------------
# 4️⃣ STREAMLIT UI CODE
# -------------------------
if option == "Disease Detection":

    st.header("🌾 Wheat Fungal Disease Detection")

    uploaded_file = st.file_uploader(
        "Upload a wheat leaf image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)

        processed_image = preprocess_image(image)

        prediction = image_model.predict(processed_image)

        predicted_class = class_names[np.argmax(prediction)]
        confidence = np.max(prediction) * 100

        st.success(f"Disease: {predicted_class}")
        st.info(f"Confidence: {confidence:.2f}%")


elif option == "Reuse Recommendation":

    st.title("🌾 Parali AI - Smart Reuse Recommendation")
    st.markdown("AI-powered stubble reuse decision system")

    soil = st.selectbox("Select Soil Type", ["Sandy", "Loamy", "Clay"])

    crop = st.text_input("Enter Crop Type (e.g., rice, maize)").lower()

    north_cities = [
        "Amritsar", "Ludhiana", "Patiala", "Jalandhar", "Bathinda",
        "Karnal", "Hisar", "Kurukshetra", "Rohtak", "Panipat", "Ambala"
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

            input_df = pd.DataFrame([{
                "Soil_Type": soil,
                "Crop_Type": crop,
                "Moisture": moisture,
                "temperature": temperature,
                "humidity": humidity,
                "rainfall": rainfall,
                "Wind_Speed": wind_speed
            }])

            prediction = reuse_model.predict(input_df)

            st.success(f"✅ Recommended Reuse Method: {prediction[0]}")

    st.markdown("## 💰 Government Incentive & Cost Analysis")

    state = st.selectbox("Select Your State", ["Punjab", "Haryana"])

    area = st.number_input(
        "Enter Land Area (in acres)",
        min_value=0.5,
        max_value=100.0,
        value=1.0,
        step=0.5
    )

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

    scheme = govt_schemes[state]

    total_incentive = scheme["incentive_per_acre"] * area
    burning_penalty_risk = scheme["estimated_burning_fine_per_acre"] * area

    estimated_reuse_cost_per_acre = 800
    total_reuse_cost = estimated_reuse_cost_per_acre * area

    net_advantage = total_incentive + burning_penalty_risk - total_reuse_cost

    st.success(f"✅ Eligible Government Incentive: ₹{total_incentive:,.0f}")
    st.warning(f"⚠️ Estimated Burning Fine Risk: ₹{burning_penalty_risk:,.0f}")
    st.info(f"♻️ Estimated Reuse Implementation Cost: ₹{total_reuse_cost:,.0f}")

    if net_advantage > 0:
        st.success(f"💸 By choosing reuse, you gain approximately ₹{net_advantage:,.0f} compared to burning.")
    else:
        st.error(f"⚠️ Reuse may cost approximately ₹{abs(net_advantage):,.0f} more than burning under current assumptions.")

elif option == "Yield Prediction":

    st.header("🌾 Crop Yield Prediction (Per Acre)")

    crop = st.selectbox("Select Crop", ["Rice", "Wheat"])
    season = st.selectbox("Select Season", ["Kharif", "Rabi"])
    state = st.text_input("Enter State Name")
    area = st.number_input("Enter Area (in hectare)", min_value=0.1)
    fertilizer = st.number_input("Fertilizer Used (kg)", min_value=0.0)
    pesticide = st.number_input("Pesticide Used (kg)", min_value=0.0)

    if st.button("Predict Yield"):

        input_df = pd.DataFrame([{
            "crop": crop,
            "season": season,
            "state": state,
            "area": area,
            "fertilizer": fertilizer,
            "pesticide": pesticide
        }])

        predicted_yield = yield_model.predict(input_df)[0]

        # Convert tonnes/hectare → quintals/acre
        yield_q_per_acre = (predicted_yield * 10) / 2.47

        # Residue calculation
        RPR = {"Rice": 1.5, "Wheat": 1.2}
        yield_ton_per_acre = yield_q_per_acre / 10
        residue = yield_ton_per_acre * RPR[crop]

        # Revenue calculation (Approx MSP)
        MSP = {"Rice": 2200, "Wheat": 2275}
        revenue = yield_q_per_acre * MSP[crop]

        # Yield Category
        if yield_q_per_acre < 12:
            category = "Low"
        elif yield_q_per_acre < 20:
            category = "Medium"
        else:
            category = "High"

        st.success("Prediction Complete ✅")

        st.write("### 📊 Results:")
        st.write(f"🌾 Predicted Yield: {yield_q_per_acre:.2f} quintals/acre")
        st.write(f"🌿 Estimated Residue: {residue:.2f} tons/acre")
        st.write(f"💰 Estimated Revenue: ₹{revenue:,.2f} per acre")
        st.write(f"📈 Yield Category: {category}")