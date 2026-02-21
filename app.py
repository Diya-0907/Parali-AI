import streamlit as st
import joblib
import numpy as np

# -------------------------------
# Load Model Package
# -------------------------------

@st.cache_resource
def load_model():
    package = joblib.load("reuse_model_complete.pkl")
    return package

package = load_model()

model = package["model"]
le_soil = package["soil_encoder"]
le_crop = package["crop_encoder"]
le_target = package["target_encoder"]

# -------------------------------
# Streamlit UI
# -------------------------------

st.set_page_config(page_title="Parali AI", page_icon="🌾")

st.title("🌾 Parali AI - Smart Reuse Recommendation System")
st.markdown("### Sustainable Stubble Management Assistant")

st.divider()

# -------------------------------
# User Inputs
# -------------------------------

soil = st.selectbox("Select Soil Type", ["Sandy", "Loamy", "Clay"])

crop = st.text_input("Enter Crop Type (example: rice, wheat, maize)").lower()

moisture = st.number_input("Moisture Level", min_value=0.0, max_value=300.0, step=0.1)

temperature = st.number_input("Temperature (°C)", min_value=-10.0, max_value=60.0, step=0.1)

humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, step=0.1)

rainfall = st.number_input("Rainfall (mm)", min_value=0.0, max_value=500.0, step=0.1)

wind_speed = st.number_input("Wind Speed (m/s)", min_value=0.0, max_value=20.0, step=0.1)

st.divider()

# -------------------------------
# Prediction Button
# -------------------------------

if st.button("🔍 Predict Reuse Method"):

    try:
        # Encode categorical inputs
        soil_encoded = le_soil.transform([soil])[0]
        crop_encoded = le_crop.transform([crop])[0]

        # Create input array
        input_data = np.array([[soil_encoded,
                                crop_encoded,
                                moisture,
                                temperature,
                                humidity,
                                rainfall,
                                wind_speed]])

        # Predict
        prediction = model.predict(input_data)
        result = le_target.inverse_transform(prediction)

        st.success(f"✅ Recommended Reuse Method: {result[0]}")

    except ValueError:
        st.error("❌ Crop type not recognized. Please enter a valid crop from training dataset.")
    except Exception as e:
        st.error(f"⚠️ Error: {e}")