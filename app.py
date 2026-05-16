import streamlit as st
import pandas as pd
import joblib

# ==============================
# LOAD MODEL + DATA
# ==============================
model = joblib.load("earthquake_pipeline.pkl")
df = pd.read_csv("cleaned_earthquake_data.csv")

# ==============================
# CLEAN DATA
# ==============================
df = df[df['place'].notna()]
df = df[~df['place'].str.contains("unknown", case=False, na=False)]
df = df[~df['place'].str.contains(r"\?\?\?", regex=True, na=False)]

# Extract state
df['state'] = df['place'].str.split(',').str[-1].str.strip().str.lower()

# ==============================
# UI
# ==============================
st.title("🌍 Earthquake Prediction & Resource Allocation System")

st.sidebar.header("🔍 Enter Earthquake Details")

# Dropdown for state
state_options = sorted(df['state'].dropna().unique())
state = st.sidebar.selectbox("Select Country/Region", state_options)

# Filter rows for selected state
state_df = df[df['state'] == state]

# Pick a sample row safely
selected_row = state_df.sample(1).iloc[0]

# Auto-fill coordinates
lat = st.sidebar.number_input("Latitude", value=float(selected_row['latitude']))
lon = st.sidebar.number_input("Longitude", value=float(selected_row['longitude']))

depth = st.sidebar.number_input("Depth (km)", value=10.0)
year = st.sidebar.slider("Year", 1990, 2025, 2023)
month = st.sidebar.slider("Month", 1, 12, 5)

# ==============================
# PREDICTION
# ==============================
def predict(lat, lon, depth, year, month, state):

    input_df = pd.DataFrame([{
        'latitude': lat,
        'longitude': lon,
        'depth': depth,
        'year': year,
        'month': month,
        'state': state
    }])

    return model.predict(input_df)[0]

# ==============================
# RISK LOGIC
# ==============================
def get_risk(mag):
    if mag < 4:
        return "Low"
    elif mag < 6:
        return "Medium"
    else:
        return "High"

# ==============================
# RESOURCE ALLOCATION
# ==============================
def allocate_resources(magnitude):

    base = {"water":500,"food":300,"medicine":200,"shelter":100}

    if magnitude < 4:
        factor = 0.5
    elif magnitude < 5:
        factor = 1
    elif magnitude < 6:
        factor = 2
    else:
        factor = 3

    return {
        "water": int(base["water"] * factor),
        "food": int(base["food"] * factor),
        "medicine": int(base["medicine"] * factor),
        "shelter": int(base["shelter"] * factor)
    }

# ==============================
# BUTTON
# ==============================
if st.sidebar.button("Predict"):

    mag = predict(lat, lon, depth, year, month, state)
    risk = get_risk(mag)

    st.subheader("🔮 Prediction Results")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Predicted Magnitude", f"{mag:.2f}")

    with col2:
        st.metric("Risk Level", risk)

    st.subheader("🚚 Resource Allocation")
    st.json(allocate_resources(mag))