# app.py
import streamlit as st
import numpy as np
import pandas as pd
import joblib
from Dataprocessor import load_and_preprocess

# -----------------------------
# Load data and model
# -----------------------------
df = load_and_preprocess("traffic.csv")
model = joblib.load("traffic_model.pkl")

# Simulate real-time traffic
df['Vehicles'] = df['Vehicles'] + np.random.randint(-10, 10, size=len(df))
df['Vehicles'] = df['Vehicles'].clip(lower=0)

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("🏙️ Smart Traffic Advisor")

location = st.selectbox("Select Junction", df['Junction'].unique())
hour = st.slider("Select hour of day", 0, 23)
day_of_week = st.selectbox(
    "Day of the week",
    ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
)
day_map = {"Monday":0,"Tuesday":1,"Wednesday":2,"Thursday":3,"Friday":4,"Saturday":5,"Sunday":6}
day_num = day_map[day_of_week]

# -----------------------------
# ML Prediction
# -----------------------------
vehicles_count = int(df[(df['Junction']==location) & (df['hour']==hour)]['Vehicles'].mean())

# Prepare input
X_input = pd.get_dummies(pd.DataFrame({
    'hour':[hour],
    'day_of_week':[day_num],
    'Vehicles':[vehicles_count],
    'Junction':[location]
}))
X_input = X_input.reindex(columns=model.feature_names_in_, fill_value=0)
predicted_congestion = model.predict(X_input)[0]

# Display congestion
if predicted_congestion == 'Low':
    st.success(f"Predicted Congestion: {predicted_congestion}")
elif predicted_congestion == 'Medium':
    st.warning(f"Predicted Congestion: {predicted_congestion}")
else:
    st.error(f"Predicted Congestion: {predicted_congestion}")

# -----------------------------
# Traffic Trend
# -----------------------------
st.subheader("Average traffic per hour")
st.bar_chart(df.groupby('hour')['Vehicles'].mean())

# -----------------------------
# Demo AI Route Suggestions
# -----------------------------
st.subheader("💡 AI Route Suggestion (Demo)")

if st.button("Get Route Suggestion"):
    # Simple demo logic based on congestion
    if predicted_congestion == 'Low':
        suggestion = f"Traffic is smooth at {location}. Normal driving routes are fine."
    elif predicted_congestion == 'Medium':
        suggestion = f"Traffic is moderate at {location}. Consider minor detours or public transport."
    else:  # High or Severe
        suggestion = f"⚠️ Congestion is high at {location}. Take alternate routes or use public transport to reduce emissions."

    st.info(suggestion)

