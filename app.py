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
else:  # High or Severe
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
    if predicted_congestion == 'Low':
        suggestion = (
            f"Traffic is smooth at {location} around {hour}:00 on {day_of_week}.\n"
            "- Normal driving routes are fine.\n"
            "- No need for detours.\n"
            "- Public transport usage optional."
        )
    elif predicted_congestion == 'Medium':
        suggestion = (
            f"Traffic is moderate at {location} around {hour}:00 on {day_of_week}.\n"
            "- Minor detours can help avoid slower streets.\n"
            "- Public transport such as buses or metro is recommended for faster travel.\n"
            "- Keep an eye on traffic apps for minor updates."
        )
    elif predicted_congestion == 'High':
        suggestion = (
            f"⚠️ High congestion at {location} around {hour}:00 on {day_of_week}.\n"
            "- Avoid main junction roads if possible.\n"
            "- Use alternate routes via side streets to save time.\n"
            "- Consider public transport options:\n"
            "   • Bus routes: Route A, B, C (closest stops near junction)\n"
            "   • Metro/Train lines: Line 1, Line 2\n"
            "- Travel during off-peak hours if possible.\n"
            "- Helps reduce emissions and traffic delays."
        )
    else:  # Severe
        suggestion = (
            f"🚨 Severe congestion at {location} around {hour}:00 on {day_of_week}.\n"
            "- Avoid driving through this junction entirely.\n"
            "- Strongly prefer public transport:\n"
            "   • Metro/Train lines: Line 1, Line 2, Line 3\n"
            "   • Bus services: Fast-track buses near junction\n"
            "- Alternate driving routes using side roads or highways recommended.\n"
            "- If possible, delay travel or work remotely.\n"
            "- Follow official traffi
