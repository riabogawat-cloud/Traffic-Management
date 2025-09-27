# app.py
import streamlit as st
import numpy as np
import pandas as pd
import joblib
from Dataprocessor import load_and_preprocess
from openai import OpenAI

# -----------------------------
# OPENAI API KEY
# -----------------------------
OPENAI_API_KEY = "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"
client = OpenAI(api_key=OPENAI_API_KEY)

# -----------------------------
# Load data and model
# -----------------------------
df = load_and_preprocess("traffic.csv")
model = joblib.load("traffic_model.pkl")

# Simulate real-time traffic
df['Vehicles'] = df['Vehicles'] + np.random.randint(-10,10,size=len(df))
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
# Get average vehicles for selected junction and hour
vehicles_count = int(df[(df['Junction']==location) & (df['hour']==hour)]['Vehicles'].mean())

# Prepare input
X_input = pd.get_dummies(pd.DataFrame({
    'hour':[hour],
    'day_of_week':[day_num],
    'Vehicles':[vehicles_count],
    'Junction':[location]
}))

# Reindex to match training features
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
# AI Route Suggestions
# -----------------------------
if st.button("Get Route Suggestion"):
    prompt = f"""
    The traffic at {location} around {hour}:00 on {day_of_week} is {predicted_congestion}.
    Suggest optimal driving routes OR public transport options.
    If congestion is High or Severe, suggest alternate routes to reduce traffic and emissions.
    """
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role":"user","content":prompt}],
        max_tokens=500
    )
    suggestion = response.choices[0].message.content
    st.info(f"💡 AI Suggestion: {suggestion}")
    
    if predicted_congestion in ['High','Severe']:
        st.warning("⚠️ Congestion is high — consider alternate routes or public transport.")


