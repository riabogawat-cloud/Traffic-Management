# modeltrain.py
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib
from Dataprocessor import load_and_preprocess

df = load_and_preprocess("traffic.csv")

# Include hour, day_of_week, Vehicles, and Junction
X = pd.get_dummies(df[['hour','day_of_week','Vehicles','Junction']])
y = df['congestion_level']

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

joblib.dump(model, "traffic_model.pkl")
