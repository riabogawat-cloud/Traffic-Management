# Dataprocessor.py
import pandas as pd

def load_and_preprocess(file_path):
    df = pd.read_csv("traffic.csv")

    df['DateTime'] = pd.to_datetime(df['DateTime'])
    df['hour'] = df['DateTime'].dt.hour
    df['day_of_week'] = df['DateTime'].dt.dayofweek

    df.ffill(inplace=True)

    df['congestion_level'] = pd.cut(
        df['Vehicles'],
        bins=[0, 30, 40, 50, 180],
        labels=['Low', 'Medium', 'High', 'Severe']
    )

    return df
