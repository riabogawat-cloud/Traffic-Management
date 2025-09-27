# Dataprocessor.py
import pandas as pd

def load_and_preprocess(file_path):
    # Load CSV
    df = pd.read_csv("traffic.csv")
    
    # Convert DateTime column
    df['DateTime'] = pd.to_datetime(df['DateTime'])
    df['hour'] = df['DateTime'].dt.hour
    df['day_of_week'] = df['DateTime'].dt.dayofweek
    
    # Fill missing values
    df.fillna(method='ffill', inplace=True)
    
    # Create congestion_level based on Vehicles
    # Adjust bins to match your dataset
    df['congestion_level'] = pd.cut(
        df['Vehicles'],
        bins=[0, 30, 40, 50, 180],  # Adjust these as per your data distribution
        labels=['Low', 'Medium', 'High', 'Severe']
    )
    
    return df
