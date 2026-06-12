# modeltrain.py
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, f1_score
import joblib
from Dataprocessor import load_and_preprocess

df = load_and_preprocess("traffic.csv")

X = pd.get_dummies(df[['hour', 'Junction']], columns=['Junction'])
y = df['congestion_level']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Macro F1:", f1_score(y_test, y_pred, average='macro'))
print(classification_report(y_test, y_pred, zero_division=0))

importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
print("Top features:\n", importances.head())

joblib.dump(model, "traffic_model.pkl")
