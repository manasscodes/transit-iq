import joblib

model = joblib.load(
    "models/transit_iq_classifier.pkl"
)

print(type(model))
print("Model loaded successfully!")