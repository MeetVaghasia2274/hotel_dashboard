import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib
import os

# ----------------------------
# 1️⃣ Create model folder
# ----------------------------
os.makedirs("model", exist_ok=True)

# ----------------------------
# 2️⃣ Load dataset
# ----------------------------
df = pd.read_csv("data/hotel_booking.csv")

# Remove rows with missing target
df = df.dropna(subset=["is_canceled"])

# ----------------------------
# 3️⃣ Select only the columns we will use in app
# ----------------------------
selected_cols = [
    "lead_time",
    "adr",
    "adults",
    "children",
    "stays_in_weekend_nights",
    "customer_type",
    "deposit_type",
    "market_segment"
]

df = df[selected_cols + ["is_canceled"]]

# ----------------------------
# 4️⃣ Encode categorical columns
# ----------------------------
categoricals = ["customer_type", "deposit_type", "market_segment"]
encoders = {}

for col in categoricals:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

# Save encoders for later use in Streamlit
joblib.dump(encoders, "model/encoders.pkl")

# ----------------------------
# 5️⃣ Train-test split
# ----------------------------
X = df.drop(columns=["is_canceled"])
y = df["is_canceled"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ----------------------------
# 6️⃣ Train model
# ----------------------------
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# ----------------------------
# 7️⃣ Save model
# ----------------------------
joblib.dump(model, "model/hotel_model.pkl")
print("✅ Model trained and saved successfully!")
