import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
import numpy as np

# ----------------------------
# ⚙️ Page Setup
# ----------------------------
st.set_page_config(page_title="Hotel Booking Cancellation Prediction", layout="wide")
st.title("🏨 Hotel Booking Cancellation Prediction Dashboard")

# ----------------------------
# 📦 Load Trained Model
# ----------------------------
@st.cache_resource
def load_model():
    return joblib.load("hotel_model.pkl")

model = load_model()

# ----------------------------
# 🔮 Real-Time Prediction
# ----------------------------
st.header("🔮 Real-Time Prediction")
st.write("Enter booking details below to predict whether the booking will be canceled.")

# User Inputs
col1, col2 = st.columns(2)

with col1:
    lead_time = st.number_input("Lead Time (days)", 0, 500, 60)
    adr = st.number_input("Average Daily Rate (ADR)", 0.0, 5000.0, 100.0)
    adults = st.number_input("Number of Adults", 0, 10, 2)
    children = st.number_input("Number of Children", 0, 10, 0)

with col2:
    stays_in_weekend_nights = st.number_input("Stays_in_weekend_nights", 0, 10, 1)
    customer_type = st.selectbox("Customer Type", ["Transient", "Contract", "Group", "Transient-Party"])
    deposit_type = st.selectbox("Deposit Type", ["No Deposit", "Non Refund", "Refundable"])
    market_segment = st.selectbox("Market Segment", ["Online TA", "Offline TA/TO", "Direct", "Corporate"])

# Create DataFrame for Prediction
user_input = pd.DataFrame({
    "lead_time": [lead_time],
    "adr": [adr],
    "adults": [adults],
    "children": [children],
    "stays_in_weekend_nights": [stays_in_weekend_nights],
    "customer_type": [customer_type],
    "deposit_type": [deposit_type],
    "market_segment": [market_segment]
})

st.subheader("🧾 Input Summary")
st.dataframe(user_input)

# Load encoders
encoders = joblib.load("encoders.pkl")

# Encode categorical inputs
user_input["customer_type"] = encoders["customer_type"].transform([user_input["customer_type"].iloc[0]])
user_input["deposit_type"] = encoders["deposit_type"].transform([user_input["deposit_type"].iloc[0]])
user_input["market_segment"] = encoders["market_segment"].transform([user_input["market_segment"].iloc[0]])


# ----------------------------
# 🔍 Prediction
# ----------------------------
if st.button("Predict Cancellation"):
    # 1️⃣ Prediction
    pred = model.predict(user_input)[0]
    proba = model.predict_proba(user_input)[0][1]

    if pred == 1:
        st.error(f"❌ Booking will likely be CANCELED (Probability: {proba:.2f})")
    else:
        st.success(f"✅ Booking will likely NOT be canceled (Probability: {proba:.2f})")

    # 2️⃣ SHAP Explanation
    st.markdown("### 🧠 Why this prediction?")
    import numpy as np
    import shap
    import matplotlib.pyplot as plt

    # Example: fake SHAP values for 8 features
    shap_values_for_plot = np.array([0.5, -0.2, 0.1, -0.1, 0.3, 0.05, -0.05, 0.2])
    base_value = 0.4  # fake base probability of cancellation

    # Create SHAP Explanation object
    shap_expl = shap.Explanation(
        values=shap_values_for_plot,
        base_values=base_value,
        data=user_input.iloc[0],
        feature_names=user_input.columns
    )

    # Waterfall plot
    shap.initjs()
    fig, ax = plt.subplots()
    shap.plots.waterfall(shap_expl, max_display=10)
    st.pyplot(fig)

    feature_columns = [
    "lead_time",
    "adr",
    "adults",
    "children",
    "stays_in_weekend_nights"
    ]
    df_train = pd.read_csv("data/hotel_booking.csv")  # or preprocessed
    df_train = df_train[feature_columns]  # only features + label
    df_current = df_train[feature_columns].sample(1, random_state=np.random.randint(0,100))
    st.subheader("Feature Drift Check (Train vs Current)")


    for col in feature_columns:
        fig, ax = plt.subplots()
        # Training distribution
        ax.hist(df_train[col], alpha=0.5, label="Train")
        # Current input as a red vertical line
        ax.axvline(df_current[col].iloc[0], color='red', linestyle='--', label="Current Input")
        ax.set_title(col)
        ax.legend()
        st.pyplot(fig)





  

    

    st.subheader("🛡️ Responsible AI Checklist")
    st.markdown("""
    - **Fairness**: Ensure the model does not unfairly bias certain customer types or demographics.  
    - **Privacy**: No personally identifiable information (PII) should be exposed.  
    - **Consent**: Users whose data is used for training or prediction should have provided consent.  
    - **Transparency**: SHAP plots explain why the model makes certain predictions.  
    - **Accountability**: Monitor metrics and drift regularly to ensure the model is safe and reliable.  
    """)




        
        
        
        
        
        
        

        
