import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

@st.cache_resource
def load_model():
    model_path = hf_hub_download(
        repo_id="bhatiashikha24/tourism-package-model",
        filename="best_tourism_package_model_v1.joblib"
    )
    return joblib.load(model_path)

model = load_model()

# Streamlit UI for Machine Failure Prediction
st.title("Toursim Package Prediction App")
st.write("""
This app predicts whether a customer is likely to **purchase a tourism package**
    based on their profile and interaction details.
""")

st.header(" Enter Customer Details")

# User input

col1, col2 = st.columns(2)

with col1:
    TypeofContact = st.selectbox("Type of Contact", ["Self Enquiry", "Company Invited"])
    CityTier = st.selectbox("City Tier", [1, 2, 3])
    Occupation = st.selectbox(
        "Occupation",
        ["Salaried", "Small Business", "Large Business", "Free Lancer"]
    )
    Gender = st.selectbox("Gender", ["Male", "Female"])
    MaritalStatus = st.selectbox("Marital Status", ["Married", "Unmarried"])
    Passport = st.selectbox("Passport", ["Yes", "No"])
    OwnCar = st.selectbox("Own Car", ["Yes", "No"])

with col2:
    ProductPitched = st.selectbox(
        "Product Pitched",
        ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"]
    )
    Designation = st.selectbox(
        "Designation",
        ["Executive", "Senior Executive", "Manager", "AVP", "VP"]
    )
    PreferredPropertyStar = st.selectbox(
        "Preferred Property Star", [3, 4, 5]
    )
    Age = st.number_input("Age", min_value=18, max_value=100, value=30)
    MonthlyIncome = st.number_input("Monthly Income", min_value=0, value=30000)

DurationOfPitch = st.number_input("Duration of Pitch (minutes)", min_value=0, value=10)
NumberOfPersonVisiting = st.number_input("Number of Persons Visiting", min_value=1, value=2)
NumberOfFollowups = st.number_input("Number of Follow-ups", min_value=0, value=1)
NumberOfTrips = st.number_input("Number of Trips", min_value=0, value=1)
PitchSatisfactionScore = st.slider("Pitch Satisfaction Score", 1, 5, 3)
NumberOfChildrenVisiting = st.number_input("Number of Children Visiting", min_value=0, value=0)


# Assemble input into DataFrame
input_data = pd.DataFrame([{
    "TypeofContact": TypeofContact,
    "CityTier": CityTier,
    "Occupation": Occupation,
    "Gender": Gender,
    "ProductPitched": ProductPitched,
    "MaritalStatus": MaritalStatus,
    "Passport": Passport,
    "OwnCar": OwnCar,
    "Designation": Designation,
    "PreferredPropertyStar": PreferredPropertyStar,
    "Age": Age,
    "DurationOfPitch": DurationOfPitch,
    "NumberOfPersonVisiting": NumberOfPersonVisiting,
    "NumberOfFollowups": NumberOfFollowups,
    "NumberOfTrips": NumberOfTrips,
    "PitchSatisfactionScore": PitchSatisfactionScore,
    "NumberOfChildrenVisiting": NumberOfChildrenVisiting,
    "MonthlyIncome": MonthlyIncome
}])


if st.button("Predict Package Purchase"):
    prediction = model.predict(input_data)[0]

    if prediction == 1:
        st.success("✅ Customer is **likely to purchase** the tourism package.")
    else:
        st.warning("❌ Customer is **unlikely to purchase** the tourism package.")
