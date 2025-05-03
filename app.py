import streamlit as st
import pickle
import numpy as np
from chatbot import get_response

# Simulate a database (for demo purposes)
USER_CREDENTIALS = {}

# Function to register new user
def sign_up(username, password):
    if username in USER_CREDENTIALS:
        return False  # Username already taken
    USER_CREDENTIALS[username] = password
    return True

# Function to authenticate the user
def authenticate(username, password):
    return USER_CREDENTIALS.get(username) == password

# Load trained diabetes prediction model
with open("diabetes_model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

# Load scaler for data normalization
with open("scaler.pkl", "rb") as scaler_file:
    scaler = pickle.load(scaler_file)

# Streamlit UI
st.title("Diabetes Prediction & Chatbot 🤖")

# Check if user is logged in
if "signed_up" not in st.session_state or not st.session_state.signed_up:
    # Show Sign Up page if not signed up
    st.header("📑 Sign Up Page")

    username = st.text_input("Choose a Username")
    password = st.text_input("Choose a Password", type="password")
    confirm_password = st.text_input("Confirm Password", type="password")

    if st.button("Sign Up"):
        if password != confirm_password:
            st.error("Passwords do not match.")
        elif sign_up(username, password):
            st.session_state.signed_up = True  # Set session state to true
            st.session_state.username = username  # Store username in session
            st.success(f"Welcome, {username}! You have successfully signed up.")
        else:
            st.error("Username already taken. Try a different one.")

elif "signed_up" in st.session_state and st.session_state.signed_up:
    # Show logged-in content (Chatbot or Diabetes Prediction) after sign-up
    st.sidebar.title(f"Welcome {st.session_state.username}")
    option = st.sidebar.selectbox("Choose an option", ["Chat with Bot", "Diabetes Prediction"])

    if option == "Chat with Bot":
        st.header("💬 Chat with the Diabetes Bot")
        user_input = st.text_input("You:", "")

        if st.button("Send"):
            response = get_response(user_input)
            st.text_area("Chatbot:", response, height=100)

    elif option == "Diabetes Prediction":
        st.header("🔍 Diabetes Risk Prediction")

        # Input fields
        pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, step=1)
        glucose = st.number_input("Glucose Level", min_value=0, max_value=300, step=1)
        blood_pressure = st.number_input("Blood Pressure", min_value=0, max_value=200, step=1)
        skin_thickness = st.number_input("Skin Thickness", min_value=0, max_value=100, step=1)
        insulin = st.number_input("Insulin Level", min_value=0, max_value=900, step=1)
        bmi = st.number_input("BMI", min_value=0.0, max_value=70.0, step=0.1)
        dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, step=0.01)
        age = st.number_input("Age", min_value=1, max_value=120, step=1)

        # Predict button
        if st.button("Predict Diabetes"):
            input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]])
            input_data_scaled = scaler.transform(input_data)  # Scale input data
            prediction = model.predict(input_data_scaled)

            if prediction[0] == 1:
                st.error("⚠️ High risk of diabetes. Please consult a doctor.")
            else:
                st.success("✅ Low risk of diabetes. Keep maintaining a healthy lifestyle!")
