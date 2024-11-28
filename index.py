import streamlit as st
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import time

try:
    df = pd.read_csv(r"./data.csv")
except FileNotFoundError:
    st.error("Unable to locate the dataset. Ensure the file `diabetes.csv` is in the correct location.")
    st.stop()

# Custom CSS Styling
st.markdown("""
<style>
body {
    background: linear-gradient(to bottom, #f0f4c3, #c8e6c9); /* Soft green gradient background */
    font-family: 'Roboto', sans-serif; /* Modern font family */
}

.container {
    background: #ffffff;
    border-radius: 15px;
    box-shadow: 0 5px 15px rgba(0, 0, 0, 0.2);
    padding: 25px;
    margin: 20px 0;
}

.action-button {
    background: #388e3c; /* Vibrant green */
    color: white;
    padding: 12px 25px;
    border: none;
    border-radius: 25px;
    cursor: pointer;
    font-size: 16px;
    transition: all 0.3s ease;
}

.action-button:hover {
    background: #2e7d32; /* Darker green on hover */
    transform: scale(1.05);
}

h1, h2, h3, h4 {
    color: #2e7d32; /* Green for headings */
}

.success {
    color: #1b5e20; /* Deep green for success messages */
    font-weight: bold;
}

.warning {
    color: #d32f2f; /* Red for warnings */
    font-weight: bold;
}

footer {
    margin-top: 40px;
    text-align: center;
    font-size: 14px;
    color: #757575; /* Muted grey for footer */
}

a {
    color: #388e3c; /* Green links */
    text-decoration: none;
    font-weight: bold;
}

a:hover {
    text-decoration: underline;
}
</style>
""", unsafe_allow_html=True)

st.markdown("<h1>🩺 Diabetes Risk Assessment</h1>", unsafe_allow_html=True)
st.markdown("<p>Analyze your health statistics to check your diabetes risk level.</p>", unsafe_allow_html=True)

st.markdown("<div class='container'><h3>Input Your Health Details</h3></div>", unsafe_allow_html=True)

# Function to Collect User Data
def collect_user_data():
    col1, col2, col3 = st.columns(3)

    with col1:
        pregnancies = st.slider('Pregnancy Count', 0, 17, 3, help="Number of pregnancies you've had.")
        blood_pressure = st.slider('Blood Pressure (mm Hg)', 0, 122, 70, help="Your diastolic blood pressure.")
        dpf = st.slider('Diabetes Pedigree Function', 0.0, 2.4, 0.47, 0.01, help="Family history score for diabetes risk.")
    with col2:
        glucose = st.slider('Glucose Concentration (mg/dL)', 0, 200, 120, help="Glucose level after a glucose test.")
        skin_thickness = st.slider('Skin Fold Thickness (mm)', 0, 100, 20, help="Measurement of the triceps skin fold.")
        age = st.slider('Age', 21, 88, 33, help="Your age in years.")
    with col3:
        insulin = st.slider('Insulin Level (IU/mL)', 0, 846, 79, help="Concentration of insulin in your blood.")
        bmi = st.slider('BMI (kg/m²)', 0.0, 67.0, 20.0, 0.1, help="Body mass index based on weight and height.")

    return pd.DataFrame({
        'Pregnancies': [pregnancies],
        'Glucose': [glucose],
        'BloodPressure': [blood_pressure],
        'SkinThickness': [skin_thickness],
        'Insulin': [insulin],
        'BMI': [bmi],
        'DiabetesPedigreeFunction': [dpf],
        'Age': [age]
    })

user_input = collect_user_data()

st.markdown("<div class='container'><h3>Your Input Data</h3></div>", unsafe_allow_html=True)
st.write(user_input)

# Model Preparation
x = df.drop(['Outcome'], axis=1)
y = df['Outcome']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

model = RandomForestClassifier(random_state=42)
model.fit(x_train, y_train)

# Prediction Button and Logic
if st.button('🔍 Analyze My Risk', key="predict-button"):
    with st.spinner('Processing your data... Please wait!'):
        time.sleep(1)
    prediction = model.predict(user_input)[0]
    model_accuracy = accuracy_score(y_test, model.predict(x_test)) * 100

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("<h2>🚦 Prediction Outcome</h2>", unsafe_allow_html=True)
    if prediction == 0:
        st.markdown("<p class='success'>✅ Low Risk: You are not likely diabetic.</p>", unsafe_allow_html=True)
    else:
        st.markdown("<p class='warning'>⚠️ High Risk: You may have a risk of diabetes!</p>", unsafe_allow_html=True)

    st.markdown(f"<p><b>Model Performance:</b> {model_accuracy:.2f}% accuracy</p>", unsafe_allow_html=True)
else:
    st.markdown("<h3>👈 Adjust the sliders and click 'Analyze My Risk'</h3>", unsafe_allow_html=True)

st.markdown("""
<footer>
    Created with 💡 by a Streamlit Enthusiast. Powered by <a href="https://streamlit.io" target="_blank">Streamlit</a>.
</footer>
""", unsafe_allow_html=True)
