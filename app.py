import streamlit as st
import pandas as pd
import pickle
import numpy as np
import time

from tensorflow.keras.models import load_model

# -----------------------------
# PAGE CONFIGURATION
# -----------------------------

st.set_page_config(
    page_title="Student Performance AI",
    page_icon="🎓",
    layout="wide"
)

# -----------------------------
# CUSTOM ANIMATED CSS
# -----------------------------

st.markdown("""
<style>

.stApp {
    background: linear-gradient(
        -45deg,
        #e0f2fe,
        #ede9fe,
        #fce7f3,
        #dcfce7
    );
    background-size: 400% 400%;
    animation: gradientBG 15s ease infinite;
}

@keyframes gradientBG {
    0% {background-position: 0% 50%;}
    50% {background-position: 100% 50%;}
    100% {background-position: 0% 50%;}
}

.title-animation {
    font-size: 42px;
    font-weight: 800;
    text-align: center;
    color: #312e81;
    animation: slideDown 1.2s ease;
}

.subtitle {
    text-align: center;
    font-size: 18px;
    color: #334155;
    animation: fadeIn 2s ease;
}

@keyframes slideDown {
    from {
        opacity: 0;
        transform: translateY(-40px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

@keyframes fadeIn {
    from {opacity: 0;}
    to {opacity: 1;}
}

div[data-testid="stForm"] {
    background: rgba(255,255,255,0.85);
    padding: 25px;
    border-radius: 20px;
    border: 1px solid rgba(255,255,255,0.8);
    box-shadow: 0 8px 25px rgba(49,46,129,0.12);
    animation: fadeIn 1.5s ease;
}

div.stButton > button,
div[data-testid="stFormSubmitButton"] > button {
    background: linear-gradient(90deg, #6366f1, #9333ea);
    color: white;
    border: none;
    border-radius: 12px;
    padding: 12px 24px;
    font-weight: bold;
    transition: all 0.3s ease;
}

div.stButton > button:hover,
div[data-testid="stFormSubmitButton"] > button:hover {
    transform: translateY(-3px) scale(1.02);
    box-shadow: 0 8px 20px rgba(99,102,241,0.35);
    color: white;
}

.result-card {
    background: rgba(255,255,255,0.9);
    padding: 25px;
    border-radius: 20px;
    text-align: center;
    box-shadow: 0 8px 25px rgba(0,0,0,0.08);
    animation: popIn 0.7s ease;
}

@keyframes popIn {
    0% {
        opacity: 0;
        transform: scale(0.85);
    }
    100% {
        opacity: 1;
        transform: scale(1);
    }
}

.section-title {
    color: #312e81;
    font-weight: 800;
}

div[data-testid="stMetric"] {
    background: rgba(255,255,255,0.8);
    padding: 18px;
    border-radius: 15px;
    box-shadow: 0 4px 15px rgba(0,0,0,0.05);
}

</style>
""", unsafe_allow_html=True)


# -----------------------------
# LOAD MODEL
# -----------------------------

@st.cache_resource
def load_resources():
    model = load_model("models/student_model.keras")

    with open("models/scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    with open("models/features.pkl", "rb") as f:
        features = pickle.load(f)

    return model, scaler, features


try:
    model, scaler, features = load_resources()

except Exception as e:
    st.error("Model load झाला नाही. आधी train_model.py run कर.")
    st.code(str(e))
    st.stop()


# -----------------------------
# HEADER
# -----------------------------

st.markdown(
    '<div class="title-animation">🎓 Student Performance AI</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Multilayer Neural Network Based Student Performance Prediction'
    '</div>',
    unsafe_allow_html=True
)

st.write("")
st.divider()


# -----------------------------
# STUDENT FORM
# -----------------------------

st.markdown(
    '<h2 class="section-title">📝 Student Information</h2>',
    unsafe_allow_html=True
)

col1, col2 = st.columns(2)

with st.form("student_form"):

    with col1:
        age = st.number_input(
            "🎂 Student Age",
            min_value=15,
            max_value=25,
            value=17
        )

        studytime = st.selectbox(
            "📚 Weekly Study Time",
            options=[1, 2, 3, 4],
            format_func=lambda x: {
                1: "Less than 2 hours",
                2: "2–5 hours",
                3: "5–10 hours",
                4: "More than 10 hours"
            }[x]
        )

        failures = st.number_input(
            "📉 Past Class Failures",
            min_value=0,
            max_value=4,
            value=0
        )

    with col2:
        absences = st.number_input(
            "🏫 Number of Absences",
            min_value=0,
            max_value=100,
            value=5
        )

        g1 = st.number_input(
            "📘 First Period Grade (G1)",
            min_value=0,
            max_value=20,
            value=10
        )

        g2 = st.number_input(
            "📗 Second Period Grade (G2)",
            min_value=0,
            max_value=20,
            value=10
        )

    submitted = st.form_submit_button(
        "🚀 Predict Student Performance",
        use_container_width=True
    )


# -----------------------------
# PREDICTION
# -----------------------------

if submitted:

    with st.spinner("🧠 Neural Network is analyzing student data..."):

        student_data = pd.DataFrame(
            [[age, studytime, failures, absences, g1, g2]],
            columns=features
        )

        scaled_data = scaler.transform(student_data)

        prediction = model.predict(
            scaled_data,
            verbose=0
        )[0][0]

        predicted_grade = float(np.clip(prediction, 0, 20))

        time.sleep(0.8)

    st.divider()

    st.markdown(
        '<h2 class="section-title">📊 Prediction Results</h2>',
        unsafe_allow_html=True
    )

    if predicted_grade < 10:
        category = "Needs Improvement"
        color = "#dc2626"
        message = "Additional academic support may be helpful."

    elif predicted_grade < 15:
        category = "Average Performance"
        color = "#d97706"
        message = "Further improvement may help strengthen performance."

    else:
        category = "Good Performance"
        color = "#15803d"
        message = "Continue consistent study habits."

    st.markdown(
        f"""
        <div class="result-card">
            <h2 style="color:#312e81;">Predicted Final Grade</h2>
            <h1 style="color:{color};font-size:55px;">
                {predicted_grade:.2f} / 20
            </h1>
            <h3 style="color:{color};">{category}</h3>
            <p>{message}</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.write("")

    # Animated progress indicator
    st.write("### 📈 Predicted Grade Indicator")

    progress = int(predicted_grade / 20 * 100)
    st.progress(progress)

    st.caption(f"Predicted grade relative to the 20-point scale: {progress}%")

    # -----------------------------
    # IMPROVEMENT SUGGESTIONS
    # -----------------------------

    st.markdown(
        '<h2 class="section-title">💡 Improvement Suggestions</h2>',
        unsafe_allow_html=True
    )

    suggestions = []

    if studytime <= 2:
        suggestions.append(
            ("📚 Study Time",
             "Create a regular study timetable and practise consistently.")
        )

    if g1 < 10 or g2 < 10:
        suggestions.append(
            ("📝 Previous Grades",
             "Revise difficult topics and practise examination questions.")
        )

    if absences > 10:
        suggestions.append(
            ("🏫 Attendance",
             "Review the reasons for absences and work towards regular attendance.")
        )

    if failures > 0:
        suggestions.append(
            ("👨‍🏫 Academic Support",
             "Consider additional guidance from teachers in difficult subjects.")
        )

    if not suggestions:
        suggestions.append(
            ("🌟 Keep Improving",
             "Maintain consistent study habits and monitor academic progress.")
        )

    for title, description in suggestions:
        st.success(f"**{title}** — {description}")

    if predicted_grade < 10:
        st.warning(
            "⚠️ This prediction suggests the student may benefit "
            "from additional academic support."
        )

    st.caption(
        "Predictions are estimates, not guaranteed results. "
        "Improvement suggestions use simple rules based on the entered data."
    )

st.divider()

st.markdown(
    "<center>🎓 Student Performance AI | Built with Python & Streamlit</center>",
    unsafe_allow_html=True
)