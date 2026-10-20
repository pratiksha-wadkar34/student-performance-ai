import streamlit as st
import pandas as pd
import numpy as np
import joblib

from pathlib import Path
from tensorflow.keras.models import load_model


# ==================================================
# 1. PAGE CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="Student Performance AI",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# ==================================================
# 2. PROJECT FILE PATHS
# ==================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "student_model.keras"
SCALER_PATH = BASE_DIR / "scaler.pkl"
FEATURES_PATH = BASE_DIR / "features.pkl"


# ==================================================
# 3. CUSTOM CSS AND ANIMATIONS
# ==================================================

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
    0% {
        background-position: 0% 50%;
    }

    50% {
        background-position: 100% 50%;
    }

    100% {
        background-position: 0% 50%;
    }
}

header[data-testid="stHeader"] {
    background: transparent;
}

.main .block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1200px;
}

.hero {
    background: linear-gradient(
        120deg,
        #4338ca,
        #7c3aed,
        #db2777
    );

    background-size: 200% 200%;
    animation: gradientMove 6s ease infinite;

    color: white;
    padding: 35px 25px;
    border-radius: 24px;
    text-align: center;
    margin-bottom: 25px;

    box-shadow: 0 10px 30px rgba(67, 56, 202, 0.20);
}

.hero h1 {
    color: white;
    font-size: 42px;
    font-weight: 800;
}

.hero p {
    color: white;
    font-size: 17px;
}

@keyframes gradientMove {
    0% {
        background-position: 0% 50%;
    }

    50% {
        background-position: 100% 50%;
    }

    100% {
        background-position: 0% 50%;
    }
}

.section-heading {
    color: #312e81;
    font-size: 28px;
    font-weight: 800;
}

div[data-testid="stForm"] {
    background: rgba(255, 255, 255, 0.88);
    padding: 25px;
    border-radius: 20px;
    border: 1px solid #ddd6fe;
    box-shadow: 0 8px 25px rgba(49, 46, 129, 0.10);
    animation: fadeIn 1s ease;
}

@keyframes fadeIn {
    from {
        opacity: 0;
        transform: translateY(15px);
    }

    to {
        opacity: 1;
        transform: translateY(0);
    }
}

div[data-testid="stFormSubmitButton"] button {
    background: linear-gradient(90deg, #6366f1, #9333ea);
    color: white;
    border: none;
    border-radius: 12px;
    padding: 12px;
    font-size: 17px;
    font-weight: bold;
    transition: all 0.3s ease;
}

div[data-testid="stFormSubmitButton"] button:hover {
    transform: translateY(-3px);
    box-shadow: 0 8px 20px rgba(99, 102, 241, 0.35);
    color: white;
}

.result-card {
    background: rgba(255, 255, 255, 0.93);
    padding: 25px;
    border-radius: 20px;
    text-align: center;
    border: 1px solid #ddd6fe;
    box-shadow: 0 8px 25px rgba(0, 0, 0, 0.08);
    animation: popIn 0.7s ease;
    margin-bottom: 15px;
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

.result-number {
    font-size: 45px;
    font-weight: 800;
    color: #4338ca;
}

.result-label {
    font-size: 18px;
    font-weight: 600;
    color: #374151;
}

.suggestion-card {
    background: rgba(255, 255, 255, 0.94);
    padding: 18px;
    border-radius: 15px;
    border-left: 5px solid #7c3aed;
    margin-bottom: 12px;
    color: #374151;
    animation: fadeIn 0.7s ease;
}

.footer {
    text-align: center;
    padding: 20px;
    color: #6b7280;
    font-size: 14px;
}

</style>
""", unsafe_allow_html=True)


# ==================================================
# 4. LOAD MODEL AND PREPROCESSING FILES
# ==================================================

@st.cache_resource
def load_project_files():

    required_files = [
        MODEL_PATH,
        SCALER_PATH,
        FEATURES_PATH
    ]

    missing_files = [
        path.name
        for path in required_files
        if not path.is_file()
    ]

    if missing_files:
        raise FileNotFoundError(
            "Required files not found: "
            + ", ".join(missing_files)
        )

    model = load_model(
        MODEL_PATH,
        compile=False
    )

    scaler = joblib.load(SCALER_PATH)

    features = joblib.load(FEATURES_PATH)

    return model, scaler, features


# ==================================================
# 5. APPLICATION HEADER
# ==================================================

st.markdown("""
<div class="hero">

    <h1>🎓 Student Performance AI</h1>

    <p>
        Multilayer Neural Network Based
        Student Performance Prediction
    </p>

    <p>
        Predict Performance | Identify Learning Needs |
        Improve Academic Results
    </p>

</div>
""", unsafe_allow_html=True)


# ==================================================
# 6. PROJECT FEATURES
# ==================================================

feature_col1, feature_col2, feature_col3 = st.columns(3)

with feature_col1:
    st.info(
        "**🎯 Grade Prediction**\n\n"
        "Predict the student's final grade using a trained neural network."
    )

with feature_col2:
    st.info(
        "**🔍 Performance Analysis**\n\n"
        "Identify students who may need additional academic support."
    )

with feature_col3:
    st.info(
        "**💡 Improvement Suggestions**\n\n"
        "Get suggestions based on the student's information."
    )

st.divider()


# ==================================================
# 7. LOAD TRAINED MODEL
# ==================================================

try:

    model, scaler, features = load_project_files()

except Exception as error:

    st.error("Unable to load the trained model.")

    st.write(
        "Check that the model files are available in the "
        "same GitHub repository folder as app.py."
    )

    st.code(str(error))

    st.stop()


# ==================================================
# 8. STUDENT INFORMATION FORM
# ==================================================

st.markdown(
    '<div class="section-heading">📝 Student Information</div>',
    unsafe_allow_html=True
)

st.write(
    "Enter the student's information to predict their final grade."
)

with st.form("student_information_form"):

    left_col, right_col = st.columns(2)

    with left_col:

        st.subheader("👤 Personal and Study Details")

        age = st.number_input(
            "🎂 Student Age",
            min_value=15,
            max_value=25,
            value=17,
            step=1
        )

        studytime_label = st.selectbox(
            "📚 Weekly Study Time",
            [
                "Less than 2 hours",
                "2 to 5 hours",
                "5 to 10 hours",
                "More than 10 hours"
            ]
        )

        studytime_map = {
            "Less than 2 hours": 1,
            "2 to 5 hours": 2,
            "5 to 10 hours": 3,
            "More than 10 hours": 4
        }

        studytime = studytime_map[studytime_label]

        failures = st.number_input(
            "📉 Past Class Failures",
            min_value=0,
            max_value=4,
            value=0,
            step=1
        )

    with right_col:

        st.subheader("📘 Academic Details")

        absences = st.number_input(
            "🏫 Number of Absences",
            min_value=0,
            max_value=100,
            value=5,
            step=1
        )

        G1 = st.number_input(
            "📘 First Period Grade (G1)",
            min_value=0,
            max_value=20,
            value=10,
            step=1
        )

        G2 = st.number_input(
            "📗 Second Period Grade (G2)",
            min_value=0,
            max_value=20,
            value=10,
            step=1
        )

    st.write("")

    submitted = st.form_submit_button(
        "🚀 Predict Student Performance",
        use_container_width=True
    )


# ==================================================
# 9. PREDICT STUDENT PERFORMANCE
# ==================================================

if submitted:

    student_information = {
        "age": age,
        "studytime": studytime,
        "failures": failures,
        "absences": absences,
        "G1": G1,
        "G2": G2
    }

    # Read feature names in the same order used during training.

    if isinstance(features, dict):
        feature_names = list(features.keys())

    elif isinstance(features, (list, tuple, pd.Index, np.ndarray)):
        feature_names = list(features)

    else:

        st.error(
            "The saved feature information has an unsupported format."
        )

        st.stop()

    missing_inputs = [
        name
        for name in feature_names
        if name not in student_information
    ]

    if missing_inputs:

        st.error(
            "The model requires additional input features: "
            + ", ".join(missing_inputs)
        )

        st.info(
            "The features used in train_model.py must match "
            "the inputs provided by app.py."
        )

        st.stop()

    input_data = pd.DataFrame(
        [
            [
                student_information[name]
                for name in feature_names
            ]
        ],
        columns=feature_names
    )

    # ----------------------------------------------
    # MODEL PREDICTION
    # ----------------------------------------------

    try:

        with st.spinner(
            "🧠 The neural network is analyzing student information..."
        ):

            scaled_data = scaler.transform(input_data)

            prediction = model.predict(
                scaled_data,
                verbose=0
            )

            predicted_grade = float(
                np.asarray(prediction).reshape(-1)[0]
            )

            predicted_grade = float(
                np.clip(predicted_grade, 0, 20)
            )

    except Exception as error:

        st.error("An error occurred during prediction.")

        st.code(str(error))

        st.stop()

    # ==============================================
    # 10. PERFORMANCE CLASSIFICATION
    # ==============================================

    if predicted_grade < 10:

        performance = "Needs Improvement"
        performance_emoji = "⚠️"

        performance_message = (
            "The student may benefit from additional academic support."
        )

    elif predicted_grade < 15:

        performance = "Average Performance"
        performance_emoji = "📘"

        performance_message = (
            "Consistent study and revision may help improve performance."
        )

    else:

        performance = "Good Performance"
        performance_emoji = "🌟"

        performance_message = (
            "Continue maintaining consistent study habits."
        )

    # ==============================================
    # 11. DISPLAY PREDICTION RESULTS
    # ==============================================

    st.divider()

    st.markdown(
        '<div class="section-heading">📊 Prediction Results</div>',
        unsafe_allow_html=True
    )

    result_col1, result_col2 = st.columns(2)

    with result_col1:

        st.markdown(f"""
        <div class="result-card">

            <div class="result-label">
                🎯 Predicted Final Grade
            </div>

            <div class="result-number">
                {predicted_grade:.2f} / 20
            </div>

            <p>Estimated Final Grade</p>

        </div>
        """, unsafe_allow_html=True)

    with result_col2:

        st.markdown(f"""
        <div class="result-card">

            <div class="result-label">
                📈 Performance Category
            </div>

            <div class="result-number">
                {performance_emoji}
            </div>

            <h3>{performance}</h3>

            <p>{performance_message}</p>

        </div>
        """, unsafe_allow_html=True)

    if predicted_grade < 10:
        st.warning(
            "This prediction indicates that additional academic "
            "support may be helpful."
        )

    elif predicted_grade < 15:
        st.info(
            "There may be opportunities to strengthen academic performance."
        )

    else:
        st.success(
            "The predicted grade falls within the good-performance category."
        )

    # ==============================================
    # 12. GRADE PROGRESS INDICATOR
    # ==============================================

    st.subheader("📊 Predicted Grade Indicator")

    grade_percentage = int(
        (predicted_grade / 20) * 100
    )

    st.progress(grade_percentage)

    st.caption(
        f"Predicted grade relative to the 20-point scale: "
        f"{grade_percentage}%"
    )

    # ==============================================
    # 13. FEATURE-WISE IMPROVEMENT SUGGESTIONS
    # ==============================================

    st.markdown(
        '<div class="section-heading">💡 Improvement Suggestions</div>',
        unsafe_allow_html=True
    )

    suggestions = []

    if studytime <= 2:

        suggestions.append((
            "📚 Study Time",
            "Create a regular study timetable and practise consistently."
        ))

    if G1 < 10:

        suggestions.append((
            "📘 First Period Grade",
            "Review difficult topics from the first grading period."
        ))

    if G2 < 10:

        suggestions.append((
            "📗 Second Period Grade",
            "Focus on topics that were difficult during the second period."
        ))

    if G2 < G1:

        suggestions.append((
            "📉 Academic Trend",
            "The second-period grade is lower than the first. "
            "Review possible reasons with a teacher."
        ))

    elif G2 > G1:

        suggestions.append((
            "📈 Academic Progress",
            "The second-period grade is higher than the first. "
            "Continue the study habits that may have helped."
        ))

    if absences >= 10:

        suggestions.append((
            "🏫 Attendance",
            "Review the reasons for absences and work towards regular attendance."
        ))

    if failures > 0:

        suggestions.append((
            "👨‍🏫 Academic Support",
            "Consider seeking additional guidance in difficult subjects."
        ))

    if not suggestions:

        suggestions.append((
            "🌟 Continue Learning",
            "Maintain consistent study habits, attend classes, "
            "and monitor academic progress."
        ))

    for title, description in suggestions:

        st.markdown(
            f"""
            <div class="suggestion-card">
                <strong>{title}</strong><br>
                {description}
            </div>
            """,
            unsafe_allow_html=True
        )

    # ==============================================
    # 14. STUDENT INFORMATION SUMMARY
    # ==============================================

    st.subheader("📋 Student Information Summary")

    summary_data = pd.DataFrame({
        "Feature": [
            "Student Age",
            "Weekly Study Time",
            "Past Class Failures",
            "Number of Absences",
            "First Period Grade (G1)",
            "Second Period Grade (G2)"
        ],

        "Value": [
            age,
            studytime_label,
            failures,
            absences,
            G1,
            G2
        ]
    })

    st.dataframe(
        summary_data,
        use_container_width=True,
        hide_index=True
    )

    # ==============================================
    # 15. DISCLAIMER
    # ==============================================

    st.caption(
        "Note: The predicted grade is an estimate, not a guaranteed result. "
        "Improvement suggestions are based on simple rules applied to "
        "the entered information."
    )


# ==================================================
# 16. FOOTER
# ==================================================

st.divider()

st.markdown("""
<div class="footer">
    🎓 Student Performance AI<br>
    Built using Python, Streamlit and a Multilayer Neural Network
</div>
""", unsafe_allow_html=True)
