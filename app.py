import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from tensorflow.keras.models import load_model


# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Student Performance AI",
    page_icon="🎓",
    layout="wide"
)


# ==========================================
# FILE PATHS
# ==========================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "student_model.keras"
SCALER_PATH = BASE_DIR / "scaler.pkl"
FEATURES_PATH = BASE_DIR / "features.pkl"


# ==========================================
# CUSTOM CSS
# ==========================================

st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(
            135deg,
            #e0f2fe,
            #f8fafc,
            #ede9fe
        );
    }

    .hero {
        background: linear-gradient(
            120deg,
            #2563eb,
            #7c3aed
        );
        padding: 35px;
        border-radius: 20px;
        text-align: center;
        color: white;
        margin-bottom: 25px;
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.15);
    }

    .hero h1 {
        font-size: 42px;
        margin-bottom: 12px;
        color: white;
    }

    .hero p {
        font-size: 18px;
        margin: 8px 0;
        color: white;
    }

    .section-card {
        background: white;
        padding: 22px;
        border-radius: 16px;
        margin-bottom: 18px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.07);
    }

    .result-card {
        background: linear-gradient(
            120deg,
            #dbeafe,
            #ede9fe
        );
        padding: 25px;
        border-radius: 18px;
        text-align: center;
        margin-top: 20px;
        border: 1px solid #c4b5fd;
    }

    .result-number {
        font-size: 48px;
        font-weight: bold;
        color: #4338ca;
    }

    .small-note {
        color: #475569;
        font-size: 14px;
    }

    div.stButton > button {
        width: 100%;
        border-radius: 10px;
        height: 48px;
        background: linear-gradient(
            90deg,
            #2563eb,
            #7c3aed
        );
        color: white;
        font-weight: bold;
        border: none;
    }

    div.stButton > button:hover {
        color: white;
        border: 1px solid #4338ca;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# ==========================================
# HEADER
# ==========================================

st.markdown(
    """
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
    """,
    unsafe_allow_html=True
)


# ==========================================
# LOAD MODEL AND SUPPORTING FILES
# ==========================================

@st.cache_resource
def load_project_files():

    missing_files = []

    for file_path in [
        MODEL_PATH,
        SCALER_PATH,
        FEATURES_PATH
    ]:
        if not file_path.exists():
            missing_files.append(file_path.name)

    if missing_files:
        raise FileNotFoundError(
            "Missing required file(s): "
            + ", ".join(missing_files)
        )

    model = load_model(
        str(MODEL_PATH),
        compile=False
    )

    scaler = joblib.load(SCALER_PATH)

    features = joblib.load(FEATURES_PATH)

    return model, scaler, features


try:
    model, scaler, features = load_project_files()

except Exception as error:
    st.error("Unable to load the project files.")

    st.write(
        "Make sure these files are available in the "
        "same GitHub folder as app.py:"
    )

    st.code(
        """
app.py
student_model.keras
scaler.pkl
features.pkl
requirements.txt
        """
    )

    st.error(f"Error details: {error}")

    st.stop()


# ==========================================
# PREPARE FEATURE NAMES
# ==========================================

if isinstance(features, dict):
    feature_names = list(features.keys())

elif isinstance(
    features,
    (list, tuple, pd.Index, np.ndarray)
):
    feature_names = list(features)

else:
    st.error(
        "The format of features.pkl is not supported. "
        "Please check how features.pkl was created."
    )
    st.stop()


feature_names = [str(name) for name in feature_names]


# ==========================================
# SIDEBAR
# ==========================================

with st.sidebar:

    st.title("🎓 Student Performance AI")

    st.markdown("---")

    st.subheader("About This Project")

    st.write(
        "This application uses a trained "
        "Multilayer Neural Network to estimate "
        "student final grades."
    )

    st.markdown("---")

    st.subheader("Prediction Guide")

    st.write("**Input:** Student academic information")

    st.write("**Model:** Multilayer Neural Network")

    st.write("**Output:** Estimated final grade")

    st.markdown("---")

    st.caption(
        "Academic decision support tool. "
        "Predictions are estimates, not guarantees."
    )


# ==========================================
# STUDENT INPUT SECTION
# ==========================================

st.markdown(
    """
    <div class="section-card">
        <h2>📝 Student Information</h2>
        <p>
            Enter the student's academic details
            to estimate final performance.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)


with st.form("student_prediction_form"):

    col1, col2 = st.columns(2)

    with col1:

        age = st.number_input(
            "Student Age",
            min_value=15,
            max_value=25,
            value=17,
            step=1
        )

        studytime = st.selectbox(
            "Weekly Study Time",
            options=[1, 2, 3, 4],
            format_func=lambda x: {
                1: "Less than 2 hours",
                2: "2 to 5 hours",
                3: "5 to 10 hours",
                4: "More than 10 hours"
            }[x]
        )

        failures = st.number_input(
            "Number of Past Class Failures",
            min_value=0,
            max_value=4,
            value=0,
            step=1
        )

    with col2:

        absences = st.number_input(
            "Number of School Absences",
            min_value=0,
            max_value=100,
            value=2,
            step=1
        )

        G1 = st.slider(
            "First Period Grade (G1)",
            min_value=0,
            max_value=20,
            value=12
        )

        G2 = st.slider(
            "Second Period Grade (G2)",
            min_value=0,
            max_value=20,
            value=13
        )

    submitted = st.form_submit_button(
        "🔍 Predict Student Performance"
    )


# ==========================================
# PREDICTION
# ==========================================

if submitted:

    student_information = {
        "age": age,
        "studytime": studytime,
        "failures": failures,
        "absences": absences,
        "G1": G1,
        "G2": G2
    }

    missing_features = [
        feature
        for feature in feature_names
        if feature not in student_information
    ]

    extra_information = [
        feature
        for feature in student_information
        if feature not in feature_names
    ]

    if missing_features:

        st.error(
            "The trained model requires additional input "
            "features that are not included in this form."
        )

        st.write("Required model features:")

        st.code(", ".join(feature_names))

        st.write("Features not provided by this form:")

        st.code(", ".join(missing_features))

        st.info(
            "The input form must match the exact features "
            "used when training the model. Update the form "
            "and feature preparation to match train_model.py."
        )

        st.stop()

    try:

        # Build input in the same order as features.pkl
        input_data = {
            feature: student_information[feature]
            for feature in feature_names
        }

        input_df = pd.DataFrame(
            [input_data],
            columns=feature_names
        )

        # Scale the input features
        scaled_input = scaler.transform(input_df)

        # Predict the final grade
        prediction = model.predict(
            scaled_input,
            verbose=0
        )

        predicted_grade = float(
            np.asarray(prediction).reshape(-1)[0]
        )

        # Student grades normally range from 0 to 20
        predicted_grade = float(
            np.clip(predicted_grade, 0, 20)
        )

        # ==========================================
        # PERFORMANCE CATEGORY
        # ==========================================

        if predicted_grade < 10:

            performance = "Needs Improvement"

            performance_message = (
                "The student may benefit from additional "
                "academic support and a structured study plan."
            )

        elif predicted_grade < 15:

            performance = "Average Performance"

            performance_message = (
                "The student is making progress. "
                "Consistent practice may help improve results."
            )

        else:

            performance = "Good Performance"

            performance_message = (
                "The student is showing good estimated "
                "academic performance. Consistent effort "
                "can help maintain progress."
            )

        # ==========================================
        # DISPLAY PREDICTION RESULT
        # ==========================================

        st.markdown(
            """
            <div class="section-card">
                <h2>📊 Prediction Results</h2>
            </div>
            """,
            unsafe_allow_html=True
        )

        result_col1, result_col2 = st.columns(2)

        with result_col1:

            st.markdown(
                f"""
                <div class="result-card">
                    <p>Predicted Final Grade</p>
                    <div class="result-number">
                        {predicted_grade:.2f}/20
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        with result_col2:

            st.markdown(
                f"""
                <div class="result-card">
                    <p>Performance Category</p>
                    <h2>{performance}</h2>
                </div>
                """,
                unsafe_allow_html=True
            )

        st.info(performance_message)

        # ==========================================
        # STUDENT SUMMARY
        # ==========================================

        st.subheader("📋 Student Information Summary")

        summary_df = pd.DataFrame(
            {
                "Feature": list(student_information.keys()),
                "Value": list(student_information.values())
            }
        )

        st.dataframe(
            summary_df,
            use_container_width=True,
            hide_index=True
        )

        # ==========================================
        # PERSONALIZED SUGGESTIONS
        # ==========================================

        st.subheader("💡 Recommended Improvement Plan")

        suggestions = []

        if studytime == 1:

            suggestions.append(
                "Increase weekly study time gradually "
                "and follow a regular study timetable."
            )

        elif studytime == 2:

            suggestions.append(
                "Maintain a consistent study schedule "
                "and include regular revision sessions."
            )

        elif studytime == 3:

            suggestions.append(
                "Continue regular study and focus on "
                "topics that need additional practice."
            )

        else:

            suggestions.append(
                "Maintain a balanced study routine and "
                "include short breaks to avoid burnout."
            )

        if G1 < 10:

            suggestions.append(
                "Review first-period topics and ask "
                "teachers for help with difficult concepts."
            )

        if G2 < 10:

            suggestions.append(
                "Practice second-period topics and use "
                "short quizzes to check understanding."
            )

        if G1 >= 10 and G2 >= 10 and G2 > G1:

            suggestions.append(
                "Keep building on the improvement "
                "shown between the first and second periods."
            )

        if G2 < G1:

            suggestions.append(
                "Review recent learning progress and "
                "identify the topics that caused the "
                "second-period grade to decrease."
            )

        if absences > 10:

            suggestions.append(
                "Improve attendance where possible and "
                "catch up on missed classroom lessons."
            )

        elif absences > 5:

            suggestions.append(
                "Try to attend classes regularly and "
                "complete any missed assignments."
            )

        else:

            suggestions.append(
                "Continue maintaining regular attendance."
            )

        if failures > 0:

            suggestions.append(
                "Seek additional support in subjects "
                "previously failed and practice basic concepts."
            )

        else:

            suggestions.append(
                "Continue reviewing lessons regularly "
                "to strengthen subject understanding."
            )

        for index, suggestion in enumerate(
            suggestions,
            start=1
        ):

            st.markdown(
                f"**{index}.** {suggestion}"
            )

        # ==========================================
        # PERFORMANCE PROGRESS BAR
        # ==========================================

        st.subheader("📈 Predicted Grade Progress")

        st.progress(
            int(round(predicted_grade / 20 * 100))
        )

        st.write(
            f"Estimated grade: {predicted_grade:.2f} "
            "out of 20"
        )

        # ==========================================
        # DOWNLOAD REPORT
        # ==========================================

        st.subheader("📥 Download Student Report")

        report_data = {
            "Student Age": age,
            "Weekly Study Time": studytime,
            "Past Class Failures": failures,
            "School Absences": absences,
            "First Period Grade (G1)": G1,
            "Second Period Grade (G2)": G2,
            "Predicted Final Grade": round(
                predicted_grade,
                2
            ),
            "Performance Category": performance,
            "Improvement Suggestions": " | ".join(
                suggestions
            )
        }

        report_df = pd.DataFrame(
            [report_data]
        )

        csv_report = report_df.to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(
            label="⬇️ Download Prediction Report (CSV)",
            data=csv_report,
            file_name="student_performance_report.csv",
            mime="text/csv"
        )

        # ==========================================
        # DISCLAIMER
        # ==========================================

        st.markdown("---")

        st.caption(
            "Disclaimer: This prediction is an estimate "
            "generated by a machine learning model. "
            "It should not be used as the sole basis for "
            "academic decisions. Actual results may differ."
        )

    except Exception as error:

        st.error(
            "An error occurred while predicting "
            "student performance."
        )

        st.write(
            "Check that the model, scaler, feature names, "
            "and input columns match the training setup."
        )

        st.code(str(error))


# ==========================================
# FOOTER
# ==========================================

st.markdown("---")

st.markdown(
    """
    <div style="text-align:center; padding:10px;">
        <p>
            🎓 Student Performance AI |
            Multilayer Neural Network
        </p>
        <p class="small-note">
            Built with Python, TensorFlow and Streamlit
        </p>
    </div>
    """,
    unsafe_allow_html=True
)
