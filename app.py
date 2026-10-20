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
# CUSTOM STYLING
# ==========================================

st.markdown(
    """
    <style>
    .stApp {
        background-color: #f0f4ff;
    }

    div[data-testid="stMetric"] {
        background-color: white;
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #dbeafe;
    }

    div.stButton > button {
        width: 100%;
        background-color: #2563eb;
        color: white;
        border-radius: 8px;
        height: 45px;
        font-weight: bold;
    }

    div.stButton > button:hover {
        background-color: #1d4ed8;
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# ==========================================
# APPLICATION HEADER
# ==========================================

st.title("🎓 Student Performance AI")

st.subheader(
    "Multilayer Neural Network Based Student Performance Prediction"
)

st.write(
    "Predict Performance | Identify Learning Needs | "
    "Improve Academic Results"
)

st.divider()


# ==========================================
# LOAD MODEL FILES
# ==========================================

@st.cache_resource
def load_project_files():

    required_files = [
        MODEL_PATH,
        SCALER_PATH,
        FEATURES_PATH
    ]

    for file_path in required_files:
        if not file_path.exists():
            raise FileNotFoundError(
                f"Required file not found: {file_path.name}"
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

    st.error("Unable to load the trained model or supporting files.")

    st.write(
        "Make sure student_model.keras, scaler.pkl, and "
        "features.pkl are in the same folder as app.py."
    )

    st.code(str(error))

    st.stop()


# ==========================================
# PREPARE FEATURE NAMES
# ==========================================

if isinstance(features, dict):
    feature_names = list(features.keys())

elif isinstance(features, (list, tuple, pd.Index, np.ndarray)):
    feature_names = list(features)

else:
    st.error(
        "Unsupported features.pkl format. "
        "Check how features.pkl was created during training."
    )
    st.stop()


feature_names = [str(feature) for feature in feature_names]


# ==========================================
# SIDEBAR
# ==========================================

with st.sidebar:

    st.title("🎓 Student Performance AI")

    st.write("### About the Project")

    st.write(
        "This application uses a trained Multilayer Neural Network "
        "to estimate student academic performance."
    )

    st.divider()

    st.write("### How It Works")

    st.write("1. Enter student information.")

    st.write("2. Submit the academic details.")

    st.write("3. View the predicted final grade.")

    st.write("4. Review personalized improvement suggestions.")

    st.divider()

    st.caption(
        "Predictions are estimates and should not be treated "
        "as guaranteed academic results."
    )


# ==========================================
# STUDENT INFORMATION FORM
# ==========================================

st.header("📝 Student Information")

st.write(
    "Enter the student's academic details to predict "
    "their estimated final grade."
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
            format_func=lambda value: {
                1: "Less than 2 hours",
                2: "2 to 5 hours",
                3: "5 to 10 hours",
                4: "More than 10 hours"
            }[value]
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
        "Predict Student Performance"
    )


# ==========================================
# MAKE PREDICTION
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

    if missing_features:

        st.error(
            "The trained model requires input features "
            "that are not available in this form."
        )

        st.write("Required model features:")

        st.code(", ".join(feature_names))

        st.write("Missing input features:")

        st.code(", ".join(missing_features))

        st.info(
            "Update the input form to match the exact features "
            "used in train_model.py."
        )

        st.stop()

    try:

        # Prepare input in the order expected by the model.

        input_data = {
            feature: student_information[feature]
            for feature in feature_names
        }

        input_df = pd.DataFrame(
            [input_data],
            columns=feature_names
        )

        # Scale the input features.

        scaled_input = scaler.transform(input_df)

        # Predict the final grade.

        prediction = model.predict(
            scaled_input,
            verbose=0
        )

        predicted_grade = float(
            np.asarray(prediction).reshape(-1)[0]
        )

        predicted_grade = float(
            np.clip(predicted_grade, 0, 20)
        )

        # ==========================================
        # PERFORMANCE CATEGORY
        # ==========================================

        if predicted_grade < 10:

            performance = "Needs Improvement"

            performance_message = (
                "The student may benefit from additional academic "
                "support and a structured study plan."
            )

        elif predicted_grade < 15:

            performance = "Average Performance"

            performance_message = (
                "The student is making progress. Regular practice "
                "and revision may help improve academic results."
            )

        else:

            performance = "Good Performance"

            performance_message = (
                "The student has a good estimated performance. "
                "Consistent effort can help maintain progress."
            )

        # ==========================================
        # DISPLAY RESULTS
        # ==========================================

        st.divider()

        st.header("📊 Prediction Results")

        result_col1, result_col2 = st.columns(2)

        with result_col1:

            st.metric(
                label="Predicted Final Grade",
                value=f"{predicted_grade:.2f} / 20"
            )

        with result_col2:

            st.metric(
                label="Performance Category",
                value=performance
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
                "Increase weekly study time gradually and follow "
                "a regular study timetable."
            )

        elif studytime == 2:

            suggestions.append(
                "Maintain a consistent study schedule and include "
                "regular revision sessions."
            )

        elif studytime == 3:

            suggestions.append(
                "Continue regular study and focus on topics "
                "that need additional practice."
            )

        else:

            suggestions.append(
                "Maintain a balanced study routine and include "
                "short breaks to avoid burnout."
            )

        if G1 < 10:

            suggestions.append(
                "Review first-period topics and ask teachers "
                "for help with difficult concepts."
            )

        if G2 < 10:

            suggestions.append(
                "Practice second-period topics and use short "
                "quizzes to check understanding."
            )

        if G2 < G1:

            suggestions.append(
                "Review recent topics and identify the reasons "
                "for the decrease in grades."
            )

        elif G2 > G1:

            suggestions.append(
                "Continue the study methods that helped improve "
                "the grade between the first and second periods."
            )

        if absences > 10:

            suggestions.append(
                "Improve attendance where possible and catch up "
                "on missed lessons and assignments."
            )

        elif absences > 5:

            suggestions.append(
                "Attend classes regularly and complete any "
                "missed assignments."
            )

        else:

            suggestions.append(
                "Continue maintaining regular class attendance."
            )

        if failures > 0:

            suggestions.append(
                "Seek additional support in subjects previously "
                "failed and practice basic concepts."
            )

        else:

            suggestions.append(
                "Review lessons regularly to strengthen "
                "subject understanding."
            )

        for index, suggestion in enumerate(suggestions, start=1):

            st.write(f"**{index}.** {suggestion}")

        # ==========================================
        # PREDICTED GRADE PROGRESS
        # ==========================================

        st.subheader("📈 Predicted Grade Progress")

        progress_value = int(
            round(predicted_grade / 20 * 100)
        )

        st.progress(progress_value)

        st.write(
            f"Predicted grade: {predicted_grade:.2f} out of 20"
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
            "First Period Grade": G1,
            "Second Period Grade": G2,
            "Predicted Final Grade": round(predicted_grade, 2),
            "Performance Category": performance,
            "Improvement Suggestions": " | ".join(suggestions)
        }

        report_df = pd.DataFrame([report_data])

        csv_report = report_df.to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(
            label="Download Prediction Report (CSV)",
            data=csv_report,
            file_name="student_performance_report.csv",
            mime="text/csv"
        )

        # ==========================================
        # DISCLAIMER
        # ==========================================

        st.divider()

        st.caption(
            "Disclaimer: This prediction is an estimate generated "
            "by a machine learning model. Actual student results "
            "may differ. Do not use this prediction as the sole "
            "basis for academic decisions."
        )

    except Exception as error:

        st.error(
            "An error occurred while predicting student performance."
        )

        st.write(
            "Check that the model, scaler, feature names, and "
            "input columns match the training configuration."
        )

        st.code(str(error))


# ==========================================
# FOOTER
# ==========================================

st.divider()

st.caption(
    "Student Performance AI | Built with Python, "
    "TensorFlow, Pandas, and Streamlit"
)
