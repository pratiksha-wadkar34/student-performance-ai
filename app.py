import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from pathlib import Path
from tensorflow.keras.models import load_model

# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Student Performance AI",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ==========================================
# PROJECT FILE PATHS
# ==========================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "student_model.keras"
SCALER_PATH = BASE_DIR / "scaler.pkl"
FEATURES_PATH = BASE_DIR / "features.pkl"


# ==========================================
# CUSTOM CSS AND ANIMATIONS
# ==========================================

st.markdown("""
<style>

.stApp {
    background: linear-gradient(
        135deg,
        #e8f5e9 0%,
        #f3e5f5 50%,
        #e3f2fd 100%
    );
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
    padding: 38px 25px;
    border-radius: 24px;
    text-align: center;
    margin-bottom: 28px;
    box-shadow: 0 10px 30px rgba(67, 56, 202, 0.20);
}

.hero h1 {
    color: white;
    font-size: 42px;
    font-weight: 800;
    margin-bottom: 10px;
}

.hero p {
    color: white;
    font-size: 18px;
    margin-bottom: 0;
}

@keyframes gradientMove {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}

.section-card {
    background: rgba(255, 255, 255, 0.85);
    padding: 22px;
    border-radius: 18px;
    border: 1px solid rgba(124, 58, 237, 0.12);
    box-shadow: 0 5px 18px rgba(30, 41, 59, 0.06);
    margin-bottom: 20px;
}

.result-card {
    background: linear-gradient(135deg, #ffffff, #ede9fe);
    padding: 28px;
    border-radius: 20px;
    text-align: center;
    border: 2px solid #c4b5fd;
    animation: appear 0.7s ease-out;
}

@keyframes appear {
    from {
        opacity: 0;
        transform: translateY(18px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

.result-number {
    font-size: 48px;
    font-weight: 800;
    color: #4338ca;
}

.result-label {
    font-size: 19px;
    font-weight: 600;
    color: #374151;
}

.tip-card {
    background: rgba(255, 255, 255, 0.92);
    padding: 17px;
    border-radius: 14px;
    border-left: 5px solid #7c3aed;
    margin-bottom: 12px;
    color: #374151;
}

div.stButton > button {
    width: 100%;
    border-radius: 12px;
    min-height: 48px;
    font-weight: 700;
    transition: all 0.25s ease;
}

div.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 5px 14px rgba(124, 58, 237, 0.20);
}

.footer {
    text-align: center;
    padding: 20px;
    color: #6b7280;
    font-size: 14px;
}

</style>
""", unsafe_allow_html=True)


# ==========================================
# LOAD TRAINED MODEL AND PREPROCESSORS
# ==========================================

@st.cache_resource
def load_project_files():

    required_files = [
        MODEL_PATH,
        SCALER_PATH,
        FEATURES_PATH
    ]

    missing_files = [
        str(path.name)
        for path in required_files
        if not path.is_file()
    ]

    if missing_files:
        raise FileNotFoundError(
            "खालील files project folder मध्ये सापडल्या नाहीत: "
            + ", ".join(missing_files)
        )

    model = load_model(MODEL_PATH, compile=False)

    scaler = joblib.load(SCALER_PATH)

    features = joblib.load(FEATURES_PATH)

    return model, scaler, features


# ==========================================
# HEADER
# ==========================================

st.markdown("""
<div class="hero">
    <h1>🎓 Student Performance AI</h1>
    <p>
        Multilayer Neural Network Based
        Student Performance Prediction
    </p>
    <p>
        Predict Performance • Identify Learning Needs • Improve Results
    </p>
</div>
""", unsafe_allow_html=True)


# ==========================================
# PROJECT INTRODUCTION
# ==========================================

intro1, intro2, intro3 = st.columns(3)

with intro1:
    st.info("🎯 **Grade Prediction**\n\nPredict the student's final grade.")

with intro2:
    st.info("🔍 **Performance Analysis**\n\nIdentify areas that may need improvement.")

with intro3:
    st.info("💡 **Smart Suggestions**\n\nGet suggestions based on student inputs.")


st.divider()


# ==========================================
# LOAD MODEL
# ==========================================

try:
    model, scaler, features = load_project_files()

except Exception as error:
    st.error(
        "⚠️ Model load झाला नाही. खालील error तपासा."
    )

    st.code(str(error))

    st.info(
        "GitHub repository मध्ये app.py, student_model.keras, "
        "scaler.pkl आणि features.pkl या files योग्य ठिकाणी आहेत "
        "का ते तपासा."
    )

    st.stop()


# ==========================================
# STUDENT INFORMATION FORM
# ==========================================

st.header("📝 Student Information")

st.write(
    "विद्यार्थ्याची माहिती भरा आणि त्याच्या अंतिम गुणांचा अंदाज मिळवा."
)

with st.form("student_information_form"):

    left_col, right_col = st.columns(2)

    with left_col:

        st.subheader("👤 Personal & Study Details")

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
        type="primary",
        use_container_width=True
    )


# ==========================================
# PREDICTION AND PERFORMANCE ANALYSIS
# ==========================================

if submitted:

    # All available input features
    student_data = {
        "age": age,
        "absences": absences,
        "studytime": studytime,
        "failures": failures,
        "G1": G1,
        "G2": G2
    }

    # Convert feature metadata into an ordered list.
    # This supports a list or a pandas Index saved in features.pkl.

    if isinstance(features, dict):
        feature_names = list(features.keys())

    elif isinstance(features, (list, tuple, pd.Index, np.ndarray)):
        feature_names = list(features)

    else:
        st.error(
            "features.pkl मधील feature format ओळखता आला नाही. "
            "train_model.py मधील features save करण्याचा code तपासा."
        )
        st.stop()

    # Ensure all model features are available.
    missing_inputs = [
        feature for feature in feature_names
        if feature not in student_data
    ]

    if missing_inputs:
        st.error(
            "Model ला आवश्यक असलेले काही features form मध्ये नाहीत: "
            + ", ".join(missing_inputs)
        )

        st.info(
            "तुझ्या train_model.py मध्ये वापरलेल्या features "
            "आणि app.py मधील inputs समान असणे आवश्यक आहे."
        )

        st.stop()

    # Create input in exactly the same order as training.
    input_df = pd.DataFrame(
        [[student_data[feature] for feature in feature_names]],
        columns=feature_names
    )

    try:

        # Apply the same scaler used during training.
        scaled_input = scaler.transform(input_df)

        # Predict final grade.
        prediction = model.predict(
            scaled_input,
            verbose=0
        )

        predicted_grade = float(
            np.asarray(prediction).reshape(-1)[0]
        )

        # Clamp prediction to the valid UCI grade range.
        predicted_grade = float(
            np.clip(predicted_grade, 0, 20)
        )

    except Exception as error:

        st.error("Prediction करताना error आला.")

        st.code(str(error))

        st.stop()

    # ======================================
    # PERFORMANCE CLASSIFICATION
    # ======================================

    if predicted_grade < 10:

        performance = "Needs Improvement"
        performance_emoji = "⚠️"

        st.warning(
            "विद्यार्थ्याला अतिरिक्त शैक्षणिक मदतीची गरज असू शकते."
        )

    elif predicted_grade < 15:

        performance = "Average Performance"
        performance_emoji = "📘"

        st.info(
            "विद्यार्थ्याची कामगिरी सुधारण्यासाठी नियमित प्रयत्न उपयुक्त ठरू शकतात."
        )

    else:

        performance = "Good Performance"
        performance_emoji = "🌟"

        st.success(
            "विद्यार्थ्याची अंदाजित कामगिरी चांगली आहे."
        )

    st.divider()

    # ======================================
    # DISPLAY RESULTS
    # ======================================

    st.header("📊 Student Performance Result")

    result_col1, result_col2 = st.columns(2)

    with result_col1:

        st.markdown(f"""
        <div class="result-card">
            <div class="result-label">🎯 Predicted Final Grade</div>
            <div class="result-number">
                {predicted_grade:.2f} / 20
            </div>
            <p>Neural Network Prediction</p>
        </div>
        """, unsafe_allow_html=True)

    with result_col2:

        st.markdown(f"""
        <div class="result-card">
            <div class="result-label">📈 Performance Category</div>
            <div class="result-number">
                {performance_emoji}
            </div>
            <h3>{performance}</h3>
        </div>
        """, unsafe_allow_html=True)

    # ======================================
    # FEATURE-WISE IMPROVEMENT SUGGESTIONS
    # ======================================

    st.header("💡 Feature-wise Improvement Suggestions")

    suggestions = []

    if studytime == 1:

        suggestions.append(
            "📚 **Study Time:** अभ्यासासाठी नियमित वेळ ठरवा. "
            "दररोज छोट्या study sessions पासून सुरुवात करा."
        )

    elif studytime == 2:

        suggestions.append(
            "📚 **Study Time:** आठवड्याच्या अभ्यासाचे नियोजन करा "
            "आणि कठीण विषयांसाठी अतिरिक्त वेळ द्या."
        )

    if absences >= 10:

        suggestions.append(
            "🏫 **Attendance:** गैरहजेरी जास्त आहे. "
            "शक्य असल्यास वर्गांना नियमित उपस्थित राहा "
            "आणि चुकलेला अभ्यास पूर्ण करा."
        )

    if failures >= 1:

        suggestions.append(
            "📖 **Past Failures:** मागील अडचणींचे विषय ओळखा. "
            "शिक्षकांची मदत घ्या आणि त्या topics ची revision करा."
        )

    if G1 < 10:

        suggestions.append(
            "📘 **First Period Grade (G1):** पहिल्या period मधील "
            "कमी गुण मिळालेल्या विषयांचा पुन्हा अभ्यास करा."
        )

    if G2 < 10:

        suggestions.append(
            "📗 **Second Period Grade (G2):** दुसऱ्या period मधील "
            "कमकुवत topics वर लक्ष द्या आणि नियमित सराव करा."
        )

    if G2 < G1:

        suggestions.append(
            "📉 **Academic Trend:** G2 हे G1 पेक्षा कमी आहे. "
            "गुण कमी होण्याची संभाव्य कारणे शिक्षकांसोबत तपासा."
        )

    elif G2 > G1:

        suggestions.append(
            "📈 **Academic Trend:** G2 मध्ये G1 पेक्षा सुधारणा दिसते. "
            "अभ्यासाची चांगली सवय कायम ठेवा."
        )

    if not suggestions:

        suggestions.append(
            "🌟 सध्याच्या inputs मध्ये ठळक warning दिसत नाही. "
            "नियमित अभ्यास, उपस्थिती आणि revision सुरू ठेवा."
        )

    for suggestion in suggestions:

        st.markdown(
            f'<div class="tip-card">{suggestion}</div>',
            unsafe_allow_html=True
        )

    # ======================================
    # INPUT SUMMARY
    # ======================================

    st.header("📋 Student Information Summary")

    summary_df = pd.DataFrame({
        "Feature": [
            "Student Age",
            "Weekly Study Time",
            "Number of Absences",
            "Past Class Failures",
            "First Period Grade (G1)",
            "Second Period Grade (G2)"
        ],

        "Value": [
            age,
            studytime_label,
            absences,
            failures,
            G1,
            G2
        ]
    })

    st.dataframe(
        summary_df,
        use_container_width=True,
        hide_index=True
    )

    st.caption(
        "टीप: Prediction हे dataset वर train केलेल्या model चा अंदाज आहे. "
        "हे विद्यार्थ्याच्या भविष्यातील कामगिरीची हमी नाही. "
        "Improvement suggestions या input-based rules आहेत."
    )


# ==========================================
# FOOTER
# ==========================================

st.markdown("""
<div class="footer">
    🎓 Student Performance AI |
    Multilayer Neural Network |
    Academic Support System
</div>
""", unsafe_allow_html=True)
