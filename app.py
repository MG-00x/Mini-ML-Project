import streamlit as st
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="AI Career Recommendation",
    page_icon="🎯",
    layout="centered"
)

# ==========================================
# LOAD DATASET
# ==========================================

df = pd.read_csv("AI_Career_Recommendation_Dataset_Balanced.csv")

# ==========================================
# PREPARE DATA
# ==========================================

X = df.drop("Career", axis=1)
y = df["Career"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ==========================================
# TRAIN MODEL
# ==========================================

model = DecisionTreeClassifier(
    max_depth=5,
    random_state=42
)

model.fit(X_train, y_train)

# ==========================================
# TITLE
# ==========================================

st.title("🎯 AI Career Recommendation System")

st.write(
    "Enter your skills and interests to get an AI-based career recommendation."
)

st.divider()

# ==========================================
# USER INPUT
# ==========================================

st.subheader("📊 Rate Your Skills")

programming = st.slider("Programming", 1, 10, 5)
mathematics = st.slider("Mathematics", 1, 10, 5)
electronics = st.slider("Electronics", 1, 10, 5)
communication = st.slider("Communication", 1, 10, 5)
problem_solving = st.slider("Problem Solving", 1, 10, 5)
ai_interest = st.slider("AI/ML Interest", 1, 10, 5)
hardware_interest = st.slider("Hardware Interest", 1, 10, 5)
networking_interest = st.slider("Networking Interest", 1, 10, 5)

# ==========================================
# PREDICTION
# ==========================================

if st.button("🚀 Recommend Career"):

    student = pd.DataFrame([[
        programming,
        mathematics,
        electronics,
        communication,
        problem_solving,
        ai_interest,
        hardware_interest,
        networking_interest
    ]], columns=X.columns)

    prediction = model.predict(student)[0]

    probabilities = model.predict_proba(student)[0]

    confidence = max(probabilities) * 100

    # ======================================
    # DISPLAY RESULT
    # ======================================

    st.divider()

    st.subheader("🎯 Recommended Career")

    st.success(prediction)

    st.metric(
        "Prediction Confidence",
        f"{confidence:.2f}%"
    )

    # ======================================
    # ALL CAREER PROBABILITIES
    # ======================================

    st.subheader("Career Probability")

    probability_df = pd.DataFrame({
        "Career": model.classes_,
        "Probability": probabilities * 100
    })

    probability_df = probability_df.sort_values(
        "Probability",
        ascending=False
    )

    st.bar_chart(
        probability_df.set_index("Career")
    )
