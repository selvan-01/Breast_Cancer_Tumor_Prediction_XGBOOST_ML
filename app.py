# ===============================================
# Streamlit App - ML Algorithm Comparison
# Breast Cancer Tumor Prediction
# ===============================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, StratifiedKFold

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC


# --------- Page Config ---------
st.set_page_config(
    page_title="ML Model Comparison",
    page_icon="🧠",
    layout="wide"
)

# --------- Custom Styling ---------
st.markdown("""
    <style>
    .main {
        background-color: #f5f7fa;
    }
    h1 {
        color: #ff4b4b;
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)


# --------- Title ---------
st.title("🧠 Breast Cancer Tumor Prediction")
st.markdown("### 🔍 ML Algorithm Comparison Dashboard")


# --------- File Upload ---------
uploaded_file = st.file_uploader("📂 Upload Dataset (CSV)", type=["csv"])

if uploaded_file is not None:
    
    # Load dataset
    dataset = pd.read_csv(uploaded_file)

    st.subheader("📊 Dataset Preview")
    st.dataframe(dataset.head())

    # Split features & target
    X = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, -1].values

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0
    )

    st.success("✅ Dataset Loaded Successfully!")

    # --------- Models ---------
    models = []

    models.append(('Logistic Regression', LogisticRegression(solver='liblinear')))
    models.append(('LDA', LinearDiscriminantAnalysis()))
    models.append(('KNN', KNeighborsClassifier()))
    models.append(('Decision Tree', DecisionTreeClassifier()))
    models.append(('Naive Bayes', GaussianNB()))
    models.append(('SVM', SVC(gamma='auto')))

    results = []
    names = []
    mean_scores = []

    st.subheader("⚙️ Model Evaluation (10-Fold Cross Validation)")

    for name, model in models:
        kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)

        cv_results = cross_val_score(
            model, X_train, y_train, cv=kfold, scoring='accuracy'
        )

        results.append(cv_results)
        names.append(name)
        mean_scores.append(cv_results.mean())

    # --------- Results Table ---------
    result_df = pd.DataFrame({
        "Model": names,
        "Accuracy": mean_scores
    })

    st.dataframe(result_df)

    # --------- Best Model ---------
    best_model = result_df.loc[result_df['Accuracy'].idxmax()]

    st.success(f"🏆 Best Model: {best_model['Model']} "
               f"({best_model['Accuracy']*100:.2f}%)")

    # --------- Visualization ---------
    st.subheader("📈 Model Comparison Chart")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(names, mean_scores)
    ax.set_ylim(0.5, 1.0)
    ax.set_xlabel("Models")
    ax.set_ylabel("Accuracy")
    ax.set_title("ML Algorithm Comparison")

    st.pyplot(fig)

else:
    st.info("👆 Upload a dataset to begin")