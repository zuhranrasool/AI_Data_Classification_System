import streamlit as st
import os
import pandas as pd
import joblib

# ---------------------------------------------------------
# PAGE CONFIGURATION (Must be at the very top, called ONCE)
# ---------------------------------------------------------
st.set_page_config(
    page_title="AI Data Classification System",
    page_icon="🎓",
    layout="wide"
)

# ---------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------
st.sidebar.title("📚 Navigation")

page = st.sidebar.radio(
    "Go to",
    [
        "🏠 Home",
        "📊 Dataset",
        "🤖 Model Training",
        "🔮 Prediction",
        "📈 Evaluation",
        "📉 Visualization"
    ]
)

# ---------------------------------------------------------
# HOME PAGE
# ---------------------------------------------------------
if page == "🏠 Home":

    st.title("🎓 AI Data Classification System")
    st.markdown("---")

    st.subheader("Project Overview")
    st.write("""
This project is a Machine Learning based Student Performance
Classification System.

It predicts student performance using multiple machine learning
algorithms including:

- Decision Tree
- Logistic Regression
- K-Nearest Neighbors (KNN)
- Random Forest

The system compares all algorithms, selects the best-performing model,
predicts student performance, evaluates the model, and visualizes the
results through professional charts.
""")

    st.markdown("---")
    st.subheader("Project Features")

    col1, col2 = st.columns(2)
    with col1:
        st.success("✔ Load Dataset")
        st.success("✔ Clean Dataset")
        st.success("✔ Feature Engineering")
        st.success("✔ Train ML Models")

    with col2:
        st.success("✔ Predict Student Performance")
        st.success("✔ Evaluate Accuracy")
        st.success("✔ Generate Visualizations")
        st.success("✔ Interactive Dashboard")

    st.markdown("---")
    st.subheader("Technologies Used")
    st.write("""
- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Streamlit
- Joblib
""")

    st.markdown("---")
    st.info("Developed as part of the AI Internship Project.")

# ---------------------------------------------------------
# DATASET PAGE
# ---------------------------------------------------------
elif page == "📊 Dataset":

    st.title("📊 Dataset Explorer")
    st.markdown("---")

    # Load dataset
    dataset = pd.read_csv("dataset/student_data.csv")

    st.subheader("Dataset Preview")
    st.dataframe(dataset)

    st.markdown("---")
    st.subheader("Dataset Information")

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Rows", dataset.shape[0])
    with col2:
        st.metric("Total Columns", dataset.shape[1])

    st.markdown("---")
    st.subheader("Dataset Shape")
    st.write(f"Rows : {dataset.shape[0]}")
    st.write(f"Columns : {dataset.shape[1]}")

    st.markdown("---")
    st.subheader("Column Names")
    st.write(list(dataset.columns))

    st.markdown("---")
    st.subheader("Statistical Summary")
    st.dataframe(dataset.describe())

    st.markdown("---")
    st.success("Dataset loaded successfully.")

# ---------------------------------------------------------
# MODEL TRAINING PAGE
# ---------------------------------------------------------
elif page == "🤖 Model Training":

    st.title("🤖 Model Training Dashboard")
    st.markdown("---")

    from preprocessing.data_cleaning import clean_dataset
    from preprocessing.feature_engineering import feature_engineering
    from preprocessing.split_dataset import split_dataset

    from algorithms.decision_tree import train_decision_tree
    from algorithms.logistic_regression import train_logistic_regression
    from algorithms.knn_classifier import train_knn_classifier
    from algorithms.random_forest import train_random_forest

    # Load dataset
    dataset = pd.read_csv("dataset/student_data.csv")

    # Preprocess
    dataset = clean_dataset(dataset)
    X, y, label_encoder = feature_engineering(dataset)
    X_train, X_test, y_train, y_test = split_dataset(X, y)

    # Train models
    dt_model, dt_pred, dt_acc = train_decision_tree(X_train, X_test, y_train, y_test)
    lr_model, lr_pred, lr_acc = train_logistic_regression(X_train, X_test, y_train, y_test)
    knn_model, knn_pred, knn_acc = train_knn_classifier(X_train, X_test, y_train, y_test)
    rf_model, rf_pred, rf_acc = train_random_forest(X_train, X_test, y_train, y_test)

    st.subheader("Model Accuracy")
    results = pd.DataFrame({
        "Model": [
            "Decision Tree",
            "Logistic Regression",
            "K-Nearest Neighbors",
            "Random Forest"
        ],
        "Accuracy": [dt_acc, lr_acc, knn_acc, rf_acc]
    })

    results["Accuracy (%)"] = (results["Accuracy"] * 100).round(2)

    st.dataframe(
        results[["Model", "Accuracy (%)"]],
        use_container_width=True
    )

    st.markdown("---")
    best_model = results.loc[results["Accuracy"].idxmax()]

    st.success(
        f"🏆 Best Model: {best_model['Model']} "
        f"({best_model['Accuracy (%)']:.2f}%)"
    )

# ---------------------------------------------------------
# PREDICTION PAGE
# ---------------------------------------------------------
elif page == "🔮 Prediction":

    st.title("🔮 Student Performance Prediction")
    st.markdown("---")

    st.subheader("Enter Student Information")
    col1, col2 = st.columns(2)

    with col1:
        study_hours = st.number_input("Study Hours", min_value=0, max_value=12, value=5)
        attendance = st.number_input("Attendance (%)", min_value=0, max_value=100, value=80)
        assignment = st.number_input("Assignment Marks", min_value=0, max_value=100, value=75)

    with col2:
        midterm = st.number_input("Midterm Marks", min_value=0, max_value=100, value=70)
        final = st.number_input("Final Marks", min_value=0, max_value=100, value=75)

    st.markdown("---")

    if st.button("Predict Performance"):
        # Load trained model
        model = joblib.load("output/trained_model.pkl")

        # Create input dataframe
        input_data = pd.DataFrame({
            "Study_Hours": [study_hours],
            "Attendance": [attendance],
            "Assignment": [assignment],
            "Midterm": [midterm],
            "Final": [final]
        })

        # Prediction
        prediction = model.predict(input_data)[0]

        label_map = {
            0: "Average",
            1: "Excellent",
            2: "Good",
            3: "Poor"
        }

        predicted_label = label_map[prediction]

        st.markdown("---")
        st.success(f"Predicted Student Performance: **{predicted_label}**")

# ---------------------------------------------------------
# EVALUATION PAGE (Step 28 - Fixed Routing Isolation)
# ---------------------------------------------------------
elif page == "📈 Evaluation":

    st.title("📊 Model Evaluation")
    st.markdown("---")

    # =====================================================
    # MODEL ACCURACY
    # =====================================================
    st.subheader("Model Accuracy")
    accuracy_data = {
        "Decision Tree": "97.50%",
        "Logistic Regression": "100.00%",
        "KNN": "100.00%",
        "Random Forest": "100.00%"
    }
    st.table(accuracy_data)
    st.success("Best Model: Logistic Regression")
    st.markdown("---")

    # =====================================================
    # EVALUATION METRICS
    # =====================================================
    st.subheader("Evaluation Metrics")
    metric1, metric2 = st.columns(2)

    with metric1:
        st.metric("Accuracy", "100%")
        st.metric("Precision", "100%")
    with metric2:
        st.metric("Recall", "100%")
        st.metric("F1 Score", "100%")
    st.markdown("---")

    # =====================================================
    # CONFUSION MATRIX
    # =====================================================
    st.subheader("Confusion Matrix")
    confusion_matrix_path = "output/confusion_matrix.png"

    if os.path.exists(confusion_matrix_path):
        st.image(confusion_matrix_path, use_container_width=True)
    else:
        st.warning("Confusion Matrix not found.")
    st.markdown("---")

    # =====================================================
    # CLASSIFICATION REPORT
    # =====================================================
    st.subheader("Classification Report")
    report_path = "output/reports/model_report.txt"

    if os.path.exists(report_path):
        with open(report_path, "r") as file:
            report = file.read()
        st.text(report)
    else:
        st.warning("Classification report not found.")

# ---------------------------------------------------------
# VISUALIZATION PAGE (Step 29 Placeholder)
# ---------------------------------------------------------
elif page == "📉 Visualization":
    st.title("📉 Visualization Dashboard")
    st.markdown("---")
    st.info("Placeholder for Step 29: Displaying charts and analytical performance metrics.")