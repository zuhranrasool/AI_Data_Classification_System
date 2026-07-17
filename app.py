import streamlit as st

# ---------------------------------------------------------
# PAGE CONFIGURATION
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

    st.info(
        "Developed as part of the AI Internship Project."
    )

# ---------------------------------------------------------
# PLACEHOLDERS FOR UPCOMING STEPS
# ---------------------------------------------------------
elif page == "📊 Dataset":

    st.title("📊 Dataset Explorer")

    st.markdown("---")

    import pandas as pd

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

    import pandas as pd

    # Load dataset
    dataset = pd.read_csv("dataset/student_data.csv")

    # Preprocess
    dataset = clean_dataset(dataset)

    X, y, label_encoder = feature_engineering(dataset)

    X_train, X_test, y_train, y_test = split_dataset(X, y)

    # Train models
    dt_model, dt_pred, dt_acc = train_decision_tree(
        X_train, X_test, y_train, y_test
    )

    lr_model, lr_pred, lr_acc = train_logistic_regression(
        X_train, X_test, y_train, y_test
    )

    knn_model, knn_pred, knn_acc = train_knn_classifier(
        X_train, X_test, y_train, y_test
    )

    rf_model, rf_pred, rf_acc = train_random_forest(
        X_train, X_test, y_train, y_test
    )

    st.subheader("Model Accuracy")

    results = pd.DataFrame({
        "Model": [
            "Decision Tree",
            "Logistic Regression",
            "K-Nearest Neighbors",
            "Random Forest"
        ],
        "Accuracy": [
            dt_acc,
            lr_acc,
            knn_acc,
            rf_acc
        ]
    })

    results["Accuracy (%)"] = (results["Accuracy"] * 100).round(2)

    st.dataframe(
        results[["Model", "Accuracy (%)"]],
        use_container_width=True
    )

    st.markdown("---")

    best_model = results.loc[
        results["Accuracy"].idxmax()
    ]

    st.success(
        f"🏆 Best Model: {best_model['Model']} "
        f"({best_model['Accuracy (%)']:.2f}%)"
    )

elif page == "🔮 Prediction":
    st.header("Prediction")
    st.write("Coming in Step 27...")

elif page == "📈 Evaluation":
    st.header("Evaluation")
    st.write("Coming in Step 28...")

elif page == "📉 Visualization":
    st.header("Visualization")
    st.write("Coming in Step 29...")