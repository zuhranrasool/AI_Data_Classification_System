# 🎓 AI Data Classification System

An interactive, modular Machine Learning framework designed to clean, process, and classify student academic performance. The system evaluates multiple classification algorithms, identifies the top performer, exports production-ready model pipelines, and presents deep performance insights through a centralized Streamlit Web Dashboard.

---

## 🚀 Features

- **Automated Data Processing Pipeline:** Seamless CSV data loading, deduplication, missing value treatment, and target categorical encoding.
- **Multi-Model Machine Learning Engine:** Concurrent training across four robust algorithms:
  - Decision Tree
  - Logistic Regression
  - K-Nearest Neighbors (KNN)
  - Random Forest
- **Production Deployment Readiness:** Auto-selects and serializes the highest-accuracy model architecture to a standalone binary pickle file (`.pkl`).
- **Granular Model Analytics:** Generates comprehensive metric profiles capturing Precision, Recall, F1 Scores, Confusion Matrices, and line/bar model comparison plots.
- **Interactive Streamlit UI:** An end-to-end multi-tab application serving data visualization analytics, bulk training metrics, and live individual prediction inputs.

---

## 📁 Folder Structure

```text
AI_Data_Classification_System/
│
├── app.py                      # Main Streamlit dashboard application
├── main.py                     # Main operational entry point to run the ML pipeline
├── requirements.txt            # Python library dependency declarations
├── README.md                   # System configuration & operational documentation
├── LICENSE
├── .gitignore                  # Excludes logs, binary outputs, and environments
│
├── dataset/
│   ├── student_data.csv        # Core synthetic dataset (100–200 records)
│   └── sample_test_data.csv    # Independent raw evaluation testing data
│
├── models/
│   ├── train_model.py          # Model architecture routing engine
│   ├── predict.py              # Out-of-sample inference pipeline execution
│   └── save_model.py           # Joblib persistence implementation
│
├── preprocessing/
│   ├── data_loader.py          # Raw file parsing interface
│   ├── data_cleaning.py        # Cleans and drops bad data rows
│   ├── feature_engineering.py  # Categorical mapping & dynamic vectorization
│   └── split_dataset.py        # Stratified/Random 80/20 partitioning split
│
├── algorithms/
│   ├── decision_tree.py
│   ├── logistic_regression.py
│   ├── knn_classifier.py
│   └── random_forest.py
│
├── evaluation/
│   ├── accuracy.py             # Metric computation
│   ├── confusion_matrix.py     # Matrix evaluation logic
│   ├── classification_report.py# Text-based reports generator
│   └── metrics.py              # Performance metric collection handler
│
├── visualization/
│   ├── plots.py                # Exploratory Data Analysis graphics
│   ├── graphs.py               # Feature statistical charts
│   └── charts.py               # Direct performance delta evaluation
│
├── utils/
│   ├── helper.py
│   ├── constants.py
│   └── file_manager.py
│
├── output/                     # Generated persistent pipeline artifacts
│   ├── trained_model.pkl
│   ├── predictions.csv
│   ├── confusion_matrix.png
│   ├── accuracy_graph.png
│   └── reports/
│       └── model_report.txt
│
├── static/                     # Styling variables and visual identity assets
│   ├── css/
│   │   └── style.css
│   └── images/
│       └── logo.png
│
└── docs/                       # Project presentation metadata
    ├── screenshots/
    └── project_flow.png