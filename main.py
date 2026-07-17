from preprocessing.data_loader import load_dataset
from preprocessing.data_cleaning import clean_dataset
from preprocessing.feature_engineering import feature_engineering
from preprocessing.split_dataset import split_dataset

from algorithms.decision_tree import train_decision_tree
from algorithms.logistic_regression import train_logistic_regression
from algorithms.knn_classifier import train_knn_classifier
from algorithms.random_forest import train_random_forest

from evaluation.accuracy import evaluate_accuracy
from evaluation.confusion_matrix import generate_confusion_matrix
from evaluation.classification_report import generate_classification_report
from evaluation.metrics import evaluate_metrics

from models.save_model import save_model
from models.predict import predict_students

from visualization.plots import generate_basic_plots
from visualization.graphs import generate_advanced_graphs


def main():

    # ==========================================================
    # STEP 6 - LOAD DATASET
    # ==========================================================
    dataset = load_dataset("dataset/student_data.csv")

    if dataset is None:
        return

    # ==========================================================
    # STEP 7 - DATA CLEANING
    # ==========================================================
    dataset = clean_dataset(dataset)

    # ==========================================================
    # STEP 8 - FEATURE ENGINEERING
    # ==========================================================
    X, y, label_encoder = feature_engineering(dataset)

    # ==========================================================
    # STEP 9 - TRAIN / TEST SPLIT
    # ==========================================================
    X_train, X_test, y_train, y_test = split_dataset(X, y)

    # ==========================================================
    # STEP 10 - DECISION TREE
    # ==========================================================
    dt_model, dt_predictions, dt_accuracy = train_decision_tree(
        X_train,
        X_test,
        y_train,
        y_test
    )

    # ==========================================================
    # STEP 11 - LOGISTIC REGRESSION
    # ==========================================================
    lr_model, lr_predictions, lr_accuracy = train_logistic_regression(
        X_train,
        X_test,
        y_train,
        y_test
    )

    # ==========================================================
    # STEP 12 - KNN
    # ==========================================================
    knn_model, knn_predictions, knn_accuracy = train_knn_classifier(
        X_train,
        X_test,
        y_train,
        y_test
    )

    # ==========================================================
    # STEP 13 - RANDOM FOREST
    # ==========================================================
    rf_model, rf_predictions, rf_accuracy = train_random_forest(
        X_train,
        X_test,
        y_train,
        y_test
    )

    # ==========================================================
    # STEP 14 - MODEL COMPARISON
    # ==========================================================
    evaluate_accuracy(
        dt_accuracy,
        lr_accuracy,
        knn_accuracy,
        rf_accuracy
    )

    # ==========================================================
    # STEP 15 - BEST MODEL SELECTION
    # ==========================================================
    best_model = lr_model

    print("\n" + "=" * 60)
    print("BEST MODEL SELECTION")
    print("=" * 60)
    print("Selected Model : Logistic Regression")
    print(f"Accuracy       : {lr_accuracy:.2%}")

    # ==========================================================
    # STEP 15 - SAVE MODEL
    # ==========================================================
    save_model(best_model)

    # ==========================================================
    # STEP 16 - PREDICTION MODULE
    # ==========================================================
    predict_students()

    # ==========================================================
    # STEP 18 - CONFUSION MATRIX
    # ==========================================================
    generate_confusion_matrix(
        best_model,
        X_test,
        y_test,
        label_encoder
    )

    # ==========================================================
    # STEP 19 - CLASSIFICATION REPORT
    # ==========================================================
    generate_classification_report(
        best_model,
        X_test,
        y_test,
        label_encoder
    )

    # ==========================================================
    # STEP 20 - EVALUATION METRICS
    # ==========================================================
    evaluate_metrics(
        y_test,
        lr_predictions
    )

    # ==========================================================
    # STEP 21 - BASIC VISUALIZATIONS
    # ==========================================================
    generate_basic_plots(dataset)

    # ==========================================================
    # STEP 22 - ADVANCED GRAPHS
    # ==========================================================
    generate_advanced_graphs(dataset)


if __name__ == "__main__":
    main()