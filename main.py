from preprocessing.data_loader import load_dataset
from preprocessing.data_cleaning import clean_dataset
from preprocessing.feature_engineering import feature_engineering
from preprocessing.split_dataset import split_dataset

from algorithms.decision_tree import train_decision_tree
from algorithms.logistic_regression import train_logistic_regression
from algorithms.knn_classifier import train_knn_classifier
from algorithms.random_forest import train_random_forest

from evaluation.accuracy import evaluate_accuracy

from models.save_model import save_model
from models.predict import predict_students


def main():

    # ============================================================
    # Step 6 - Load Dataset
    # ============================================================
    dataset = load_dataset("dataset/student_data.csv")

    if dataset is not None:

        # ============================================================
        # Step 7 - Data Cleaning
        # ============================================================
        dataset = clean_dataset(dataset)

        # ============================================================
        # Step 8 - Feature Engineering
        # ============================================================
        X, y, label_encoder = feature_engineering(dataset)

        # ============================================================
        # Step 9 - Train/Test Split
        # ============================================================
        X_train, X_test, y_train, y_test = split_dataset(X, y)

        # ============================================================
        # Step 10 - Decision Tree
        # ============================================================
        dt_model, dt_predictions, dt_accuracy = train_decision_tree(
            X_train,
            X_test,
            y_train,
            y_test
        )

        # ============================================================
        # Step 11 - Logistic Regression
        # ============================================================
        lr_model, lr_predictions, lr_accuracy = train_logistic_regression(
            X_train,
            X_test,
            y_train,
            y_test
        )

        # ============================================================
        # Step 12 - KNN
        # ============================================================
        knn_model, knn_predictions, knn_accuracy = train_knn_classifier(
            X_train,
            X_test,
            y_train,
            y_test
        )

        # ============================================================
        # Step 13 - Random Forest
        # ============================================================
        rf_model, rf_predictions, rf_accuracy = train_random_forest(
            X_train,
            X_test,
            y_train,
            y_test
        )

        # ============================================================
        # Step 17 - Accuracy Evaluation
        # ============================================================
        evaluate_accuracy(
            dt_accuracy,
            lr_accuracy,
            knn_accuracy,
            rf_accuracy
        )

        # Determine the best model
        accuracies = {
            "Decision Tree": dt_accuracy,
            "Logistic Regression": lr_accuracy,
            "KNN": knn_accuracy,
            "Random Forest": rf_accuracy
        }

        best_model = max(accuracies, key=accuracies.get)

        # ============================================================
        # Step 15 - Save Best Model
        # ============================================================
        if best_model == "Decision Tree":
            save_model(dt_model)

        elif best_model == "Logistic Regression":
            save_model(lr_model)

        elif best_model == "KNN":
            save_model(knn_model)

        elif best_model == "Random Forest":
            save_model(rf_model)

        # ============================================================
        # Step 16 - Prediction Module
        # ============================================================
        predict_students()


if __name__ == "__main__":
    main()