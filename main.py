from preprocessing.data_loader import load_dataset
from preprocessing.data_cleaning import clean_dataset
from preprocessing.feature_engineering import feature_engineering
from preprocessing.split_dataset import split_dataset

from algorithms.decision_tree import train_decision_tree
from algorithms.logistic_regression import train_logistic_regression
from algorithms.knn_classifier import train_knn_classifier
from algorithms.random_forest import train_random_forest


def main():

    # Step 6 - Load Dataset
    dataset = load_dataset("dataset/student_data.csv")

    if dataset is not None:

        # Step 7 - Data Cleaning
        dataset = clean_dataset(dataset)

        # Step 8 - Feature Engineering
        X, y, label_encoder = feature_engineering(dataset)

        # Step 9 - Train/Test Split
        X_train, X_test, y_train, y_test = split_dataset(X, y)

        # Step 10 - Decision Tree
        dt_model, dt_predictions, dt_accuracy = train_decision_tree(
            X_train,
            X_test,
            y_train,
            y_test
        )

        # Step 11 - Logistic Regression
        lr_model, lr_predictions, lr_accuracy = train_logistic_regression(
            X_train,
            X_test,
            y_train,
            y_test
        )

        # Step 12 - KNN
        knn_model, knn_predictions, knn_accuracy = train_knn_classifier(
            X_train,
            X_test,
            y_train,
            y_test
        )

        # Step 13 - Random Forest
        rf_model, rf_predictions, rf_accuracy = train_random_forest(
            X_train,
            X_test,
            y_train,
            y_test
        )

        # Step 14 - Compare Models
        print("\n" + "=" * 60)
        print("MODEL ACCURACY SUMMARY")
        print("=" * 60)

        print(f"Decision Tree        : {dt_accuracy:.2%}")
        print(f"Logistic Regression  : {lr_accuracy:.2%}")
        print(f"KNN                  : {knn_accuracy:.2%}")
        print(f"Random Forest        : {rf_accuracy:.2%}")

        accuracies = {
            "Decision Tree": dt_accuracy,
            "Logistic Regression": lr_accuracy,
            "KNN": knn_accuracy,
            "Random Forest": rf_accuracy
        }

        best_model = max(accuracies, key=accuracies.get)

        print("\n" + "=" * 60)
        print("BEST MODEL SELECTION")
        print("=" * 60)
        print(f"Selected Model : {best_model}")
        print(f"Accuracy       : {accuracies[best_model]:.2%}")


if __name__ == "__main__":
    main()