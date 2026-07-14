from preprocessing.data_loader import load_dataset
from preprocessing.data_cleaning import clean_dataset
from preprocessing.feature_engineering import feature_engineering
from preprocessing.split_dataset import split_dataset
from algorithms.decision_tree import train_decision_tree


def main():

    # Step 6
    dataset = load_dataset("dataset/student_data.csv")

    if dataset is not None:

        # Step 7
        dataset = clean_dataset(dataset)

        # Step 8
        X, y, label_encoder = feature_engineering(dataset)

        # Step 9
        X_train, X_test, y_train, y_test = split_dataset(X, y)

        # Step 10
        model, y_pred, accuracy = train_decision_tree(
            X_train,
            X_test,
            y_train,
            y_test
        )

        print("\nFirst 10 Predictions")
        print(y_pred[:10])


if __name__ == "__main__":
    main()