from preprocessing.data_loader import load_dataset
from preprocessing.data_cleaning import clean_dataset
from preprocessing.feature_engineering import feature_engineering
from preprocessing.split_dataset import split_dataset


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

        print("\nFirst Five Training Records")
        print(X_train.head())

        print("\nFirst Five Testing Records")
        print(X_test.head())


if __name__ == "__main__":
    main()