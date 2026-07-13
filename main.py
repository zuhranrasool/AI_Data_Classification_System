from preprocessing.data_loader import load_dataset
from preprocessing.data_cleaning import clean_dataset
from preprocessing.feature_engineering import feature_engineering


def main():

    dataset = load_dataset("dataset/student_data.csv")

    if dataset is not None:

        # Step 7
        dataset = clean_dataset(dataset)

        # Step 8
        X, y, label_encoder = feature_engineering(dataset)

        print("\nFirst Five Feature Rows")
        print(X.head())

        print("\nFirst Five Target Values")
        print(y.head())


if __name__ == "__main__":
    main()