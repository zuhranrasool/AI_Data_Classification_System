from preprocessing.data_loader import load_dataset
from preprocessing.data_cleaning import clean_dataset


def main():

    dataset = load_dataset("dataset/student_data.csv")

    if dataset is not None:

        dataset = clean_dataset(dataset)

        print("\nFirst Five Records")
        print(dataset.head())


if __name__ == "__main__":
    main()