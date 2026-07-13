from preprocessing.data_loader import load_dataset


def main():

    dataset = load_dataset("dataset/student_data.csv")

    if dataset is not None:

        print("\nFirst Five Records")
        print("-" * 60)
        print(dataset.head())

        print("\nDataset Shape")
        print("-" * 60)
        print(dataset.shape)

        print("\nColumn Names")
        print("-" * 60)
        print(dataset.columns.tolist())

        print("\nDataset Information")
        print("-" * 60)
        dataset.info()

        print("\nMissing Values")
        print("-" * 60)
        print(dataset.isnull().sum())


if __name__ == "__main__":
    main()