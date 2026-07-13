import pandas as pd


def load_dataset(file_path):
    """
    Load dataset from CSV file.
    """

    try:
        data = pd.read_csv(file_path)

        print("=" * 60)
        print("Dataset Loaded Successfully")
        print("=" * 60)

        return data

    except FileNotFoundError:
        print("Dataset file not found.")
        return None