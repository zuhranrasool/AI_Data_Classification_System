import pandas as pd


def clean_dataset(data):
    """
    Clean the dataset by checking missing values,
    duplicates, data types, and invalid values.
    """

    print("\n" + "=" * 60)
    print("DATA CLEANING")
    print("=" * 60)

    # -----------------------------
    # Missing Values
    # -----------------------------
    print("\nMissing Values:")
    print(data.isnull().sum())

    # -----------------------------
    # Duplicate Records
    # -----------------------------
    duplicates = data.duplicated().sum()
    print(f"\nDuplicate Records: {duplicates}")

    if duplicates > 0:
        data = data.drop_duplicates()
        print("Duplicate records removed successfully.")
    else:
        print("No duplicate records found.")

    # -----------------------------
    # Data Types
    # -----------------------------
    print("\nData Types:")
    print(data.dtypes)

    # -----------------------------
    # Invalid Value Check
    # -----------------------------
    numeric_columns = [
        "Study_Hours",
        "Attendance",
        "Assignment",
        "Midterm",
        "Final"
    ]

    print("\nChecking for invalid values...")

    for column in numeric_columns:
        invalid = (data[column] < 0).sum()

        if invalid > 0:
            print(f"{column}: {invalid} invalid values found.")
        else:
            print(f"{column}: OK")

    print("\nData Cleaning Completed Successfully.")

    return data