from sklearn.preprocessing import LabelEncoder


def feature_engineering(data):
    """
    Perform feature engineering:
    - Encode target labels
    - Separate features and target
    """

    print("\n" + "=" * 60)
    print("FEATURE ENGINEERING")
    print("=" * 60)

    # Copy dataset
    dataset = data.copy()

    # Encode Performance column
    label_encoder = LabelEncoder()
    dataset["Performance"] = label_encoder.fit_transform(dataset["Performance"])

    print("\nPerformance Label Mapping:")

    for index, label in enumerate(label_encoder.classes_):
        print(f"{label} -> {index}")

    # Features (X)
    X = dataset.drop(columns=["Student_ID", "Performance"])

    # Target (y)
    y = dataset["Performance"]

    print("\nFeature Matrix Shape:", X.shape)
    print("Target Vector Shape:", y.shape)

    return X, y, label_encoder