from sklearn.model_selection import train_test_split


def split_dataset(X, y):
    """
    Split the dataset into training and testing sets.
    """

    print("\n" + "=" * 60)
    print("TRAIN / TEST SPLIT")
    print("=" * 60)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    print(f"\nTraining Features Shape : {X_train.shape}")
    print(f"Testing Features Shape  : {X_test.shape}")
    print(f"Training Labels Shape   : {y_train.shape}")
    print(f"Testing Labels Shape    : {y_test.shape}")

    return X_train, X_test, y_train, y_test