from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


def train_knn_classifier(X_train, X_test, y_train, y_test):
    """
    Train and evaluate a K-Nearest Neighbors Classifier.
    """

    print("\n" + "=" * 60)
    print("K-NEAREST NEIGHBORS (KNN)")
    print("=" * 60)

    # Create model
    model = KNeighborsClassifier(n_neighbors=5)

    # Train model
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)

    # Accuracy
    accuracy = accuracy_score(y_test, y_pred)

    print(f"\nKNN Accuracy: {accuracy:.2%}")

    return model, y_pred, accuracy