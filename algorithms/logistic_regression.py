from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


def train_logistic_regression(X_train, X_test, y_train, y_test):
    """
    Train and evaluate a Logistic Regression Classifier.
    """

    print("\n" + "=" * 60)
    print("LOGISTIC REGRESSION")
    print("=" * 60)

    # Create model
    model = LogisticRegression(
        max_iter=1000,
        random_state=42
    )

    # Train model
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)

    # Accuracy
    accuracy = accuracy_score(y_test, y_pred)

    print(f"\nLogistic Regression Accuracy: {accuracy:.2%}")

    return model, y_pred, accuracy