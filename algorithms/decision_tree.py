from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


def train_decision_tree(X_train, X_test, y_train, y_test):
    """
    Train and evaluate a Decision Tree Classifier.
    """

    print("\n" + "=" * 60)
    print("DECISION TREE CLASSIFIER")
    print("=" * 60)

    # Create model
    model = DecisionTreeClassifier(random_state=42)

    # Train model
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)

    # Accuracy
    accuracy = accuracy_score(y_test, y_pred)

    print(f"\nDecision Tree Accuracy: {accuracy:.2%}")

    return model, y_pred, accuracy