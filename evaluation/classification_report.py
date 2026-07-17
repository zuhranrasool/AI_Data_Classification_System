import os
from sklearn.metrics import classification_report


def generate_classification_report(model, X_test, y_test, label_encoder):
    """
    Generate and save the classification report.
    """

    print("\n" + "=" * 60)
    print("CLASSIFICATION REPORT")
    print("=" * 60)

    # Predict test data
    predictions = model.predict(X_test)

    # Generate report
    report = classification_report(
        y_test,
        predictions,
        target_names=label_encoder.classes_
    )

    # Display report
    print(report)

    # Create reports folder if it doesn't exist
    os.makedirs("output/reports", exist_ok=True)

    # Save report
    with open("output/reports/model_report.txt", "w") as file:
        file.write(report)

    print("Classification report saved successfully.")
    print("Location: output/reports/model_report.txt")