import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix


def generate_confusion_matrix(model, X_test, y_test, label_encoder):
    """
    Generate and save the confusion matrix.
    """

    print("\n" + "=" * 60)
    print("CONFUSION MATRIX")
    print("=" * 60)

    # Make predictions
    predictions = model.predict(X_test)

    # Generate confusion matrix
    cm = confusion_matrix(y_test, predictions)

    # Create display
    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=label_encoder.classes_
    )

    fig, ax = plt.subplots(figsize=(8, 6))
    display.plot(ax=ax, cmap="Blues", colorbar=False)

    plt.title("Confusion Matrix")

    # Save image
    plt.savefig("output/confusion_matrix.png")

    print("Confusion Matrix saved successfully.")
    print("Location: output/confusion_matrix.png")

    plt.close()