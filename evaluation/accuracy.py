def evaluate_accuracy(dt_accuracy, lr_accuracy, knn_accuracy, rf_accuracy):
    """
    Display the accuracy of all trained models.
    """

    print("\n" + "=" * 60)
    print("MODEL ACCURACY EVALUATION")
    print("=" * 60)

    print(f"Decision Tree        : {dt_accuracy:.2%}")
    print(f"Logistic Regression  : {lr_accuracy:.2%}")
    print(f"K-Nearest Neighbors  : {knn_accuracy:.2%}")
    print(f"Random Forest        : {rf_accuracy:.2%}")

    accuracies = {
        "Decision Tree": dt_accuracy,
        "Logistic Regression": lr_accuracy,
        "K-Nearest Neighbors": knn_accuracy,
        "Random Forest": rf_accuracy
    }

    best_model = max(accuracies, key=accuracies.get)

    print("\n" + "=" * 60)
    print("BEST MODEL BASED ON ACCURACY")
    print("=" * 60)
    print(f"Model    : {best_model}")
    print(f"Accuracy : {accuracies[best_model]:.2%}")