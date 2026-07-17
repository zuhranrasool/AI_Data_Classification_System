import joblib
import pandas as pd


def predict_students(
    model_path="output/trained_model.pkl",
    test_file="dataset/sample_test_data.csv",
    output_file="output/predictions.csv"
):
    """
    Load the trained model and predict student performance.
    """

    print("\n" + "=" * 60)
    print("PREDICTION MODULE")
    print("=" * 60)

    # Load trained model
    model = joblib.load(model_path)

    # Load test dataset
    test_data = pd.read_csv(test_file)

    # Remove Student_ID if present
    if "Student_ID" in test_data.columns:
        student_ids = test_data["Student_ID"]
        X_test = test_data.drop(columns=["Student_ID"])
    else:
        student_ids = None
        X_test = test_data.copy()

    # Predict
    predictions = model.predict(X_test)

    # Convert numeric predictions to labels
    label_map = {
        0: "Average",
        1: "Excellent",
        2: "Good",
        3: "Poor"
    }

    predicted_labels = [label_map[p] for p in predictions]

    # Save predictions
    result = test_data.copy()

    if student_ids is not None:
        result["Student_ID"] = student_ids

    result["Predicted_Performance"] = predicted_labels

    result.to_csv(output_file, index=False)

    print("\nPredictions completed successfully.")
    print(f"Results saved to: {output_file}")

    print("\nFirst Five Predictions")
    print(result.head())

    return result