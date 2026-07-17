import joblib
import os


def save_model(model, filename="output/trained_model.pkl"):
    """
    Save the trained machine learning model.
    """

    print("\n" + "=" * 60)
    print("SAVE TRAINED MODEL")
    print("=" * 60)

    # Create output folder if it does not exist
    os.makedirs("output", exist_ok=True)

    # Save model
    joblib.dump(model, filename)

    print(f"\nModel saved successfully.")
    print(f"Location: {filename}")