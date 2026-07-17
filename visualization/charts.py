import os
import matplotlib.pyplot as plt


def generate_performance_charts(accuracies):
    """
    Generate performance comparison charts for all models.
    """

    print("\n" + "=" * 60)
    print("PERFORMANCE CHARTS")
    print("=" * 60)

    # Create output folder
    os.makedirs("output", exist_ok=True)

    models = list(accuracies.keys())
    scores = [accuracy * 100 for accuracy in accuracies.values()]

    # ==========================
    # Bar Chart (Fixed name to match app.py)
    # ==========================

    plt.figure(figsize=(8, 5))
    plt.bar(models, scores, color=['#4AF2A1', '#4A90E2', '#50E3C2', '#B8E986'])

    plt.title("Model Accuracy Comparison")
    plt.xlabel("Algorithms")
    plt.ylabel("Accuracy (%)")
    plt.ylim(0, 105)

    for i, value in enumerate(scores):
        plt.text(i, value + 1, f"{value:.2f}%", ha="center")

    plt.tight_layout()
    # CHANGED: Saved as accuracy_graph.png to match your app.py path exactly
    plt.savefig("output/accuracy_graph.png")
    plt.close()

    print("Bar Chart saved successfully.")
    print("Location: output/accuracy_graph.png")

    # ==========================
    # Line Chart
    # ==========================

    plt.figure(figsize=(8, 5))
    plt.plot(models, scores, marker="o", color="#F5A623", linewidth=2)

    plt.title("Model Accuracy Comparison")
    plt.xlabel("Algorithms")
    plt.ylabel("Accuracy (%)")
    plt.ylim(0, 105)
    plt.grid(True)

    plt.tight_layout()
    plt.savefig("output/model_accuracy_line_chart.png")
    plt.close()

    print("\nLine Chart saved successfully.")
    print("Location: output/model_accuracy_line_chart.png")


# =====================================================
# Standalone Execution block for manual terminal running
# =====================================================
if __name__ == "__main__":
    # Test dictionary using the accuracy data from your evaluation setup
    sample_accuracies = {
        "Decision Tree": 0.9750,
        "Logistic Regression": 1.0000,
        "KNN": 1.0000,
        "Random Forest": 1.0000
    }
    generate_performance_charts(sample_accuracies)