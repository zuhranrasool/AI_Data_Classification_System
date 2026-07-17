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
    # Bar Chart
    # ==========================

    plt.figure(figsize=(8, 5))
    plt.bar(models, scores)

    plt.title("Model Accuracy Comparison")
    plt.xlabel("Algorithms")
    plt.ylabel("Accuracy (%)")
    plt.ylim(0, 105)

    for i, value in enumerate(scores):
        plt.text(i, value + 1, f"{value:.2f}%", ha="center")

    plt.tight_layout()
    plt.savefig("output/model_accuracy_bar_chart.png")
    plt.close()

    print("Bar Chart saved successfully.")
    print("Location: output/model_accuracy_bar_chart.png")

    # ==========================
    # Line Chart
    # ==========================

    plt.figure(figsize=(8, 5))
    plt.plot(models, scores, marker="o")

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