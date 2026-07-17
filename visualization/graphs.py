import os
import matplotlib.pyplot as plt


def generate_advanced_graphs(dataset):
    """
    Generate advanced graphs for the dataset.
    """

    print("\n" + "=" * 60)
    print("ADVANCED GRAPHS")
    print("=" * 60)

    # Create output folder
    os.makedirs("output", exist_ok=True)

    # ==========================================================
    # Histogram
    # ==========================================================
    plt.figure(figsize=(8, 5))

    plt.hist(
        dataset["Study_Hours"],
        bins=10
    )

    plt.title("Distribution of Study Hours")
    plt.xlabel("Study Hours")
    plt.ylabel("Number of Students")

    plt.tight_layout()
    plt.savefig("output/study_hours_histogram.png")
    plt.close()

    # ==========================================================
    # Scatter Plot
    # ==========================================================
    plt.figure(figsize=(8, 5))

    plt.scatter(
        dataset["Study_Hours"],
        dataset["Final"]
    )

    plt.title("Study Hours vs Final Marks")
    plt.xlabel("Study Hours")
    plt.ylabel("Final Marks")

    plt.tight_layout()
    plt.savefig("output/study_hours_vs_final.png")
    plt.close()

    # ==========================================================
    # Box Plot
    # ==========================================================
    plt.figure(figsize=(8, 5))

    plt.boxplot(dataset["Final"])

    plt.title("Final Marks Distribution")
    plt.ylabel("Final Marks")

    plt.tight_layout()
    plt.savefig("output/final_marks_boxplot.png")
    plt.close()

    print("\nHistogram saved successfully.")
    print("Location: output/study_hours_histogram.png")

    print("\nScatter Plot saved successfully.")
    print("Location: output/study_hours_vs_final.png")

    print("\nBox Plot saved successfully.")
    print("Location: output/final_marks_boxplot.png")