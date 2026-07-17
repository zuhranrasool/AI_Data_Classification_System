import os
import matplotlib.pyplot as plt


def generate_basic_plots(dataset):
    """
    Generate basic visualizations for the dataset.
    """

    print("\n" + "=" * 60)
    print("BASIC VISUALIZATIONS")
    print("=" * 60)

    # Create output folder
    os.makedirs("output", exist_ok=True)

    # Count students in each performance category
    performance_counts = dataset["Performance"].value_counts()

    # -----------------------------
    # Bar Chart
    # -----------------------------
    plt.figure(figsize=(8, 5))

    performance_counts.plot(kind="bar")

    plt.title("Student Performance Distribution")
    plt.xlabel("Performance")
    plt.ylabel("Number of Students")

    plt.tight_layout()

    plt.savefig("output/performance_bar_chart.png")

    plt.close()

    # -----------------------------
    # Pie Chart
    # -----------------------------
    plt.figure(figsize=(6, 6))

    performance_counts.plot(
        kind="pie",
        autopct="%1.1f%%",
        startangle=90
    )

    plt.ylabel("")
    plt.title("Performance Distribution")

    plt.tight_layout()

    plt.savefig("output/performance_pie_chart.png")

    plt.close()

    print("\nBar Chart saved successfully.")
    print("Location: output/performance_bar_chart.png")

    print("\nPie Chart saved successfully.")
    print("Location: output/performance_pie_chart.png")