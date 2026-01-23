import os
import shutil
import matplotlib.pyplot as plt
from Visualization.visualization_saving_decorator import auto_save_plot   # Replace with the actual module name


# Example function that creates two separate figures
@auto_save_plot(output_dir="Visualization_Test")
def create_two_plots():
    # --- First figure ---
    fig1 = plt.figure()
    plt.plot([1, 2, 3], [4, 5, 6])
    fig1.suptitle("First Plot")

    # --- Second figure ---
    fig2 = plt.figure()
    plt.plot([10, 20, 30], [40, 50, 60])
    fig2.suptitle("Second Plot")


def test_auto_save_plot_saves_all_figures():
    """
    Test that the decorator:
    - Creates the output directory
    - Saves ALL figures generated inside the wrapped function
    - Uses the correct filenames based on suptitles
    - Produces non-empty image files
    """

    # --- Clean old test directory safely ---
    if os.path.exists("Visualization_Test"):
        try:
            shutil.rmtree("Visualization_Test")
        except PermissionError:
            # Windows sometimes locks folders; ignore and continue
            pass

    # --- Run the function that generates two figures ---
    create_two_plots()

    # --- Assertions ---
    assert os.path.exists("Visualization_Test")

    files = os.listdir("Visualization_Test")
    assert "First Plot.png" in files
    assert "Second Plot.png" in files

    for f in files:
        full_path = os.path.join("Visualization_Test", f)
        assert os.path.getsize(full_path) > 0
