import os
import matplotlib
matplotlib.use("Agg")  # Force non-GUI backend for tests
import matplotlib.pyplot as plt
from Visualization.visualization_saving_decorator import auto_save_plot


# ----------------------------------------------------------------------
# Helper functions decorated with auto_save_plot
# ----------------------------------------------------------------------

@auto_save_plot
def create_two_plots():
    """Creates two figures with suptitles."""
    fig1 = plt.figure()
    plt.plot([1, 2, 3], [4, 5, 6])
    fig1.suptitle("First Plot")

    fig2 = plt.figure()
    plt.plot([10, 20, 30], [40, 50, 60])
    fig2.suptitle("Second Plot")


@auto_save_plot
def no_plots():
    """Creates no figures at all."""
    return "ok"


@auto_save_plot
def plot_with_illegal_title():
    """Creates a figure with illegal filename characters."""
    fig = plt.figure()
    plt.plot([1, 2], [3, 4])
    fig.suptitle("Bad:/\\*?Title")


@auto_save_plot
def plot_without_title():
    """Creates a figure without any title or suptitle."""
    plt.figure()
    plt.plot([1, 2], [3, 4])


@auto_save_plot
def single_plot():
    """Creates exactly one figure."""
    fig = plt.figure()
    plt.plot([1, 2], [3, 4])
    fig.suptitle("Only One")


@auto_save_plot
def failing_plot():
    """Creates a figure but raises an exception."""
    plt.figure()
    raise ValueError("Test error")


# ----------------------------------------------------------------------
# Test cases
# ----------------------------------------------------------------------

def test_auto_save_plot_saves_all_figures(tmp_path):
    """
    Ensures:
    - All figures created inside the function are saved
    - Filenames match the suptitles
    - Output is stored inside pytest's temporary directory
    """
    output_dir = tmp_path / "plots"
    create_two_plots(output_dir=str(output_dir))

    assert output_dir.exists()

    files = os.listdir(output_dir)
    assert "First Plot.png" in files
    assert "Second Plot.png" in files

    for f in files:
        assert (output_dir / f).stat().st_size > 0


def test_no_figures_generated(tmp_path):
    """
    Ensures:
    - No directory is created when no figures are generated
    - No exception is raised
    """
    output_dir = tmp_path / "plots"
    result = no_plots(output_dir=str(output_dir))

    assert result == "ok"
    assert not output_dir.exists()


def test_illegal_characters_in_title(tmp_path):
    """
    Ensures:
    - Illegal characters in titles are sanitized
    - File is saved with a safe filename
    """
    output_dir = tmp_path / "plots"
    plot_with_illegal_title(output_dir=str(output_dir))

    files = os.listdir(output_dir)

    # Sanitized version should remove illegal characters
    assert any("BadTitle.png" in f or "Bad Title.png" in f for f in files)


def test_plot_without_title(tmp_path):
    """
    Ensures:
    - When no title exists, fallback filename is used
    """
    output_dir = tmp_path / "plots"
    plot_without_title(output_dir=str(output_dir))

    files = os.listdir(output_dir)

    assert "plot_without_title_figure_1.png" in files


def test_single_figure(tmp_path):
    """
    Ensures:
    - Exactly one figure is saved
    - Filename matches the suptitle
    """
    output_dir = tmp_path / "plots"
    single_plot(output_dir=str(output_dir))

    files = os.listdir(output_dir)

    assert len(files) == 1
    assert "Only One.png" in files


def test_exception_inside_function(tmp_path):
    """
    Ensures:
    - The original exception is re-raised
    - No directory is created when the function fails
    """
    output_dir = tmp_path / "plots"

    try:
        failing_plot(output_dir=str(output_dir))
    except ValueError:
        pass  # Expected
    else:
        assert False, "Exception was not raised"

    assert not output_dir.exists()
