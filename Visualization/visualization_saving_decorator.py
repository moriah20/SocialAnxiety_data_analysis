import logging
import os
from functools import wraps
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


def auto_save_plot(func=None, *, output_dir=None):
    """
    Hybrid decorator that supports BOTH usage styles:

    1. Old style (decorator with arguments):
        @auto_save_plot(output_dir="my_folder")
        def plot():
            ...

    2. New style (decorator without arguments):
        @auto_save_plot
        def plot():
            ...

    3. Runtime override:
        plot(output_dir="another_folder")

    Behavior:
    - Automatically saves all matplotlib figures created inside the wrapped function.
    - Supports multiple figures, suptitles, axes titles.
    - Sanitizes filenames.
    - Logs all saved files.
    """

    # Case 1: decorator called WITH arguments → return actual decorator
    if func is None:
        def decorator(inner_func):
            return auto_save_plot(inner_func, output_dir=output_dir)
        return decorator

    # Case 2: decorator called WITHOUT arguments → func is the wrapped function
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Runtime override takes priority
        runtime_output_dir = kwargs.pop("output_dir", None)

        # Determine final output directory:
        # 1. runtime override
        # 2. decorator argument
        # 3. default folder
        final_output_dir = (
            runtime_output_dir
            or output_dir
            or "plots"
        )

        # Close any previously open figures to avoid cross-contamination
        plt.close("all")

        try:
            # Execute the wrapped function
            result = func(*args, **kwargs)

            # Collect all figures created during execution
            figure_numbers = plt.get_fignums()
            figures = [plt._pylab_helpers.Gcf.figs[num].canvas.figure for num in figure_numbers]

            if not figures:
                logger.warning(
                    f"No figures were generated inside function '{func.__name__}'."
                )
                return result

            # Ensure output directory exists
            os.makedirs(final_output_dir, exist_ok=True)

            for i, fig in enumerate(figures, start=1):
                # Determine a meaningful title for the saved file
                title = ""

                # 1. Try suptitle
                if fig._suptitle is not None:
                    title = fig._suptitle.get_text()

                # 2. Try axes title
                if not title and fig.axes:
                    title = fig.axes[0].get_title()

                # 3. Fallback title
                if not title:
                    title = f"{func.__name__}_figure_{i}"

                # Sanitize title for filesystem compatibility
                safe_title = "".join(
                    c for c in title if c.isalnum() or c in (" ", "_", "-")
                ).rstrip()

                filename = f"{safe_title}.png"
                full_path = os.path.join(final_output_dir, filename)

                # Save the figure
                fig.savefig(full_path)
                logger.info(f"Saved figure to: {full_path}")

            return result

        except Exception as e:
            logger.error(
                f"Error occurred while executing or saving plots in '{func.__name__}': {e}",
                exc_info=True
            )
            raise

    return wrapper
