import logging
import os
from functools import wraps
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


def auto_save_plot(output_dir):
    """
    Decorator that automatically saves all matplotlib figures created inside the
    wrapped function. Supports:
        - Multiple figures
        - FacetGrid objects
        - Suptitles and axes titles
    The wrapped function may return a value, and this decorator will preserve it.
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Close any previously open figures to avoid cross-contamination
            plt.close('all')

            try:
                # Execute the wrapped function and capture its return value
                result = func(*args, **kwargs)

                # Collect all figures created during execution
                figure_numbers = plt.get_fignums()
                figures = [plt.figure(num) for num in figure_numbers]

                if not figures:
                    logger.warning(
                        f"No figures were generated inside function '{func.__name__}'."
                    )
                    return result

                # Ensure output directory exists
                os.makedirs(output_dir, exist_ok=True)

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
                        c for c in title if c.isalnum() or c in (' ', '_', '-')
                    ).rstrip()

                    filename = f"{safe_title}.png"
                    full_path = os.path.join(output_dir, filename)

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

    return decorator
