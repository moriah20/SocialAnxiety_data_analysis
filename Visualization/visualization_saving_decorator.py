import logging
import os
from functools import wraps
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

def auto_save_plot(output_dir):
    """
    Decorator that automatically saves ALL matplotlib figures created inside the
    wrapped function. It supports:
        - Multiple figures
        - FacetGrid objects
        - Suptitles
        - Axes titles
    The wrapped function does NOT need to return anything.
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                # Close previous figures
                plt.close('all')

                # Run the plotting function
                func(*args, **kwargs)

                # Get ALL open figures
                figures = list(map(plt.figure, plt.get_fignums()))

                if not figures:
                    logger.warning("No figures were created.")
                    return

                # Ensure directory exists
                os.makedirs(output_dir, exist_ok=True)

                for i, fig in enumerate(figures, start=1):

                    # --- Extract title ---
                    title = ""

                    # 1. Try suptitle
                    if fig._suptitle is not None:
                        title = fig._suptitle.get_text()

                    # 2. Try axes title
                    if not title and fig.axes:
                        title = fig.axes[0].get_title()

                    # 3. Fallback
                    if not title:
                        title = f"figure_{i}"

                    # Clean title for filename
                    safe_title = "".join(
                        c for c in title if c.isalnum() or c in (' ', '_', '-')
                    ).rstrip()

                    filename = f"{safe_title}.png"
                    full_path = os.path.join(output_dir, filename)

                    # Save figure
                    fig.savefig(full_path)
                    logger.info(f"Saved figure: {full_path}")

                # Show all figures
                #plt.show()

            except Exception as e:
                logger.error(f"Error while saving plot: {e}", exc_info=True)
                raise

        return wrapper

    return decorator
