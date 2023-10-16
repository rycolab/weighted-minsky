import numpy as np


def H(x: np.ndarray) -> np.ndarray:
    """The hard-thresholding (Heaviside) function.

    Args:
        x (np.ndarray): The input.

    Returns:
        np.ndarray: The output.
    """
    return (x > 0).astype(int)
