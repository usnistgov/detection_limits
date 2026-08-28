import pytest
import numpy as np
import sys
import os

# Ensure src is in the path so detection_limits can be imported
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

@pytest.fixture
def synthetic_data():
    """
    Generates synthetic image data for testing.
    Returns:
        tuple: (perfect_image, noisy_image, mask, noise_std)
    """
    shape = (100, 100)
    mask = np.zeros(shape, dtype=np.uint8)
    # Create a square foreground in the center
    mask[25:75, 25:75] = 1
    
    # Perfect image: Background = 100, Foreground = 200
    perfect_image = np.full(shape, 100, dtype=np.float64)
    perfect_image[mask == 1] = 200
    
    # Noisy image: Add Gaussian noise
    noise_std = 10.0
    np.random.seed(42) # Ensure reproducibility
    noise = np.random.normal(0, noise_std, shape)
    noisy_image = perfect_image + noise
    
    return perfect_image, noisy_image, mask, noise_std
