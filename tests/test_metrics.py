import pytest
import numpy as np
from detection_limits.metrics import calculate_all_snr_with_mask, calculate_all_metrics

def test_calculate_all_snr_with_mask_perfect(synthetic_data):
    """
    Test SNR calculation on a perfect image (no noise).
    """
    perfect_image, _, mask, _ = synthetic_data
    
    # For a perfect image, variance should be 0 (or very close to it)
    # We pass a small noise value to avoid division by zero in some formulas if needed, 
    # though the function handles some 0 cases.
    noise_param = 1e-10 
    
    snr_metrics = calculate_all_snr_with_mask(perfect_image, mask, noise=noise_param)
    
    # Unpack results (14 values)
    (snr1, snr2, snr3, snr4, snr5, snr6, snr7, snr8, snr9, snr10, 
     fg_mean, bg_mean, fg_var, bg_var) = snr_metrics
    
    # Check basic statistics
    assert np.isclose(fg_mean, 200.0)
    assert np.isclose(bg_mean, 100.0)
    assert np.isclose(fg_var, 0.0)
    assert np.isclose(bg_var, 0.0)
    
    # SNR1 = fg_var / bg_var. If bg_var is 0, it returns 0.
    assert snr1 == 0 

def test_calculate_all_snr_with_mask_noisy(synthetic_data):
    """
    Test SNR calculation on a noisy image.
    """
    _, noisy_image, mask, noise_std = synthetic_data
    
    snr_metrics = calculate_all_snr_with_mask(noisy_image, mask, noise=noise_std)
    
    (snr1, snr2, snr3, snr4, snr5, snr6, snr7, snr8, snr9, snr10, 
     fg_mean, bg_mean, fg_var, bg_var) = snr_metrics
    
    # Check if means are approximately correct (within standard error)
    # Standard error of mean = std / sqrt(N)
    # N_fg = 50*50 = 2500, N_bg = 10000 - 2500 = 7500
    assert np.isclose(fg_mean, 200.0, atol=3*noise_std/np.sqrt(2500))
    assert np.isclose(bg_mean, 100.0, atol=3*noise_std/np.sqrt(7500))
    
    # Check variances are approximately noise_std^2 = 100
    assert np.isclose(fg_var, noise_std**2, rtol=0.2) # Allow 20% error for variance estimation
    assert np.isclose(bg_var, noise_std**2, rtol=0.2)
    
    # Check SNR5: foreground_mean / noise
    expected_snr5 = 200.0 / noise_std
    assert np.isclose(snr5, expected_snr5, rtol=0.1)

def test_calculate_all_metrics_structure(synthetic_data):
    """
    Test that calculate_all_metrics returns the expected dictionary structure.
    """
    perfect_image, noisy_image, mask, noise_std = synthetic_data
    
    metrics = calculate_all_metrics(
        image=noisy_image,
        mask=mask,
        reference_image=perfect_image,
        noise_val=noise_std,
        contrast_val=100,
        intensityimage_filename="test_image.tif",
        set_index=1
    )
    
    expected_keys = [
        "IMAGE-NAME", "Set_index", "Noise_level", "Contrast_level",
        "SNR1", "SNR2", "SNR3", "SNR4", "SNR5", "SNR6", "SNR7", "SNR8", "SNR9", "SNR10",
        "Foreground_mean", "Background_mean", "Foreground_var", "Background_var",
        "Mean_intensity", "Std_intensity", "Variance_intensity",
        "Michelson_contrast", "RMS_contrast", "SSIM", "PSNR",
        "Edge_density", "MI", "NMI", "CE"
    ]
    
    for key in expected_keys:
        assert key in metrics
    
    # Check specific values
    assert metrics["Noise_level"] == noise_std
    assert metrics["Contrast_level"] == 100
    assert metrics["IMAGE-NAME"] == "test_image.ome.tif"
