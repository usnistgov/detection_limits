import pytest
import numpy as np
from detection_limits.metrics import calculate_all_snr_with_mask, calculate_all_metrics, metrics
import os
import pandas as pd
from skimage import io

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

def test_calculate_all_snr_with_mask_uniform_images(synthetic_data):
    """
    Test SNR calculation on uniform images (all white or all black).
    These are edge cases where variance is 0.
    """
    _, _, mask, _ = synthetic_data
    shape = mask.shape
    
    # Case 1: Uniform White Image
    white_image = np.full(shape, 255, dtype=np.float64)
    snr_metrics = calculate_all_snr_with_mask(white_image, mask, noise=1.0)
    
    (snr1, snr2, snr3, snr4, snr5, snr6, snr7, snr8, snr9, snr10, 
     fg_mean, bg_mean, fg_var, bg_var) = snr_metrics
    
    assert fg_mean == 255.0
    assert bg_mean == 255.0
    assert fg_var == 0.0
    assert bg_var == 0.0
    
    # Check all SNRs for Uniform White Image
    assert snr1 == 0 # bg_var is 0
    assert snr2 == 0 # fg_var is 0
    assert snr3 == 0 # bg_std is 0
    assert snr4 == 0 # bg_std is 0
    assert snr5 == 255.0 # fg_mean / noise
    assert snr6 == 255.0**2 # fg_mean^2 / noise^2
    assert snr7 == 0.0 # fg_var / noise^2
    assert np.isclose(snr8, np.sqrt(255.0)) # sqrt(fg_mean) / noise
    assert snr9 == 0 # bg_std is 0
    assert snr10 == 0.0 # (fg_mean - bg_mean) / noise = 0

    # Case 2: Uniform Black Image
    black_image = np.zeros(shape, dtype=np.float64)
    snr_metrics = calculate_all_snr_with_mask(black_image, mask, noise=1.0)
    
    (snr1, snr2, snr3, snr4, snr5, snr6, snr7, snr8, snr9, snr10, 
     fg_mean, bg_mean, fg_var, bg_var) = snr_metrics
     
    assert fg_mean == 0.0
    assert bg_mean == 0.0
    assert fg_var == 0.0
    assert bg_var == 0.0
    
    # Check all SNRs for Uniform Black Image
    assert snr1 == 0
    assert snr2 == 0
    assert snr3 == 0
    assert snr4 == 0
    assert snr5 == 0.0
    assert snr6 == 0.0
    assert snr7 == 0.0
    assert snr8 == 0.0
    assert snr9 == 0
    assert snr10 == 0.0

def test_calculate_all_snr_with_mask_half_half():
    """
    Test SNR calculation on a perfect half-white, half-black image.
    This tests infinite contrast / zero variance background cases.
    """
    shape = (100, 100)
    image = np.zeros(shape, dtype=np.float64)
    image[:, :50] = 255 # Left half white
    
    mask = np.zeros(shape, dtype=np.uint8)
    mask[:, :50] = 1 # Left half foreground (White)
    
    # FG = White (255), BG = Black (0)
    snr_metrics = calculate_all_snr_with_mask(image, mask, noise=1.0)
    
    (snr1, snr2, snr3, snr4, snr5, snr6, snr7, snr8, snr9, snr10, 
     fg_mean, bg_mean, fg_var, bg_var) = snr_metrics
    
    assert fg_mean == 255.0
    assert bg_mean == 0.0
    assert fg_var == 0.0
    assert bg_var == 0.0
    
    # Check all SNRs for Half-Half Image
    assert snr1 == 0 # bg_var is 0
    assert snr2 == 0 # fg_var is 0
    assert snr3 == 0 # bg_std is 0
    assert snr4 == 0 # bg_std is 0
    assert snr5 == 255.0 # fg_mean / noise
    assert snr6 == 255.0**2 # fg_mean^2 / noise^2
    assert snr7 == 0.0 # fg_var / noise^2
    assert np.isclose(snr8, np.sqrt(255.0)) # sqrt(fg_mean) / noise
    assert snr9 == 0 # bg_std is 0
    assert snr10 == 255.0 # (fg_mean - bg_mean) / noise = 255/1

def test_metrics_batch_processing(tmp_path, synthetic_data):
    """
    Test the batch processing 'metrics' function.
    """
    perfect_image, _, mask, _ = synthetic_data

    # Create directories
    input_intensity_dir = tmp_path / "intensity"
    input_mask_dir = tmp_path / "mask"
    output_dir = tmp_path / "output"
    input_intensity_dir.mkdir()
    input_mask_dir.mkdir()
    output_dir.mkdir()
    
    output_csv = output_dir / "results.csv"
    
    # Create dummy data
    # We need filenames that match the parsing logic: ...noise_XXX_contrast_YYY...
    # And we need a "reference" image which is usually the one with min noise and contrast.
    
    # File 1: Reference (Low noise, Low contrast)
    # noise_001, contrast_001
    ref_filename = "image_noise_001_contrast_001.tif"
    # Use perfect_image from fixture
    io.imsave(input_intensity_dir / ref_filename, perfect_image.astype(np.uint8))
    
    ref_mask_filename = "image_noise_001_contrast_001.tif"
    io.imsave(input_mask_dir / ref_mask_filename, mask)
    
    # File 2: Test Image (Higher noise, Higher contrast)
    # noise_010, contrast_010
    test_filename = "image_noise_010_contrast_010.tif"
    test_image = np.random.randint(0, 255, mask.shape, dtype=np.uint8)
    io.imsave(input_intensity_dir / test_filename, test_image)
    
    test_mask_filename = "image_noise_010_contrast_010.tif"
    io.imsave(input_mask_dir / test_mask_filename, mask)
    
    # Run the metrics function
    metrics(
        input_intensity_source=str(input_intensity_dir),
        input_mask_source=str(input_mask_dir),
        output_filepath=str(output_csv),
        set_index=1
    )
    
    # Check if output CSV exists
    assert output_csv.exists()
    
    # Read CSV and check content
    df = pd.read_csv(output_csv)
    
    # We expect 2 rows (one for each image)
    assert len(df) == 2
    
    # Check columns
    expected_cols = ["IMAGE-NAME", "Noise_level", "Contrast_level", "SNR1"]
    for col in expected_cols:
        assert col in df.columns
        
    # Check values for the second image (index 1 in sorted order usually, but depends on sorting)
    # Our files: image_noise_001... and image_noise_010...
    # Sorted: 001 then 010.
    
    row0 = df.iloc[0]
    assert row0["Noise_level"] == 1
    assert row0["Contrast_level"] == 1
    
    row1 = df.iloc[1]
    assert row1["Noise_level"] == 10
    assert row1["Contrast_level"] == 10

def test_metrics_batch_processing_with_lists(tmp_path, synthetic_data):
    """
    Test the batch processing 'metrics' function using lists of files as input.
    """
    perfect_image, _, mask, _ = synthetic_data

    # Create directories
    input_intensity_dir = tmp_path / "intensity_list"
    input_mask_dir = tmp_path / "mask_list"
    output_dir = tmp_path / "output_list"
    input_intensity_dir.mkdir()
    input_mask_dir.mkdir()
    output_dir.mkdir()
    
    output_csv = output_dir / "results_list.csv"
    
    # Create dummy data
    # File 1: Reference (Low noise, Low contrast)
    ref_filename = "image_noise_001_contrast_001.tif"
    ref_path = input_intensity_dir / ref_filename
    io.imsave(ref_path, perfect_image.astype(np.uint8))
    
    ref_mask_filename = "image_noise_001_contrast_001.tif"
    ref_mask_path = input_mask_dir / ref_mask_filename
    io.imsave(ref_mask_path, mask)
    
    # File 2: Test Image (Higher noise, Higher contrast)
    test_filename = "image_noise_010_contrast_010.tif"
    test_path = input_intensity_dir / test_filename
    test_image = np.random.randint(0, 255, mask.shape, dtype=np.uint8)
    io.imsave(test_path, test_image)
    
    test_mask_filename = "image_noise_010_contrast_010.tif"
    test_mask_path = input_mask_dir / test_mask_filename
    io.imsave(test_mask_path, mask)
    
    # Create lists of file paths
    intensity_files = [str(ref_path), str(test_path)]
    mask_files = [str(ref_mask_path), str(test_mask_path)]
    
    # Run the metrics function with lists
    metrics(
        input_intensity_source=intensity_files,
        input_mask_source=mask_files,
        output_filepath=str(output_csv),
        set_index=1
    )
    
    # Check if output CSV exists
    assert output_csv.exists()
    
    # Read CSV and check content
    df = pd.read_csv(output_csv)
    
    # We expect 2 rows
    assert len(df) == 2
    
    # Check columns
    expected_cols = ["IMAGE-NAME", "Noise_level", "Contrast_level", "SNR1"]
    for col in expected_cols:
        assert col in df.columns
        
    # Check values
    row0 = df.iloc[0]
    assert row0["Noise_level"] == 1
    assert row0["Contrast_level"] == 1
    
    row1 = df.iloc[1]
    assert row1["Noise_level"] == 10
    assert row1["Contrast_level"] == 10

