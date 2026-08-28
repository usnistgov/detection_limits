# Detection Limits Library Metrics

This document describes the image quality metrics calculated by the `detection_limits` library.

## Variables Key

| Symbol | Description | Code Variable |
| :--- | :--- | :--- |
| $\mu_{fg}$ | Mean intensity of the foreground region | `foreground_mean` |
| $\sigma^2_{fg}$ | Variance of the foreground region | `foreground_var` |
| $\mu_{bg}$ | Mean intensity of the background region | `background_mean` |
| $\sigma^2_{bg}$ | Variance of the background region | `background_var` |
| $\sigma_{bg}$ | Standard deviation of the background region | `background_std` |
| $\sigma_{noise}$ | Simulated (known) noise parameter | `noise_val` |
| $I_{max}$ | Maximum pixel intensity in the image | `np.max(image)` |
| $I_{min}$ | Minimum pixel intensity in the image | `np.min(image)` |
| $\sigma_{img}$ | Standard deviation of the entire image | `std_intensity` |

## SNR Metrics

These metrics evaluate the Signal-to-Noise Ratio (SNR) using various definitions, comparing foreground/background statistics or using the known simulation noise parameter.

| Metric ID | Name / Description | Equation |
| :--- | :--- | :--- |
| **SNR1** | SNR Power Estimate | $$ \frac{\sigma^2_{fg}}{\sigma^2_{bg}} $$ |
| **SNR2** | SNR RMS Power Estimate | $$ \frac{\sqrt{\mu_{fg}}}{\sqrt{\sigma^2_{fg}}} $$ |
| **SNR3** | SNR Inverse CV² Estimate | $$ \frac{\mu_{fg}^2}{\sigma^2_{bg}} $$ |
| **SNR4** | SNR Inverse CV Estimate | $$ \frac{\mu_{fg}}{\sigma_{bg}} $$ |
| **SNR5** | SNR Inverse CV Parameter | $$ \frac{\mu_{fg}}{\sigma_{noise}} $$ |
| **SNR6** | SNR Inverse CV² Parameter | $$ \frac{\mu_{fg}^2}{\sigma_{noise}^2} $$ |
| **SNR7** | SNR Power Parameter | $$ \frac{\sigma^2_{fg}}{\sigma_{noise}^2} $$ |
| **SNR8** | SNR RMS Power Parameter | $$ \frac{\sqrt{\mu_{fg}}}{\sigma_{noise}} $$ |
| **SNR9** | Cohen's d Estimate | $$ \frac{\mu_{fg} - \mu_{bg}}{\sigma_{bg}} $$ |
| **SNR10** | Cohen's d Parameter | $$ \frac{\mu_{fg} - \mu_{bg}}{\sigma_{noise}} $$ |

## General Image Quality Metrics

| Metric Name | Description | Equation / Method |
| :--- | :--- | :--- |
| **Michelson Contrast** | Contrast based on luminance range | $$ \frac{I_{max} - I_{min}}{I_{max} + I_{min}} $$ |
| **RMS Contrast** | Root Mean Square Contrast | $$ \sigma_{img} $$ |
| **Edge Density** | Density of edges detected by Canny detector | $$ \frac{\sum \text{Canny}(I)}{\text{Total Pixels}} $$ |
| **SSIM** | Structural Similarity Index | `skimage.metrics.structural_similarity` |
| **PSNR** | Peak Signal-to-Noise Ratio | `cv2.PSNR` |
| **MI** | Mutual Information (with mask) | `sklearn.metrics.mutual_info_score` |
| **NMI** | Normalized Mutual Information (with mask) | `sklearn.metrics.normalized_mutual_info_score` |
| **CE** | Cross Entropy (Image Entropy) | `scipy.stats.entropy` (on histogram) |
