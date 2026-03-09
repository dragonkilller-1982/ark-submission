import cv2    #noisy
import numpy as np

# Load the image
image = cv2.imread(r"C:\Users\soumy\Downloads\noisy.jpg")

if image is None:
    print("Error: Could not find image. Check the filename or path.")
else:
    # 1. Strong Bilateral Filter
    # This is better than Median Blur because it doesn't just blur everything;
    # it only smooths pixels with similar colors, keeping edges crisp.
    # d=9, sigmaColor=75, sigmaSpace=75
    denoised = cv2.bilateralFilter(image, 9, 75, 75)

    # 2. Multi-pass Non-Local Means (Light strength)
    # This cleans up the remaining color "jitter" without making it look like plastic.
    denoised_fine = cv2.fastNlMeansDenoisingColored(denoised, None, 10, 10, 7, 21)

    # 3. Detail Enhancement
    # This OpenCV function is specifically designed to make features visible
    # without creating the "glitchy" artifacts you saw earlier.
    # sigma_s controls smoothness, sigma_r controls edge preservation.
    enhanced = cv2.detailEnhance(denoised_fine, sigma_s=10, sigma_r=0.15)

    # 4. Final Contrast Boost
    # We use a slight Gamma Correction to pull the mountain features out of the shadows.
    gamma = 1.2
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    final_output = cv2.LUT(enhanced, table)

    # Save and show
    cv2.imwrite('enhanced_landscape.jpg', final_output)
    cv2.imshow('Final Cleaned Result', final_output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()