import cv2

import numpy as np

from segmentation.threshold_segmenter import ThresholdSegmenter


# Load a test image.
image = cv2.imread("corn2.JPG")

# Extract masks for the corn ears.
segmenter = ThresholdSegmenter()
ear_masks = segmenter.segment_image(image)

# Find and print the portion of each ear that is consumed.
for mask in ear_masks:
    # Determine how many of the kernels have been removed.
    cob_pixels = np.count_nonzero(mask.cob_mask)
    union_pixels = np.count_nonzero(mask.ear_mask)
    kernel_fraction = cob_pixels / union_pixels

    print(f"Ear is {kernel_fraction * 100}% consumed.")

    cv2.imshow("Ear", mask.ear_mask.astype(np.uint8) * 255)
    cv2.waitKey(-1)

