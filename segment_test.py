import cv2

import numpy as np


# Load a test image.
image = cv2.imread("corn.jpg")

# Make the image small so processing is faster.
small_image = cv2.resize(image, (300, 300))
# Perform mean shift filtering.
shifted = cv2.pyrMeanShiftFiltering(small_image, 15, 35)
cv2.imshow("Input", shifted)
cv2.waitKey(-1)

# Convert to HSV so color thresholding is easier.
hsv_image = cv2.cvtColor(shifted, cv2.COLOR_BGR2HSV)
# Threshold the kernels and the cob.
kernel_mask = cv2.inRange(hsv_image, (18, 128, 128), (35, 255, 255))
cob_mask = cv2.inRange(hsv_image, (0, 115, 108), (16, 222, 233))
#cv2.imshow("Kernels", kernel_mask)
#cv2.waitKey(-1)
#cv2.imshow("Cob", cob_mask)
#cv2.waitKey(-1)

# Erode and dilate the masks to remove noise.
kernel = np.ones((3, 3), np.uint8)
kernels_closed = cv2.morphologyEx(kernel_mask, cv2.MORPH_CLOSE, kernel)
kernels_opened = cv2.morphologyEx(kernels_closed, cv2.MORPH_OPEN, kernel,
                                  iterations=2)
cob_closed = cv2.morphologyEx(cob_mask, cv2.MORPH_CLOSE, kernel)
cob_opened = cv2.morphologyEx(cob_closed, cv2.MORPH_OPEN, kernel,
                              iterations=2)
cv2.imshow("Denoised", kernels_opened)
cv2.waitKey(-1)
cv2.imshow("Denoised", cob_opened)
cv2.waitKey(-1)

# Find the union of the two.
ear_mask = np.clip(0, 255, cob_opened + kernels_opened)
# Perform morphological closing to remove any small gaps.
ear_mask = cv2.morphologyEx(ear_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
cv2.imshow("Union", ear_mask)
cv2.waitKey(-1)

# Find the contours in the image.
contours, _ = cv2.findContours(ear_mask, cv2.RETR_EXTERNAL,
                               cv2.CHAIN_APPROX_SIMPLE)
ear_mask_with_contours = np.stack((ear_mask, ear_mask, ear_mask), axis=2)
cv2.drawContours(ear_mask_with_contours, contours, -1, (0, 0, 255), 1)

# Fit ellipses to the contours.
ellipses = []
for contour in contours:
    if len(contour) < 5:
        # Contour is too small to fit an ellipse to.
        continue
    center, size, angle = cv2.fitEllipse(contour)

    # Filter out any ellipses that are not oblong enough.
    width, height = size
    length_ratio = width / height
    if length_ratio < 1.0:
        # The ellipse is vertical, so flip this.
        length_ratio = 1.0 / length_ratio
    if length_ratio < 3.0:
        # Unlikely to be corn, because it's not oblong enough. Discard this.
        continue

    ellipses.append((center, size, angle))

# Draw the bounding ellipses.
ear_mask_with_ellipses = np.copy(ear_mask_with_contours)
for ellipse in ellipses:
    center, size, angle = ellipse
    center = tuple([int(d) for d in center])
    size = tuple([int(d) for d in size])
    angle = int(angle)
    cv2.ellipse(ear_mask_with_ellipses, center, size, angle, 0, 360,
                (0, 255, 0))
cv2.imshow("Ellipses", ear_mask_with_ellipses)
cv2.waitKey(-1)

# Mask out one ear at a time.
ear_masks = []
for ellipse in ellipses:
    center, size, angle = ellipse
    center = tuple([int(d) for d in center])
    size = tuple([int(d) for d in size])
    angle = int(angle)

    background_mask = np.zeros_like(ear_mask)
    cv2.ellipse(background_mask, center, size, angle, 0, 360,
                (255, 255, 255), thickness=-1)
    ear_masks.append(background_mask)

# Process each ear individually.
for mask in ear_masks:
    ear_kernels = np.logical_and(kernels_opened, mask
                                 ).astype(np.uint8) * 255
    ear_cob = np.logical_and(cob_opened, mask
                             ).astype(np.uint8) * 255
    ear_union = np.clip(0, 255, ear_kernels + ear_cob)

    cv2.imshow("Ear", ear_union)
    cv2.waitKey(-1)

    # Determine how many of the kernels have been removed.
    cob_pixels = np.count_nonzero(ear_cob)
    union_pixels = np.count_nonzero(ear_union)
    kernel_fraction = cob_pixels / union_pixels

    print(f"Ear is {kernel_fraction * 100}% consumed.")

