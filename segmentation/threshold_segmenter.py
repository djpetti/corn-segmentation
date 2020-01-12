from typing import List, Tuple

import cv2

import numpy as np

from .corn_segmenter import CornSegmenter
from .ear_mask import EarMask


class ThresholdSegmenter(CornSegmenter):
    """
    A CornSegmenter that relies entirely on thresholding to determine the parts
    of the ear.
    """

    # Size to use for images that we process.
    _IMAGE_SIZE = (300, 300)
    # Spatial window radius for mean shift filtering.
    _SPATIAL_WINDOW = 15
    # Color window radius for mean shift filtering.
    _COLOR_WINDOW = 35
    # Kernel to use for morphological filtering.
    _MORPH_KERNEL = np.ones((3, 3), dtype=np.uint8)

    # Lower HSV bound for thresholding the kernels.
    _KERNELS_LOWER_HSV = (18, 100, 65)
    # Upper HSV bound for thresholding the kernels.
    _KERNELS_UPPER_HSV = (30, 255, 255)
    # Lower HSV bound for thresholding the cob.
    _COB_LOWER_HSV = (0, 115, 108)
    # Upper HSV bound for thresholding the cob.
    _COB_UPPER_HSV = (16, 222, 233)

    # Minimum area in pixels we require for an ear of corn.
    _EAR_MINIMUM_AREA = 1000
    # Minimum ratio of major axis to minor axis for the best-fit ellipse around
    # an ear of corn.
    _EAR_MIN_AXIS_RATIO = 3.0

    def __init__(self,
                 kernels_lower_hsv: Tuple[int, int, int] = _KERNELS_LOWER_HSV,
                 kernels_upper_hsv: Tuple[int, int, int] = _KERNELS_UPPER_HSV,
                 cob_lower_hsv: Tuple[int, int, int] = _COB_LOWER_HSV,
                 cob_upper_hsv: Tuple[int, int, int] = _COB_UPPER_HSV):
        """
        :param kernels_lower_hsv: The lower HSV bound to use for thresholding
        kernels.
        :param kernels_upper_hsv: The upper HSV bound to use for thresholding
        kernels.
        :param cob_lower_hsv: The lower HSV bound to use for thresholding cobs.
        :param cob_upper_hsv: The upper HSV bound to use for thresholding cobs.
        """
        self.__kernels_lower_hsv = kernels_lower_hsv
        self.__kernels_upper_hsv = kernels_upper_hsv
        self.__cob_lower_hsv = cob_lower_hsv
        self.__cob_upper_hsv = cob_upper_hsv

    @classmethod
    def __prepare_image(cls, image: np.ndarray) -> np.ndarray:
        """
        Prepares the image for segmentation.
        :param image: The image to segment.
        :return: The image, resized, posterized, and converted to the HSV
        colorspace.
        """
        # Make the image small so processing is faster.
        small_image = cv2.resize(image, cls._IMAGE_SIZE)
        # Perform mean shift filtering.
        shifted = cv2.pyrMeanShiftFiltering(small_image, cls._SPATIAL_WINDOW,
                                            cls._COLOR_WINDOW)
        # Convert to HSV so color thresholding is easier.
        hsv_image = cv2.cvtColor(shifted, cv2.COLOR_BGR2HSV)

        return hsv_image

    @classmethod
    def __clean_mask(cls, mask: np.ndarray) -> np.ndarray:
        """
        Uses morphological filtering to clean up a mask.
        :param mask: The mask to clean up.
        :return: The cleaned mask.
        """
        mask_closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cls._MORPH_KERNEL)
        mask_opened = cv2.morphologyEx(mask_closed, cv2.MORPH_OPEN,
                                       cls._MORPH_KERNEL, iterations=2)

        return mask_opened

    def __mask_ear_parts(self, image: np.ndarray) -> EarMask:
        """
        Masks the parts of an ear of corn through thresholding. The masks
        produced by this approach are fairly naive.
        :param image: The raw image containing ears of corn.
        :return: The mask of all the kernels in the image, and the mask of all
        the cobs in the image.
        """
        # Threshold the kernels and the cob.
        kernel_mask = cv2.inRange(image, self.__kernels_lower_hsv,
                                  self.__kernels_upper_hsv)
        cob_mask = cv2.inRange(image, self.__cob_lower_hsv,
                               self.__cob_upper_hsv)

        # Clean up the masks.
        clean_kernels = self.__clean_mask(kernel_mask)
        clean_cob = self.__clean_mask(cob_mask)

        # Find the union of the two.
        ear_mask = np.clip(0, 255, clean_kernels + clean_cob)
        # Perform morphological closing to remove any small gaps.
        clean_ear = cv2.morphologyEx(ear_mask, cv2.MORPH_CLOSE,
                                     self._MORPH_KERNEL, iterations=2)

        return EarMask(kernel_mask=clean_kernels.astype(np.bool),
                       cob_mask=clean_cob.astype(np.bool),
                       ear_mask=clean_ear.astype(np.bool))

    @classmethod
    def __extract_ears(cls, ears_mask: np.ndarray,
                       ) -> Tuple[np.ndarray, List[int]]:
        """
        Takes a combined rough mask for all the ears and extracts clean masks
        for each individual ears.
        :param ears_mask: The combined mask image for all the ears.
        :return: The array of all the contours in the combined ear mask image,
        as well as the indices of the contours that correspond to actual ears of
        corn.
        """
        # Find the contours in the image.
        contours, _ = cv2.findContours(ears_mask.astype(np.uint8),
                                       cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)

        # Fit ellipses to the contours.
        idx_to_contour = {i: c for i, c in enumerate(contours)}
        # Filter out contours with too little area.
        idx_to_contour = {i: c for i, c in idx_to_contour.items()
                          if cv2.contourArea(c) >= cls._EAR_MINIMUM_AREA}
        # Filter out contours that contain too few points to fit an ellipse to.
        idx_to_contour = {i: c for i, c in idx_to_contour.items()
                          if len(c) >= 5}

        # Filter out contours that don't have the right geometry.
        ear_indices = []
        for i, contour in idx_to_contour.items():
            center, size, angle = cv2.fitEllipse(contour)

            # Filter out any ellipses that are not oblong enough.
            width, height = size
            length_ratio = width / height
            if length_ratio < 1.0:
                # The ellipse is vertical, so flip this.
                length_ratio = 1.0 / length_ratio
            if length_ratio < cls._EAR_MIN_AXIS_RATIO:
                # Unlikely to be corn, because it's not oblong enough.
                continue

            # Filter out any that are not vertical enough.
            _, _, width, height = cv2.boundingRect(contour)
            if height < width:
                # Unlikely to be corn, because it's horizontal.
                continue

            # This contour is good.
            ear_indices.append(i)

        return contours, ear_indices

    @classmethod
    def __create_ear_masks(cls, combined_masks: EarMask) -> List[EarMask]:
        """
        Separates a combined rough ear mask into individual clean masks for
        each ear.
        :param combined_masks: The combined rough mask of all the ears.
        :return: A list of masks for each ear.
        """
        # Find the ear contours.
        contours, ear_indices = cls.__extract_ears(combined_masks.ear_mask)

        # Mask out one ear at a time.
        ear_masks = []
        for i in ear_indices:
            background_mask = np.zeros_like(combined_masks.ear_mask,
                                            dtype=np.uint8)
            cv2.drawContours(background_mask, contours, i, (255, 255, 255), -1)
            ear_masks.append(background_mask)

        # Separate the masks for each ear into masks for the kernel and cob.
        kernel_masks = []
        cob_masks = []
        for ear_mask in ear_masks:
            ear_kernels = np.logical_and(combined_masks.kernel_mask, ear_mask)
            ear_cob = np.logical_and(combined_masks.cob_mask, ear_mask)
            kernel_masks.append(ear_kernels)
            cob_masks.append(ear_cob)

        all_masks = []
        for kernel_mask, cob_mask, ear_mask in zip(kernel_masks, cob_masks,
                                                   ear_masks):
            all_masks.append(EarMask(kernel_mask=kernel_mask,
                                     cob_mask=cob_mask,
                                     ear_mask=ear_mask.astype(np.bool)))

        return all_masks

    def segment_image(self, image: np.ndarray) -> List[EarMask]:
        """
        See superclass for documentation.
        """
        # Prepare the image.
        hsv_image = self.__prepare_image(image)
        # Separate the kernels and cob.
        combined_masks = self.__mask_ear_parts(hsv_image)
        # Clean and extract masks for each individual ear.
        return self.__create_ear_masks(combined_masks)