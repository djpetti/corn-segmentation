from pathlib import Path
from typing import List

import cv2

import numpy as np

from .corn_segmenter import CornSegmenter
from .ear_mask import EarMask
from .sharpmask.sharpmask import SharpMask
from .threshold_segmenter import ThresholdSegmenter


class SharpmaskSegmenter(CornSegmenter):
    """
    A corn segmenter that uses Sharpmask in order to locate the ears in the
    image.

    Sharpmask is described here: https://arxiv.org/pdf/1603.08695.pdf
    Also, this code relies on the TensorFlow implementation of Sharpmask
    available here: https://github.com/aby2s/sharpmask
    """

    # Size to use for images that we process.
    _IMAGE_SIZE = (300, 300)

    # Lower HSV bound for thresholding the kernels.
    _KERNELS_LOWER_HSV = (18, 34, 65)
    # Upper HSV bound for thresholding the kernels.
    _KERNELS_UPPER_HSV = (30, 255, 255)
    # Lower HSV bound for thresholding the cob.
    _COB_LOWER_HSV = (0, 87, 100)
    # Upper HSV bound for thresholding the cob.
    _COB_UPPER_HSV = (16, 255, 255)

    def __init__(self, *, checkpoint_path: Path):
        """
        :param checkpoint_path: The path to the checkpoint folder containing
        the pre-trained Sharpmask model parameters.
        """
        # We predict on half of the input image at a time, so the size has to
        # be specified correctly.
        height, width = self._IMAGE_SIZE
        half_width = width // 2
        prediction_size = (height, half_width, 3)

        # Create the internal threshold segmenter to use.
        self.__threshold_segmenter = ThresholdSegmenter(
            kernels_lower_hsv=self._KERNELS_LOWER_HSV,
            kernels_upper_hsv=self._KERNELS_UPPER_HSV,
            cob_lower_hsv=self._COB_LOWER_HSV,
            cob_upper_hsv=self._COB_UPPER_HSV,
        )

        # Create the model.
        self.__model = SharpMask(train_path=None, validation_path=None,
                                 checkpoint_path=checkpoint_path,
                                 input_shape=prediction_size)
        self.__model.restore()

    def __find_ear_masks(self, image: np.ndarray) -> np.ndarray:
        """
        Finds the masks for both of the ears in a particular input image.
        :param image: The image to find the ear masks for.
        :return: An array, where the first dimension corresponds to each
        ear instance in the image, and the second two correspond to the mask.
        """
        # Sharpmask always looks for an object in the center, so if we split
        # the frame down the middle we tend to get a reasonable result for this
        # data.
        _, width, _ = image.shape
        middle = width // 2
        left_half = image[:, :middle]
        right_half = image[:, middle:]

        # Predict on both halves.
        left_mask = self.__model.predict_sharpmask(left_half)
        right_mask = self.__model.predict_sharpmask(right_half)

        # Extend both masks so they are the same size as the input.
        left_mask = np.pad(left_mask, ((0, 0), (0, width - middle)))
        right_mask = np.pad(right_mask, ((0, 0), (width - middle, 0)))

        return np.stack((left_mask, right_mask), axis=0)

    def segment_image(self, image: np.ndarray) -> List[EarMask]:
        """
        See superclass for documentation.
        """
        # Resize the image initially.
        image_small = cv2.resize(image, self._IMAGE_SIZE)

        ear_masks = self.__find_ear_masks(image_small)

        # Combine the ear masks and apply them to the image.
        combined_masks = np.zeros(self._IMAGE_SIZE, dtype=np.bool)
        for mask in ear_masks:
            combined_masks = np.logical_or(combined_masks, mask)
        color_masks = np.stack([combined_masks] * 3, axis=2)
        masked_image = image_small * color_masks

        cv2.imwrite("corn_masked.jpg", masked_image)

        # Use the more primitive ThresholdSegmenter on the masked image.
        return self.__threshold_segmenter.segment_image(masked_image)

