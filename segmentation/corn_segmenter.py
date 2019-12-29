from typing import List
import abc

import numpy as np

from .ear_mask import EarMask


class CornSegmenter(abc.ABC):
    """
    Encapsulates a high-level algorithm for segmenting ears of corn and
    determining where kernels have been removed.
    """

    @abc.abstractmethod
    def segment_image(self, image: np.ndarray) -> List[EarMask]:
        """
        Segments an input image containing ears of corn.
        :param image: The image to segment.
        :return: The segmentation results for each ear in the image.
        """
