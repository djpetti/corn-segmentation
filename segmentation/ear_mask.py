import numpy as np

from pydantic import validator
from pydantic.dataclasses import dataclass

from .type_helpers import ArbitraryTypesConfig


@dataclass(frozen=True, config=ArbitraryTypesConfig)
class EarMask:
    """
    Encapsulates masks for an ear of corn.
    :param kernel_mask: The mask for the kernels in the ear.
    :param cob_mask: The mask for the cob.
    :param ear_mask: The combined mask for the entire ear. Should be the union
    of the cob and kernel masks.
    """
    kernel_mask: np.ndarray
    cob_mask: np.ndarray
    ear_mask: np.ndarray

    @staticmethod
    def __validate_mask_image(mask: np.ndarray) -> np.ndarray:
        """
        Validates a binary mask image.
        :param mask: The mask to validate.
        :return: The validated mask.
        """
        if len(mask.shape) != 2:
            raise ValueError("Mask image must be 2D.")
        if mask.dtype != np.bool:
            raise ValueError("Mask must be boolean.")

        return mask

    @validator("kernel_mask")
    def kernel_mask_valid(cls, mask: np.ndarray) -> np.ndarray:
        """
        Ensures that the kernel mask is valid.
        :param mask: The mask to validate.
        :return: The validated mask.
        """
        return cls.__validate_mask_image(mask)

    @validator("cob_mask")
    def cob_mask_valid(cls, mask: np.ndarray) -> np.ndarray:
        """
        Ensures that the cob mask is valid.
        :param mask: The mask to validate.
        :return: The validated mask.
        """
        return cls.__validate_mask_image(mask)

    @validator("ear_mask")
    def ear_mask_valid(cls, mask: np.ndarray) -> np.ndarray:
        """
        Ensures that the ear mask is valid.
        :param mask: The mask to validate.
        :return: The validated mask.
        """
        return cls.__validate_mask_image(mask)
