from pathlib import Path
import argparse

import cv2

import numpy as np

from segmentation.sharpmask_segmenter import SharpmaskSegmenter
from segmentation.threshold_segmenter import ThresholdSegmenter


def _make_threshold_segmenter(args: argparse.Namespace) -> ThresholdSegmenter:
    """
    Creates a ThresholdSegmenter instance based on user-supplied configuration.
    :param args: The CLI arguments.
    :return: The ThresholdSegmenter it created.
    """
    return ThresholdSegmenter()


def _make_sharpmask_segmenter(args: argparse.Namespace) -> SharpmaskSegmenter:
    """
    Creates a SharpmaskSegmenter instance based on user-supplied configuration.
    :param args: The CLI arguments.
    :return: The SharpmaskSegmenter it created.
    """
    return SharpmaskSegmenter(checkpoint_path=Path(args.model))


def _make_parser() -> argparse.ArgumentParser:
    """
    Creates a parser to use for reading CLI arguments.
    :return: The parser that it created.
    """
    parser = argparse.ArgumentParser(description="Segments images of corn and"
                                                 " determines the fraction of"
                                                 " each ear consumed.")
    parser.add_argument("corn_image", help="The path to the image of corn to "
                                           "read and process.")
    subparsers = parser.add_subparsers()

    parser_classical = subparsers.add_parser("classical")
    parser_classical.set_defaults(make_segmenter=_make_threshold_segmenter)

    parser_sharpmask = subparsers.add_parser("sharpmask")
    parser_sharpmask.add_argument("model", help="The path to the saved model "
                                                "checkpoint.")
    parser_sharpmask.set_defaults(make_segmenter=_make_sharpmask_segmenter)

    return parser


def main() -> None:
    parser = _make_parser()
    args = parser.parse_args()

    # Load a test image.
    image = cv2.imread(args.corn_image)

    # Extract masks for the corn ears.
    segmenter = args.make_segmenter(args)
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


if __name__ == "__main__":
    main()

