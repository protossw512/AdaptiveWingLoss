import cv2
import numpy as np
from typing import Tuple, Optional


def resize_image(image: np.ndarray, desired_shape: Tuple[int, int] = (35, 35),
                 dimension_to_match: Optional[str] = None,
                 crop_center: Optional[bool] = False,
                 add_padding: Optional[bool] = False,
                 interpolation: Optional[int] = None) -> np.ndarray:

    if dimension_to_match is not None:
        image = resize_preserving_aspect_ratio(image, desired_shape, dimension_to_match, interpolation)

    else:
        if interpolation is None:
            interpolation = get_interpolation_method(image.shape[:2], desired_shape[:2])
        image = cv2.resize(image, (desired_shape[1], desired_shape[0]), interpolation=interpolation)

    if add_padding:
        image = pad(image, desired_shape)

    return image


def resize_preserving_aspect_ratio(image: np.ndarray, desired_shape: Tuple[int, int] = (35, 35),
                                   dimension_to_match: str = 'bigger', interp: Optional[int] = None) -> np.ndarray:
    """
    Resize given image while preserving its aspect ratio. Resizes in a way that dimensions are either the same size or smaller.
    :param image: Image to resize.
    :param desired_shape: New image shape.
    :param dimension_to_match: The dimension to match with desired size. If 'bigger', then resizes in a way that output dimensions are
    either the same size or smaller. If 'smaller', then resizes in a way that output dimensions are either the same size or bigger.
    :param interp: Interpolation to use. If None, uses the best one.
    :return: Resized image.
    """
    image_height, image_width = image.shape[:2]
    desired_height, desired_width = desired_shape

    im_ratio = image_width / image_height
    s_ratio = desired_width / desired_height

    if (im_ratio > s_ratio and dimension_to_match == 'bigger') or (im_ratio < s_ratio and dimension_to_match == 'smaller'):  # horizontal image
        new_width = desired_width
        new_height = np.round(new_width / im_ratio).astype(int)

    elif (im_ratio < s_ratio and dimension_to_match == 'bigger') or (im_ratio > s_ratio and dimension_to_match == 'smaller'):  # vertical image
        new_height = desired_height
        new_width = np.round(new_height * im_ratio).astype(int)

    else:  # square image
        new_height, new_width = desired_height, desired_width

    if interp is None:
        interp = get_interpolation_method(image.shape[:2], desired_shape)

    return cv2.resize(image, (new_width, new_height), interpolation=interp)


def pad(image: np.ndarray, desired_shape: Tuple[int, int] = (35, 35), border_type: int = cv2.BORDER_CONSTANT, border_value: int = (0, 0, 0)) -> np.ndarray:
    """
    Pads the image.
    :param image: Image to pad.
    :param desired_shape: New shape.
    :param border_type: Type of border filling to use. See cv2.BORDER for options.
    :param border_value: If using BORDER_CONSTANT, set the constant value.
    :return: Padded image.
    """
    image_height, image_width = image.shape[:2]
    desired_height, desired_width = desired_shape

    im_ratio = image_width / image_height
    s_ratio = desired_width / desired_height

    pad_left, pad_right, pad_top, pad_bot = 0, 0, 0, 0

    if im_ratio > s_ratio:  # horizontal image
        new_w = desired_width
        new_h = np.round(new_w / im_ratio).astype(int)
        pad_v = (desired_height - new_h) / 2
        pad_top, pad_bot = np.floor(pad_v).astype(int), np.ceil(pad_v).astype(int)

    if im_ratio < s_ratio:  # vertical image
        new_h = desired_height
        new_w = np.round(new_h * im_ratio).astype(int)
        pad_h = (desired_width - new_w) / 2
        pad_left, pad_right = np.floor(pad_h).astype(int), np.ceil(pad_h).astype(int)

    padded_image = cv2.copyMakeBorder(image, pad_top, pad_bot, pad_left, pad_right, borderType=border_type, value=border_value)

    return padded_image


def resize_and_pad(image: np.ndarray, desired_shape: Tuple[int, int] = (35, 35), border_type: int = cv2.BORDER_CONSTANT, border_value: int = (0, 0, 0)) -> np.ndarray:
    """
    Image to resize and pad.
    :param image: Image to resize and pad.
    :param desired_shape: New shape of the image.
    :param border_type: Type of border filling to use. See cv2.BORDER for options.
    :param border_value: If using BORDER_CONSTANT, set the constant value.
    :return: Resized and padded image.
    """
    image = resize_preserving_aspect_ratio(image, desired_shape)
    image = pad(image, desired_shape, border_type=border_type, border_value=border_value)

    return image


def get_interpolation_method(image_shape: Tuple[int, int], desired_shape: Tuple[int, int]) -> int:
    """
    Finds the right interpolation method for resizing. INTER_AREA if downsampling; INTER_CUBIC if upsampling.
    :param image_shape:
    :param desired_shape:
    :return: Interpolation method.
    """
    image_height, image_width = image_shape[:2]
    desired_height, desired_width = desired_shape[:2]

    if image_height > desired_height or image_width > desired_width:
        # AREA is better for downsampling
        interp = cv2.INTER_AREA

    else:
        # CUBIC is better for upsampling
        interp = cv2.INTER_CUBIC

    return interp
