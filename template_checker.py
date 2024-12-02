import os

import cv2
import numpy as np

from main_utils import find_rectangle_corners, apply_perspective_transform


def get_templates(folder_path: str) -> dict:
    """Reads all templates from a given folder and returns a dictionary with the template names as keys
    and the template images as values.

    Args:
        folder_path: The path to the folder containing the templates.

    Returns:
        A dictionary with the template names as keys and the template images as values.
    """
    template_names = [
        'vehicle_1row_8characters2.png',
        'vehicle_1row_9characters2.png',
        'vehicle_2row_8characters2.png',
        'vehicle_2row_9characters2.png',
        'trailer_8characters2.png',
        'trailer_9characters2.png']
    _templates = {}
    for name in template_names:
        _templates[name[:-4]] = cv2.imread(os.path.join(folder_path, name))
    return _templates


def get_max_correlation(input_img: np.ndarray, template_img: np.ndarray) -> float:
    """Calculates the maximum correlation between an input image and a template image, both represented in grayscale.

    Args:
        input_img: The input image in which to search for the template.
        template_img: The template image to be correlated with the input image.

        Returns:
            float: The maximum correlation coefficient found between the input image and the template image.
        """
    # Change the resolution of the input image to the resolution of the template
    template_img = template_img.copy()
    input_img = input_img.copy()
    input_img = cv2.resize(input_img, (template_img.shape[1], template_img.shape[0]), interpolation=cv2.INTER_CUBIC)

    input_img = input_img.astype('float32')
    template_img = template_img.astype('float32')

    max_correlation = -1
    mean_template_image = np.mean(template_img)
    mean_input_image = np.mean(input_img)

    # Sliding a template over an input image
    for row_shift in range(-30, 31):
        for col_shift in range(-30, 31):
            template_shift = np.roll(np.roll(template_img, col_shift, axis=1), row_shift, axis=0)

            # Compute the correlation
            c_input_image = input_img - mean_input_image
            c_template_image = template_shift - mean_template_image
            num = (c_input_image * c_template_image).sum()
            denum = ((c_input_image ** 2).sum() * (c_template_image ** 2).sum()) ** 0.5
            correlation = num / denum

            if correlation > max_correlation:  # Update the maximum correlation
                max_correlation = correlation

    return max_correlation


# test code
input_image = cv2.imread('images_for_test/vehicle_2row_8ch_0.png')
templates = get_templates('template_images')

corners = find_rectangle_corners(input_image, show_results=False)
ret, rotated_image = apply_perspective_transform(input_image, corners, show_results=False)

for key in templates.keys():
    cor = get_max_correlation(cv2.cvtColor(rotated_image, cv2.COLOR_BGR2GRAY),
                              cv2.cvtColor(templates[key], cv2.COLOR_BGR2GRAY))
    print(key, cor)

cv2.imshow('input image', rotated_image)
cv2.waitKey(0)
