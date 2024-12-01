import random
import os
import csv
import re

from PIL import Image
import numpy as np
import albumentations as A
import cv2


def ad_csv_image_label_structure(dir_path: str, eng2rus=True):
    """Creates a CSV file with image names and corresponding labels from the specified directory.

    Args:
        dir_path: Path to the directory containing the image subdirectories.
        eng2rus: If True, converts English letters to Russian according to the specified dictionary.

    Returns:
        None: The function does not return values, but creates a 'labels.csv' file in each subdirectory.

     Notes:
         - The names of the images must be in the format '<label>_<additional information>.<extension>'.
         - Letters and numbers are processed separately: the numbers remain unchanged,and the letters are
         converted when the eng2rus parameter is enabled.
    """
    eng_rus = {'A': 'А', 'B': 'В', 'C': 'С', 'E': 'Е', 'H': 'Н', 'K': 'К', 'M': 'М', 'O': 'О', 'P': 'Р', 'T': 'Т',
               'X': 'Х', 'Y': 'У'}
    for file in os.listdir(dir_path):
        csv_list = [['filename', 'words']]
        for image_name in os.listdir(os.path.join(dir_path, str(file))):
            if image_name == 'labels.csv':
                os.remove(os.path.join(dir_path, str(file), 'labels.csv'))
                continue
            image_label, _ = os.path.splitext(image_name)
            image_label = image_label.split('_')[0]
            if eng2rus:
                new_label = ''
                for c in image_label:
                    if c not in '1234567890':
                        new_label += eng_rus[c]
                    else:
                        new_label += c
            else:
                new_label = image_label
            csv_list.append([image_name, new_label])
        csv_file_path = os.path.join(dir_path, file, 'labels.csv')
        # Write data to csv
        with open(csv_file_path, mode='w', newline='', encoding='utf-8') as file_w:
            writer = csv.writer(file_w, delimiter=',')
            writer.writerows(csv_list)


def data_structure2csv(dir_path: str, dst_path: str, eng2rus=False):
    """Converts the data structure from the image directory to a CSV file and saves the images to a new directory.

    This function creates subdirectories in the specified destination directory, copies images from the source
    directory, and creates a CSV file with image names and corresponding labels. If necessary, the labels can be
    converted from the English alphabet to Russian.

    Args:
        dir_path: Path to the source directory containing subdirectories with images and labels.
        dst_path: The path to the destination directory where the images and CSV files will be saved.
        eng2rus: If True, converts English letters to Russian according to the specified dictionary.

    Returns:
        None: The function does not return values, but creates subdirectories and a 'labels.csv' file
        in the destination directory.
    """
    eng_rus = {'A': 'А', 'B': 'В', 'C': 'С', 'E': 'Е', 'H': 'Н', 'K': 'К', 'M': 'М', 'O': 'О', 'P': 'Р', 'T': 'Т',
               'X': 'Х', 'Y': 'У'}
    if not os.path.isdir(dst_path):  # Create dataset folder
        os.mkdir(dst_path)
    for file in os.listdir(dir_path):
        if not os.path.isdir(os.path.join(dst_path, file)):  # Create dataset subfolder
            os.mkdir(os.path.join(dst_path, file))
        csv_list = [['filename', 'words']]
        for image_label in os.listdir(os.path.join(dir_path, str(file))):
            for image_name in os.listdir(os.path.join(dir_path, str(file), image_label)):
                new_label = ''
                if eng2rus:
                    for c in image_label:
                        if c not in '1234567890':
                            new_label += eng_rus[c]
                        else:
                            new_label += c
                else:
                    new_label = image_label
                csv_list.append([image_name, new_label])
                # save image in new folder
                image = Image.open(os.path.join(dir_path, file, image_label, image_name))
                image.save(os.path.join(dst_path, file, image_name))
        csv_file_path = os.path.join(dst_path, file, 'labels.csv')
        # Write data to csv
        with open(csv_file_path, mode='w', newline='', encoding='utf-8') as file_w:
            writer = csv.writer(file_w, delimiter=',')
            writer.writerows(csv_list)


def augment_dataset(dir_path: str, dst_path: str, ratio: int, show_progress=True):
    """
    Increases the data set using various image augmentations and stores them in the specified directory.

    This function goes through all the subdirectories in the specified directory with images, applies
    the specified augmentations to each image and saves the results in a new directory.

    Args:
        dir_path: The path to the source directory containing the image subdirectories.
        dst_path: The path to the destination directory where the augmented images will be saved.
        ratio: The number of times the augmentation will be applied to each image.
        show_progress: If True, displays the progress of image processing. By default, True.

    Returns:
        None: The function does not return values, but creates subdirectories and stores the augmented images
        in the destination directory.
    """
    # Augmentation pipeline
    transform = A.Compose([
        A.GridDistortion(num_steps=random.randint(2, 10), distort_limit=(-0.3, 0.3), p=0.5),
        A.Downscale(interpolation_pair={'downscale': cv2.INTER_NEAREST, 'upscale': cv2.INTER_LINEAR},
                    scale_range=(0.4, 0.7), p=0.3),
        A.Compose([
            A.AdvancedBlur(blur_limit=(3, 9), sigma_x_limit=(0.5, 2), sigma_y_limit=(0.5, 2), p=1),
            A.LongestMaxSize(max_size=200, p=1)
        ], p=0.5),
        A.MotionBlur(blur_limit=(5, 13), allow_shifted=True, p=0.6),
        A.AdvancedBlur(blur_limit=(3, 9), sigma_x_limit=(0.5, 2), sigma_y_limit=(0.5, 2), p=0.5),
        A.Affine(rotate=(-5, 5), fit_output=True, mode=cv2.BORDER_REPLICATE, shear=(-5, 5), p=0.8),
        A.Perspective(keep_size=False, fit_output=True, pad_mode=cv2.BORDER_REPLICATE, p=0.5),
        A.RandomFog(alpha_coef=0.6, fog_coef_lower=0.5, fog_coef_upper=0.7, p=0.3)
    ])

    if not os.path.isdir(dst_path):  # Create dataset folder
        os.mkdir(dst_path)
    for file in os.listdir(dir_path):  # Going through folders
        if not os.path.isdir(os.path.join(dst_path, file)):  # Create dataset subfolder ['train, 'valid']
            os.mkdir(os.path.join(dst_path, file))
        amount_images = len(os.listdir(os.path.join(dir_path, str(file))))
        for amount, image_name in enumerate(os.listdir(os.path.join(dir_path, str(file))), start=1):
            if show_progress and amount % 100 == 0:  # Show progress
                print(f'{file}: {amount / amount_images}%')
            root, ext = os.path.splitext(image_name)
            if ext in '.jpg.png':  # Checking whether we are working with the image
                image = cv2.imread(os.path.join(dir_path, str(file), image_name))
                for i in range(ratio):  # We apply the augmentation to the image ratio times
                    new_root = root + f'_{i}'
                    transformed = transform(image=image)
                    cv2.imwrite(os.path.join(dst_path, file, new_root + ext), transformed['image'])
                del image


def equalize_dataset(train_path: str, align_ratio=False) -> dict[str: dict[str: int, str: dict[int: set]]]:
    """This function counts the number of occurrences of each character in the dataset and aligns the dataset so that
    the number of occurrences of each character is approximately equal for all characters.

    Args:
        train_path: path to the dataset.
        align_ratio: If the value is True, the dataset will be aligned

    Returns:
        characters_entry: A dictionary with characters as keys and their occurrences in the dataset as values

    Notes:
        The characters_entry dictionary has the following structure:
        {'character': {'count': int, 'labels': {int: set}}}
        where 'count' is the total number of occurrences of the character in the dataset
        and 'labels' is a dictionary with the number of occurrences of the character in the label
        as keys and sets of labels as values.
    """
    characters_entry = {}
    for image_name in os.listdir(train_path):
        root, ext = os.path.splitext(image_name)
        if ext == '.csv':
            continue
        root = root.split('_')[0]
        for c in root:
            characters_entry.setdefault(c, {'count': 0, 'labels': {}})
            characters_entry[c]['count'] += 1  # Count the total number of occurrences of the character in the dataset
            c_in_label = root.count(c)  # Count the number of occurrences of the character in the label
            characters_entry[c]['labels'][c_in_label] = characters_entry[c]['labels'].setdefault(c_in_label, set())
            characters_entry[c]['labels'][c_in_label].add(image_name)  # Add new label

    if align_ratio:
        flag = True
        part = 1 / len(characters_entry.keys())  # Еhe ratio of data with an even distribution of classes

        while flag:
            # Check the ratio of classes
            total_count = sum([c['count'] for c in characters_entry.values()])  # Total number of characters
            # The ratio of the number of class symbols to the total number
            checker = [characters_entry[key]['count'] / total_count for key in characters_entry.keys()]
            checker = [part * 0.5 <= val <= part * 1.5 for val in checker]
            if all(checker):
                flag = False

            # Adding the optimal label to the dataset
            min_to_max = sorted(characters_entry.keys(), key=lambda x: characters_entry[x]['count'])
            for i, key in enumerate(min_to_max):

                if i == 0:  # Search for a label in the class with the least number of occurrences
                    # Coefficients influencing the choice of the best label
                    mult = np.array([characters_entry[key]['count'] / total_count for key in min_to_max[1:]])
                    min_sum_label = 10000

                    for amount in range(6, 0, -1):
                        labels = characters_entry[key]['labels'].get(amount, None)

                        if labels is None:
                            continue

                        for label in labels:
                            num_cls_chr = []  # The number of class characters in the label
                            for j in range(1, len(min_to_max)):
                                key_2 = min_to_max[j][0]
                                num_cls_chr.append(label.split('_')[0].count(key_2))
                            num_cls_chr = (np.array(num_cls_chr) + 1) * mult

                            if sum(num_cls_chr) < min_sum_label:  # Assigning the best label
                                best_label = label
                                min_sum_label = sum(num_cls_chr)

                    # Updating the number of characters in the class
                    amount = re.split(r'[_.]', best_label)[0].count(key)
                    characters_entry[key]['count'] += amount

                    # Assigning a new name to the label
                    img_id = -1  # Count the number of labels in the dataset
                    for next_label in characters_entry[key]['labels'][amount]:
                        if re.split(r'[_.]', next_label)[0] == re.split(r'[_.]', best_label)[0]:
                            img_id += 1
                    new_label = re.split(r'[_.]', best_label)[0] + f'_{img_id}.png'

                    # Saving the image under new name
                    characters_entry[key]['labels'][amount].add(new_label)  # Add a new name to the dictionary
                    image = Image.open(os.path.join(train_path, best_label))
                    image.save(os.path.join(train_path, new_label))
                    continue

                else:  # Update new occurrences to other classes
                    amount = re.split(r'[_.]', new_label)[0].count(key)
                    if amount != 0:
                        characters_entry[key]['count'] += amount
                        characters_entry[key]['labels'][amount].add(new_label)

    return characters_entry
