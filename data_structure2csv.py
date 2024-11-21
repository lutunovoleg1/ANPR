import os
import csv
from PIL import Image


def data_structure2csv(dir_path, dst_path, eng2rus=False):
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


data_structure2csv('ocr_datasets/data_structure_dataset',
                   '/ocr_datasets/csv_dataset')
