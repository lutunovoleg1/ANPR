import os
import csv


def ad_csv2image_label_structure(dir_path, eng2rus=True):
    eng_rus = {'A': 'А', 'B': 'В', 'C': 'С', 'E': 'Е', 'H': 'Н', 'K': 'К', 'M': 'М', 'O': 'О', 'P': 'Р', 'T': 'Т',
               'X': 'Х', 'Y': 'У'}
    for file in os.listdir(dir_path):
        csv_list = [['filename', 'words']]
        for image_name in os.listdir(os.path.join(dir_path, str(file))):
            if image_name == 'labels.csv':
                os.remove(os.path.join(dir_path, str(file), 'labels.csv'))
                continue
            image_label, _ = os.path.splitext(image_name)
            image_label = image_label.split('_')[0]  # Убираем убираем часть метки, которая показывает копии изображения
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


ad_csv2image_label_structure('ocr_datasets/augmented_dataset')
