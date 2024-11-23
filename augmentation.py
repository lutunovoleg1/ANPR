import albumentations as A
import cv2
import os


def augment_dataset(dir_path, dst_path, ratio, show_progress=True):
    # Augmentation pipeline
    transform = A.Compose([
        A.AdvancedBlur(blur_limit=(3, 9), sigma_x_limit=(0.5, 2), sigma_y_limit=(0.5, 2), p=0.5),
        A.Affine(rotate=(-16, 16), fit_output=True, p=1, mode=cv2.BORDER_REPLICATE, shear=(-20, 20)),
        A.LongestMaxSize(max_size=200, p=1),
        A.RandomFog(alpha_coef=1, fog_coef_lower=0.5, fog_coef_upper=0.7, p=0.25)
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


augment_dataset('ocr_datasets/image-label_dataset', 'ocr_datasets/augmented_dataset', 1)
