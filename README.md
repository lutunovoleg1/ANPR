
# Licence plate recognition
![1](https://shields.microej.com/badge/Python-3.11-blue)
![GitHub top language](https://img.shields.io/github/languages/top/artem-vol/Licence_plate_recognition)

Automatic recognition of car license plates on video using YOLO and EasyOCR models. YOLO will be used for the object detection task, EasyOCR - for the optical character recognition task, and an algorithm has also been implemented that determines the coordinates of the license plate corners using an algorithm based on the Hough algorithm.

## Installation
1. Make sure that you have Python version 3.11 installed.
2. Clone the repository:
```
git clone git@github.com:artem-vol/Fire-smoke_detection.git
```
3. Install the dependencies from the file requirements.txt:
```
pip install -r requirements.txt
```
## Usage
Run main.py to perform license plate recognition on video.
  
You can change the following parameters of ```auto_number_plate_rec``` function:
- **video_path** (str): The path to the input video file. 
- **save_video** (bool): If True, the processing result is saved to a video file. By default, False.
- **show_video** (bool): If True, the processing result is displayed in the window. By default, False.
- **show_fps** (bool): If True, the current FPS is displayed on the frame. By default, True.
- **output_video_path** (str): The path to save the processed video. The default is "output_video.mp4".

## Examples

![image](https://github.com/user-attachments/assets/cec7b933-d5f2-4e99-b790-657718e772bc)
![image](https://github.com/user-attachments/assets/891af56a-bc8a-41dd-82f2-e25d376189f8)


