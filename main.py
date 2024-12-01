import cv2
from ultralytics import YOLO
import numpy as np
import easyocr
import time
from main_utils import scale_image, find_rectangle_corners, apply_perspective_transform, prepare_result


def auto_number_plate_rec(video_path, save_video=False, show_video=False, show_fps=True,
                          output_video_path="output_video.mp4"):
    # Initialize the models, if the device has cuda, it is installed automatically
    reader = easyocr.Reader(['ru'],
                            model_storage_directory='custom_EasyOCR/model',
                            user_network_directory='custom_EasyOCR/user_network',
                            recog_network='ru_numbers')
    model = YOLO('segmentation_models/yolov8_n/weights/best.pt')
    model.fuse()

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():  # Handling exceptions if video is not opened
        raise Exception('Error: Could not open the video')

    # Get input video size
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if save_video:  # Create a buffer of the moving average
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_save = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_w, frame_h))

    if show_fps:  # Creating a moving average calculation window
        fps_buffer = [0 for _ in range(50)]

    while cap.isOpened():
        if show_fps:
            time1 = time.time()

        ret, frame = cap.read()
        if not ret:  # If frame is not read, end the loop
            break

        # Create field for car plates and their text numbers
        field_w = frame.shape[1] // 6  # Width of right field
        out = np.hstack((frame, np.zeros((frame.shape[0], field_w, 3), dtype=np.uint8)))

        # Get YOLO results
        results = model(frame, imgsz=640, iou=0.0, conf=0.6, verbose=False)
        boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
        conf = results[0].boxes.conf.cpu().numpy().astype(float)

        for i in range(len(boxes)):
            # Show a bounding boxes of cars licence plates
            color = (0, 255, 0)
            cv2.rectangle(out, (boxes[i][:2]), (boxes[i][2:]), color=color, thickness=2)
            cv2.putText(out, f'numberplate:{round(conf[i], 2)}', (boxes[i][0], boxes[i][1]),
                        cv2.FONT_HERSHEY_DUPLEX, 0.7, color, 2)

            # Determine the car licence bounding box
            x1, y1, x2, y2 = boxes[i][0], boxes[i][1], boxes[i][2], boxes[i][3]
            box_number = frame[y1:y2, x1:x2]
            # Determine the car licence bounding box with additional indentation
            dx = round((boxes[i][2] - boxes[i][0]) * 0.08)  # x-axis indentation
            dy = round((boxes[i][3] - boxes[i][1]) * 0.25)  # y-axis indentation
            x1, y1, x2, y2 = x1 - dx, y1 - dy, x2 + dx, y2 + dy  # new coordinates
            # determine whether the new coordinates are included in the image
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(frame_w, x2), min(frame_h, y2)
            dilated_box_number = frame[y1:y2, x1:x2]

            # licence plate image processing
            points = find_rectangle_corners(dilated_box_number)
            points = [points[0], points[1], points[3], points[2]]
            perspective_ret, dilated_box_number = apply_perspective_transform(dilated_box_number, points)
            if perspective_ret:
                box_number = dilated_box_number

            # Get OCR model results
            result = reader.readtext(scale_image(box_number, 200, cv2.INTER_CUBIC),
                                     allowlist='1234567890АВСЕНКМОРТХУ',
                                     min_size=15,
                                     low_text=0.15,
                                     batch_size=2,
                                     ycenter_ths=0.8,
                                     height_ths=0.5,
                                     slope_ths=0.5,
                                     contrast_ths=0.2)
            text, template_valid, confs = prepare_result(result)

            # Place results on the right field
            if text is not None:
                # Place licence plate
                box_number = cv2.resize(box_number, (field_w, frame.shape[0] // 20), cv2.INTER_NEAREST)
                out[box_number.shape[0] * i * 2: box_number.shape[0] * (i * 2 + 1), frame.shape[1]:] = box_number
                # place text of license plate
                if not template_valid:  # If the text matches the number template, its color will be green
                    color = (255, 255, 255)
                else:
                    color = (0, 255, 0)
                cv2.putText(out, str(text), (frame.shape[1], box_number.shape[0] * (i * 2 + 2) - 5),
                            cv2.FONT_HERSHEY_COMPLEX, 0.8, color, 2)

        if cv2.waitKey(1) & 0xFF == ord('q'):  # Exit when the keyboard is pressed
            break

        if show_fps:
            del fps_buffer[-1]  # Delete the last buffer element
            fps_buffer.insert(0, float(time.time()) - float(time1))  # Add a new item to the buffer
            cv2.putText(out, f'FPS: {round(len(fps_buffer) / sum(fps_buffer), 2)}', (0, int(out.shape[1] * 0.02)),
                        cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)

        if save_video:
            out = cv2.resize(out, (frame_w, frame_h), cv2.INTER_NEAREST)  # Change image size to input size
            out_save.write(out)

        if show_video:
            out = scale_image(out, 1300, cv2.INTER_NEAREST)
            cv2.imshow('win', out)

    if save_video:
        out_save.release()
    cap.release()
    cv2.destroyAllWindows()


auto_number_plate_rec('videos_for_test/1.mp4', save_video=False, show_video=True)
