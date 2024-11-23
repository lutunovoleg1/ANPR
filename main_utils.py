import cv2
import numpy as np


def scale_image(input_image: np.ndarray, desired_width: int, interpolation: int) -> np.ndarray:
    """Scales the input image to the desired width while maintaining the aspect ratio.

    Args:
        input_image: The input image to resize.
        desired_width: A number representing the desired width of the output image.
        interpolation: An interpolation method from the OpenCV library.

    Returns:
        Scaled image with the desired width while preserving the aspect ratio.
    """
    aspect_ratio = desired_width / input_image.shape[1]
    desired_height = int(input_image.shape[0] * aspect_ratio)
    return cv2.resize(input_image, (desired_width, desired_height), interpolation)


def find_rectangle_corners(input_img, show_res=True):
    def filter_lines():
        """Categorizes detected vertical and horizontal lines into four groups based on
        their position relative to the image center.

           The function processes two global lists of lines: v_lines (vertical lines) and h_lines (horizontal lines).
           It determines whether each line is on the left or right side of the vertical center point,
           and whether it is on the upper or lower side of the horizontal center point.
           The categorized lines are stored in four separate lists:

           - Vertical lines on the left side of the center
           - Horizontal lines on the upper side of the center
           - Vertical lines on the right side of the center
           - Horizontal lines on the bottom side of the center

           Additionally, for each category, only the longest line is retained.

           Returns:
               list: A list containing four sublists, where:
                   - _line_positions[0] contains vertical line on the left side
                   - _line_positions[1] contains horizontal line on the upper side
                   - _line_positions[2] contains vertical line on the right side
                   - _line_positions[3] contains horizontal line on the bottom side

           Note:
               This function relies on global variables v_lines and h_lines, which should be defined in the outer scope.
               It also uses gray_h.shape and gray_v.shape to determine the center points for categorization.
               The function assumes that lines are represented as tuples/lists with coordinates in the format:
               (_x1, _y1, _x2, _y2) for vertical and horizontal lines.
           """
        central_point_h = (gray_h.shape[1] // 2, gray_h.shape[0] // 2)
        central_point_v = (gray_v.shape[1] // 2, gray_v.shape[0] // 2)

        _line_positions = [[] for _ in range(4)]
        if v_lines is not None:
            for _line in v_lines:
                # We refer to the right or left side by the x coordinate of the middle of the line
                _x1, _, _x2, _ = _line[0]  # x2>x1
                if _x1 + (_x2 - _x1) // 2 > central_point_v[0]:
                    _line_positions[2].append(_line)
                else:
                    _line_positions[0].append(_line)
        if h_lines is not None:
            for _line in h_lines:
                # Relate to the upper or lower line by the y coordinate of the line's middle
                _, _y1, _, _y2 = _line[0]  # y2>y1
                if _y1 + (_y2 - _y1) // 2 > central_point_h[1]:
                    _line_positions[3].append(_line)
                else:
                    _line_positions[1].append(_line)

        # find the longest lines
        for _i in range(len(_line_positions)):
            _line_positions[_i] = sorted(_line_positions[_i], key=lambda val: ((val[0][2] - val[0][0]) ** 2 + (
                    val[0][3] - val[0][1]) ** 2) ** 0.5, reverse=True)[:1]

        return _line_positions

    def original_line_shape():
        """Scale the line positions to the original dimensions of the image.

        This function accepts predefined variables (input_img, gray_h, gray_v, line_positions),
        which must be available in the scope. It calculates the scaling factors
        for horizontal and vertical lines and applies them to the line positions.

        Returns:
        list: The updated list of line positions scaled to the original dimensions
        """
        k_horizontal = [input_img.shape[0] / gray_h.shape[0], input_img.shape[1] / gray_h.shape[1]]
        k_vertical = [input_img.shape[0] / gray_v.shape[0], input_img.shape[1] / gray_v.shape[1]]
        for _i in range(len(line_positions)):
            if line_positions[_i]:
                for _line in line_positions[_i]:
                    if _i == 0 or _i == 2:  # Select coefficient depending on the type of line
                        k = k_vertical
                    else:
                        k = k_horizontal
                    _line[0][0] *= k[1]
                    _line[0][1] *= k[0]
                    _line[0][2] *= k[1]
                    _line[0][3] *= k[0]
        return line_positions

    def find_intersection():
        for _i in range(len(line_positions)):
            if not points[_i] and line_positions[_i] and line_positions[(_i + 1) % 4]:  # Если две соседние линии не пустые
                line_1, line_2 = line_positions[_i][0], line_positions[(_i + 1) % 4][0]
                x1_1, y1_1, x1_2, y1_2 = line_1[0]
                x2_1, y2_1, x2_2, y2_2 = line_2[0]
                m1, p1 = x1_2 - x1_1, y1_2 - y1_1  # Каноническся форма
                m2, p2 = x2_2 - x2_1, y2_2 - y2_1
                a1, b1, c1 = p1, -m1, -p1 * x1_1 + m1 * y1_1  # переход к параметрическому виду
                a2, b2, c2 = p2, -m2, -p2 * x2_1 + m2 * y2_1

                if all([m1, m2, p1, p2]):  # проверка 0
                    bs2 = b2 - a2 / a1 * b1
                    if bs2 != 0:  # Единственное решение
                        cs2 = -c2 + a2 / a1 * c1
                        y = cs2 / bs2
                        x = (-c1 - b1 * y) / a1
                elif (m1 == 0 and p2 == 0) or (m2 == 0 and p1 == 0):
                    if p2 == 0:  # Считам что горизонатльная лини line 1
                        y1_1 = y2_1
                        x2_1 = x1_1  # тогда вертикальная line 2
                    x, y = x2_1, y1_1
                elif m1 == 0 or m2 == 0:  # Если есть вертикальная линия
                    if m2 == 0:  # Меняем местами считаем что вертикальная линия line_1
                        x1_1, a2, b2, c2 = x2_1, a1, b1, c1
                    x = x1_1
                    y = (-a2 * x - c2) / b2
                else:  # Если есть горизонтальная линия
                    if p2 == 0:  # Меняем местами считаем что горизонтальная линия line_1
                        y1_1, a2, b2, c2 = y2_1, a1, b1, c1
                    y = y1_1
                    x = (-b2 * y - c2) / a2

                if x <= input_img.shape[1] and y <= input_img.shape[1]:  # Входит ли пересечение в область изображения
                    x, y = round(x), round(y)
                    points[_i] = [x, y]
        return points

    def refind_lines():
        def back_rotate(_x_out_1, _y_out_1, _x_out_2, _y_out_2):
            """Делаем условный поворот на 90 градусов против часовой стрелки, если поворот был совершен"""
            if _i in [1, 3]:
                _x_out_1, _y_out_1 = _y_out_1, input_img.shape[0] - _x_out_1
                _x_out_2, _y_out_2 = _y_out_2, input_img.shape[0] - _x_out_2
            return _x_out_1, _y_out_1, _x_out_2, _y_out_2

        # Если существует 2 точки пересечения и 3 линии
        if sum(1 for val in points if val) == 2 and sum(1 for val in line_positions if val) == 3:
            for _i in range(len(line_positions)):
                if not line_positions[_i]:
                    x1_1, y1_1, x1_2, y1_2 = line_positions[_i - 1][0][0]  # previous line
                    x2_1, y2_1, x2_2, y2_2 = line_positions[(_i + 1) % 4][0][0]  # next line
                    corner_1, corner_2 = points[(_i + 1) % 4].copy(), points[(_i + 2) % 4].copy()

                    if _i in [1, 3]:  # Make a conditional 90 clockwise turn (only for the missing horizontal line)
                        y1_1, x1_1, y1_2, x1_2 = x1_1, input_img.shape[0] - y1_1, x1_2, input_img.shape[0] - y1_2
                        y2_1, x2_1, y2_2, x2_2 = x2_1, input_img.shape[0] - y2_1, x2_2, input_img.shape[0] - y2_2
                        corner_1[1], corner_1[0] = corner_1[0], input_img.shape[0] - corner_1[1]
                        corner_2[1], corner_2[0] = corner_2[0], input_img.shape[0] - corner_2[1]

                    m1, p1 = x1_2 - x1_1, y1_2 - y1_1  # Каноническся форма
                    m2, p2 = x2_2 - x2_1, y2_2 - y2_1
                    m3, p3 = corner_1[0] - corner_2[0], corner_1[1] - corner_2[1]
                    a1, b1, c1 = p1, -m1, -p1 * x1_1 + m1 * y1_1  # переход к параметрическому виду
                    a2, b2, c2 = p2, -m2, -p2 * x2_1 + m2 * y2_1
                    a3, b3, c3 = p3, -m3, -p3 * corner_2[0] + m3 * corner_2[1]

                    if b3 == 0:  # Если линия имеет нулевую разницу по оси х
                        if _i in [0, 3]:
                            x_out_1, x_out_2, y_out_1, y_out_2 = 0, 0, corner_1[1], corner_2[1]
                        else:
                            y_out_1, y_out_2 = corner_1[1], corner_2[1]
                            if _i == 2:
                                x_out_1 = x_out_2 = input_img.shape[1]
                            else:
                                x_out_1 = x_out_2 = input_img.shape[0]
                        line_positions[_i] = [[[*back_rotate(x_out_1, y_out_1, x_out_2, y_out_2)]]]
                        return line_positions
                    elif a3 / b3 >= 0:  # От наклона зависит от какой линии будем брать прямую
                        a, b, c = a1, b1, c1
                    else:
                        a, b, c = a2, b2, c2

                    if _i in [0, 3]:  # Определяем от какой границы будем строить линию
                        x_out_1 = 0
                    elif _i == 1:
                        x_out_1 = input_img.shape[0]
                    else:
                        x_out_1 = input_img.shape[1]
                    y_out_1 = int((-a * x_out_1 - c) / b)

                    # Параллельно переносим
                    if _i in [0, 3]:  # Поиск х координаты
                        x_out_2 = x_out_1 + abs(corner_1[0] - corner_2[0])
                    else:
                        x_out_2 = x_out_1 - abs(corner_1[0] - corner_2[0])

                    if a3 / b3 <= 0:  # Поиск у координаты
                        y_out_2 = y_out_1 + abs(corner_1[1] - corner_2[1])
                    else:
                        y_out_2 = y_out_1 - abs(corner_1[1] - corner_2[1])

                    line_positions[_i] = [[[*back_rotate(x_out_1, y_out_1, x_out_2, y_out_2)]]]
                    break

        return line_positions

    img_copy = input_img.copy()
    img_copy = cv2.resize(img_copy, (150, 60), cv2.INTER_LINEAR)  # Get one size fits all
    # Фильтрация изображения
    gray = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)
    gray_f = cv2.bilateralFilter(gray, d=11, sigmaColor=200, sigmaSpace=200)

    # Изменяем разрешение для лучшего обнаружения вертикальной и горизонтальной линии номера
    gray_h = gray_f.copy()
    gray_v = cv2.resize(gray_f, None, fx=1, fy=2)

    # Детекция краев
    sobel_h = cv2.Sobel(src=gray_h, scale=1, delta=0, ddepth=cv2.CV_32F, dx=0, dy=1,
                        ksize=3)  # Sobel Edge Detection on the Y axis
    sobel_h = cv2.convertScaleAbs(sobel_h)
    sobel_h[:, :int(sobel_h.shape[0] * 0.2)] = 0
    sobel_v = cv2.Sobel(src=gray_v, scale=1, delta=0, ddepth=cv2.CV_32F, dx=1, dy=0,
                        ksize=3)  # Sobel Edge Detection on the X axis
    sobel_v = cv2.convertScaleAbs(sobel_v)
    # Урезаем часть изображения градиента
    sobel_h[:, :int(sobel_h.shape[1] * 0.1)], sobel_h[:, int(sobel_h.shape[1] * 0.9):] = 0, 0
    sobel_v[:int(sobel_v.shape[0] * 0.2), :], sobel_v[int(sobel_v.shape[0] * 0.8):, :] = 0, 0

    # Бинаризация производных
    _, thresh_h = cv2.threshold(sobel_h, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thresh_h = cv2.morphologyEx(thresh_h, cv2.MORPH_ERODE, np.ones((2, 10)), iterations=1)
    _, thresh_v = cv2.threshold(sobel_v, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thresh_v = cv2.morphologyEx(thresh_v, cv2.MORPH_ERODE, np.ones((5, 5)), iterations=1)

    # Использование Hough Transform для нахождения линий
    # Для собеля по у
    h_lines = cv2.HoughLinesP(thresh_h.astype(np.uint8), 1, np.pi / 180, threshold=30, minLineLength=30, maxLineGap=5)
    # Для собеля по х
    v_lines = cv2.HoughLinesP(thresh_v.astype(np.uint8), 1, np.pi / 180, threshold=30, minLineLength=20, maxLineGap=5)

    line_positions = filter_lines()  # Фильтрация линий
    line_positions = original_line_shape()  # Вернуть оригинальный размер линий
    points = [[] for _ in range(4)]  # left_top, right_top, right_bottom, left_bottom
    points = find_intersection()  # Найти пересечения линий
    line_positions = refind_lines()
    points = find_intersection()  # Найти пересечения линий

    if show_res:  # show all results
        # Return input shape
        img_copy = cv2.resize(img_copy, [input_img.shape[1], input_img.shape[0]], cv2.INTER_NEAREST)
        for i in range(len(line_positions)):  # show lines
            if line_positions[i] is not None:
                if i == 0:
                    color = (255, 0, 0)
                elif i == 1:
                    color = (0, 255, 0)
                elif i == 2:
                    color = (0, 0, 255)
                else:
                    color = (255, 255, 255)
                for line in line_positions[i]:
                    x1, y1, x2, y2 = line[0]
                    cv2.line(img_copy, (x1, y1), (x2, y2), color, 1)

        # Show dots
        for point in points:
            if point:
                cv2.circle(img_copy, point, 5, (0, 0, 255), -1)

        stack_images_h = np.hstack((gray_h, sobel_h, thresh_h))
        stack_images_v = np.hstack((gray_v, sobel_v, thresh_v))
        cv2.imshow('sobel_h, thresh_h', stack_images_h)
        cv2.imshow('sobel_v, thresh_v', stack_images_v)
        cv2.imshow('find_rectangle_corners_result', img_copy)
        cv2.imshow('filtered_gray_image', gray_f)
        cv2.waitKey(1)

    return points


def apply_perspective_transform(input_image: np.ndarray, points: list[list[int, int]], show_results=True) \
        -> tuple[bool, np.ndarray]:
    """Applies a perspective transformation to an image.

    Args:
        input_image: The input image to which the transformation should be applied
        points : A list of four points, where each point is represented as a list of two integers
        (x, y) that define the corners  of the region to be transformed.
        The points should be in the order: top-left,top-right, bottom-right, bottom-left.
        show_results (bool): A flag indicating whether to display the transformed image. Defaults to True.

    Returns:
        A tuple where the first element is a boolean indicating whether the transformation was successful (True)
        or not (False), and the second element is the transformed image as a numpy array.
        If the transformation fails, the original input image is returned as the second element of the tuple.
    """
    if not all(points):
        return False, input_image

    img_copy = input_image.copy()
    in_points = np.array(points, dtype='float32')

    # Define the width and height
    top_left, top_right, bottom_left, bottom_right = in_points
    right_height = np.sqrt((top_right[0] - bottom_right[0]) ** 2 + (top_right[1] - bottom_right[1]) ** 2)
    left_height = np.sqrt((top_left[0] - bottom_left[0]) ** 2 + (top_left[1] - bottom_left[1]) ** 2)

    # Output image size
    max_height = max(int(right_height), int(left_height))
    max_width = round(4.64 * max_height)  # Use the real ratio of width and height

    # Perspective transformation
    cvt_points = np.float32(([0, 0], [max_width, 0], [0, max_height], [max_width, max_height]))
    matrix = cv2.getPerspectiveTransform(in_points, cvt_points)
    img_out = cv2.warpPerspective(img_copy, matrix, (max_width, max_height))

    if show_results:  # Show result
        cv2.imshow('perspective_transform', img_out)
        cv2.waitKey(1)

    return True, img_out


def prepare_result(ocr_result: list[tuple[list, str, float]]) -> tuple[str, bool, tuple] | tuple[None, None, None]:
    """Processes OCR recognition results by comparing them with the car number template.

    Args:
        ocr_result: A list of recognition results, where each element is a tuple containing:
            - coordinates: The position (x, y) of the recognized text.
            - text: Recognized text.
            - confidence: A confidence indicator for the recognized text.

    Returns:
        tuple: A tuple containing:
            - text: Processed text with character substitution according to the rules.
            - template_valid: A flag indicating whether the number is valid (True if it matches the pattern).
            - confs: A tuple of confidence indicators for each recognition result.

    Notes:
        The function replaces '0' with 'О' and '8' with 'В' in specific positions.
        It checks the length of the number and verifies that each character matches the specified rules.
        If all characters are valid according to the template, it returns True for template_valid.
        If the ocr_result list is empty, it returns (None, None, None).
    """
    num2char = {'0': 'О', '8': 'В'}
    char2num = {'О': '0', 'В': '8'}

    if ocr_result:
        coordinates, texts, confs = list(zip(*ocr_result))
        text = texts[0]

        template_valid = False
        counter = 0
        if 8 <= len(text) <= 9:  # Check number length
            for i in range(len(text)):
                if i in [0, 4, 5]:
                    if not text[i].isalpha():  # Letter processing
                        text = text[:i] + num2char.get(text[i], text[i]) + text[i + 1:]
                    if text[i].isalpha():
                        counter += 1
                if i in [1, 2, 3, 6, 7, 8]:  # Processing of numbers
                    if not text[i].isdigit():
                        text = text[:i] + char2num.get(text[i], text[i]) + text[i + 1:]
                    if text[i].isdigit():
                        counter += 1
            if counter == len(text):  # Checking the length after permutations
                template_valid = True
        return text, template_valid, confs
    else:
        return None, None, None
