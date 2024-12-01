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


def find_rectangle_corners(input_img: np.ndarray, show_results=False) -> list[list[int, int]]:
    """This function takes an image of the license plate of the car and finds the coordinates of the points of
    intersection of its borders

    It takes an image as input and returns a list of points, where each point is represented as a list of two
    coordinates (x, y). First, the function reduces the image size to a fixed size, then applies a two-sided filter,
    determines the edges using the Sobel operator, and then determines the lines using the Hough transform.
    Then it filters the lines by their location and finds the points where the lines intersect.
    Finally, it resizes the intersection points to the original image size and returns them.

    Args:
        input_img: The input image of the license plate of the car.
        show_results: A boolean value indicating whether to show the results of the function.

    Returns:
        A list of points, where each point is represented as a list of two coordinates (x, y).
    """

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
        """Finds the intersection points between pairs of lines defined in the global variable line_positions.

         The function checks for intersections between adjacent lines and saves the coordinates of the
         intersection point to the global points list if an intersection is found.

         Returns:
            list: An updated list of points, where each entry contains the coordinates of the intersection point

         Notes:
            - It is assumed that line_positions, points and input_img are defined globally.
            - Intersections are checked only for adjacent lines specified in line_positions.
            - The coordinates of the intersection are rounded to the nearest integer values.
            - Checking for coordinates going beyond the limits of the image
            """
        for _i in range(len(line_positions)):
            # If the point does not exist and two adjacent lines exist
            if not points[_i] and line_positions[_i] and line_positions[(_i + 1) % 4]:
                line_1, line_2 = line_positions[_i][0], line_positions[(_i + 1) % 4][0]
                x1_1, y1_1, x1_2, y1_2 = line_1[0]
                x2_1, y2_1, x2_2, y2_2 = line_2[0]

                m1, p1 = x1_2 - x1_1, y1_2 - y1_1  # Canonical form
                m2, p2 = x2_2 - x2_1, y2_2 - y2_1

                a1, b1, c1 = p1, -m1, -p1 * x1_1 + m1 * y1_1  # Switching to a parametric form
                a2, b2, c2 = p2, -m2, -p2 * x2_1 + m2 * y2_1

                if all([m1, m2, p1, p2]):  # Arbitrary lines
                    bs2 = b2 - a2 / a1 * b1
                    if bs2 != 0:  # The only solution
                        cs2 = -c2 + a2 / a1 * c1
                        y = cs2 / bs2
                        x = (-c1 - b1 * y) / a1

                elif (m1 == 0 and p2 == 0) or (m2 == 0 and p1 == 0):  # Horizontal and vertical line or vice versa
                    if p2 == 0:  # Consider that the horizontal line is line 1
                        y1_1 = y2_1
                        x2_1 = x1_1  # Then the vertical line is 2
                    x, y = x2_1, y1_1

                elif m1 == 0 or m2 == 0:  # If there is a vertical line
                    if m2 == 0:  # Swapping places, we consider that the vertical line is line_1
                        x1_1, a2, b2, c2 = x2_1, a1, b1, c1
                    x = x1_1
                    y = (-a2 * x - c2) / b2

                else:  # If there is a horizontal line
                    if p2 == 0:  # Swapping places, we consider that the horizontal line is line_1
                        y1_1, a2, b2, c2 = y2_1, a1, b1, c1
                    y = y1_1
                    x = (-b2 * y - c2) / a2

                # Checking the occurrence of an intersection in the image area
                if x <= input_img.shape[1] and y <= input_img.shape[1]:
                    x, y = round(x), round(y)
                    points[_i] = [x, y]
        return points

    def refind_lines():
        """Defines and restores lines in the image.

        Checks for two intersection points and three lines. If the conditions are met, calculates the missing line based
        on the coordinates of neighboring lines and their angles. Handles the case when the X-axis difference is zero.

        Returns:
            list: An updated list of line positions, where each position is represented as a list of coordinates.
        """

        def back_rotate(_x_out_1, _y_out_1, _x_out_2, _y_out_2):
            """Make a conditional 90-degree counterclockwise turn if the turn has been made"""
            if _i in [1, 3]:
                _x_out_1, _y_out_1 = _y_out_1, input_img.shape[0] - _x_out_1
                _x_out_2, _y_out_2 = _y_out_2, input_img.shape[0] - _x_out_2
            return _x_out_1, _y_out_1, _x_out_2, _y_out_2

        if sum(1 for val in points if val) == 2 and sum(1 for val in line_positions if val) == 3:
            for _i in range(len(line_positions)):
                if not line_positions[_i]:  # Searching for a missing line
                    x1_1, y1_1, x1_2, y1_2 = line_positions[_i - 1][0][0]  # previous line
                    x2_1, y2_1, x2_2, y2_2 = line_positions[(_i + 1) % 4][0][0]  # next line
                    corner_1, corner_2 = points[(_i + 1) % 4].copy(), points[(_i + 2) % 4].copy()

                    if _i in [1, 3]:  # Make a conditional 90 clockwise turn (only for the missing top and bottom lines)
                        y1_1, x1_1, y1_2, x1_2 = x1_1, input_img.shape[0] - y1_1, x1_2, input_img.shape[0] - y1_2
                        y2_1, x2_1, y2_2, x2_2 = x2_1, input_img.shape[0] - y2_1, x2_2, input_img.shape[0] - y2_2
                        corner_1[1], corner_1[0] = corner_1[0], input_img.shape[0] - corner_1[1]
                        corner_2[1], corner_2[0] = corner_2[0], input_img.shape[0] - corner_2[1]

                    m1, p1 = x1_2 - x1_1, y1_2 - y1_1  # Canonical form
                    m2, p2 = x2_2 - x2_1, y2_2 - y2_1
                    m3, p3 = corner_1[0] - corner_2[0], corner_1[1] - corner_2[1]

                    a1, b1, c1 = p1, -m1, -p1 * x1_1 + m1 * y1_1  # transition to a parametric form
                    a2, b2, c2 = p2, -m2, -p2 * x2_1 + m2 * y2_1
                    a3, b3, c3 = p3, -m3, -p3 * corner_2[0] + m3 * corner_2[1]

                    if b3 == 0:  # If the line has zero x-axis difference
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

                    elif a3 / b3 > 0:  # The slope depends on which line we will take a straight line
                        a, b, c = a1, b1, c1
                    else:
                        a, b, c = a2, b2, c2

                    if _i in [0, 3]:  # Determine which border we will build the line from
                        x_out_1 = 0
                    elif _i == 1:
                        x_out_1 = input_img.shape[0]
                    else:
                        x_out_1 = input_img.shape[1]
                    y_out_1 = int((-a * x_out_1 - c) / b)

                    # Transfer the found line in parallel
                    dx = abs(corner_1[0] - corner_2[0])
                    dy = abs(corner_1[1] - corner_2[1])
                    if _i in [0, 3]:
                        x_out_2 = x_out_1 + dx
                        y_out_2 = y_out_1 + dy if a3 / b3 < 0 else y_out_1 - dy
                    else:
                        x_out_2 = x_out_1 - dx
                        y_out_2 = y_out_1 - dy if a3 / b3 < 0 else y_out_1 + dy

                    line_positions[_i] = [[[*back_rotate(x_out_1, y_out_1, x_out_2, y_out_2)]]]
                    break

        return line_positions

    img_copy = input_img.copy()
    img_copy = cv2.resize(img_copy, (150, 60), cv2.INTER_LINEAR)  # Get one size fits all

    # Image filtering
    gray = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)
    gray_f = cv2.bilateralFilter(gray, d=11, sigmaColor=200, sigmaSpace=200)

    # Change the resolution for better detection of the vertical and horizontal line of the number plate
    gray_h = gray_f.copy()
    gray_v = cv2.resize(gray_f, None, fx=1, fy=2)

    # Edge detection
    sobel_h = cv2.Sobel(src=gray_h, scale=1, delta=0, ddepth=cv2.CV_32F, dx=0, dy=1,
                        ksize=3)
    sobel_h = cv2.convertScaleAbs(sobel_h)
    sobel_v = cv2.Sobel(src=gray_v, scale=1, delta=0, ddepth=cv2.CV_32F, dx=1, dy=0,
                        ksize=3)
    sobel_v = cv2.convertScaleAbs(sobel_v)

    # Crop part of the gradient image to remove unnecessary gradients
    sobel_h[:, :int(sobel_h.shape[1] * 0.1)], sobel_h[:, int(sobel_h.shape[1] * 0.9):] = 0, 0
    sobel_v[:int(sobel_v.shape[0] * 0.2), :], sobel_v[int(sobel_v.shape[0] * 0.8):, :] = 0, 0

    # Binarization of gradient images and noise removal
    _, thresh_h = cv2.threshold(sobel_h, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thresh_h = cv2.morphologyEx(thresh_h, cv2.MORPH_ERODE, np.ones((2, 10)), iterations=1)
    _, thresh_v = cv2.threshold(sobel_v, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thresh_v = cv2.morphologyEx(thresh_v, cv2.MORPH_ERODE, np.ones((5, 5)), iterations=1)

    # Using Hough Transform to find lines
    h_lines = cv2.HoughLinesP(thresh_h.astype(np.uint8), 1, np.pi / 180, threshold=30, minLineLength=30, maxLineGap=5)
    v_lines = cv2.HoughLinesP(thresh_v.astype(np.uint8), 1, np.pi / 180, threshold=30, minLineLength=20, maxLineGap=5)

    line_positions = filter_lines()  # Classify lines by their location and search for lines of maximum
    line_positions = original_line_shape()  # Resize the lines to the size of the input image

    # Finding the coordinates of the intersection of lines
    points = [[] for _ in range(4)]
    points = find_intersection()
    line_positions = refind_lines()
    points = find_intersection()

    if show_results:  # show all results
        # Return input shape
        img_copy = cv2.resize(img_copy, (input_img.shape[1], input_img.shape[0]), cv2.INTER_NEAREST)
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


def apply_perspective_transform(input_image: np.ndarray, points: list[list[int, int]], show_results=False) \
        -> tuple[bool, np.ndarray]:
    """Applies a perspective transformation to an image.

    Args:
        input_image: The input image to which the transformation should be applied
        points : A list of four points, where each point is represented as a list of two integers
        (x, y) that define the corners  of the region to be transformed.
        The points should be in the order: top-left,top-right, bottom-right, bottom-left.
        show_results (bool): A flag indicating whether to display the transformed image. Defaults to False.

    Returns:
        A tuple where the first element is a boolean indicating whether the transformation was successful (True)
        or not (False), and the second element is the transformed image as a numpy array.
        If the transformation fails, the original input image is returned as the second element of the tuple.
    """
    if not all(points):
        return False, input_image

    img_copy = input_image.copy()
    points = np.array(points)
    in_points = np.zeros((4, 2), dtype='float32')

    points_sum = np.sum(points, axis=1)
    in_points[0] = points[np.argmin(points_sum)]
    in_points[3] = points[np.argmax(points_sum)]

    points_dif = np.diff(points, axis=1)
    in_points[1] = points[np.argmin(points_dif)]
    in_points[2] = points[np.argmax(points_dif)]

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
                    continue
                if i in [1, 2, 3, 6, 7, 8]:  # Processing of numbers
                    if not text[i].isdigit():
                        text = text[:i] + char2num.get(text[i], text[i]) + text[i + 1:]
                    if text[i].isdigit():
                        counter += 1
                    continue
            if counter == len(text):  # Checking the length after permutations
                template_valid = True
        return text, template_valid, confs
    else:
        return None, None, None
