import math

import numpy as np
import cv2

from .target_processor import *
from .contour_tools import *
from .synced_values import *


__all__ = ['ExampleProcessor']


class ExampleProcessor(TargetProcessor):
    output_success = SyncedValue('Output/success', False)
    output_distance = SyncedValue('Output/distance', 0)
    output_horizontal_angle = SyncedValue('Output/horizontal_angle', 0)
    output_vertical_angle = SyncedValue('Output/vertical_angle', 0)

    def __init__(self, size):
        super().__init__(size)

        self.lower_hue = 0
        self.lower_sat = 100
        self.lower_val = 140
        self.upper_hue = 45
        self.upper_sat = 210
        self.upper_val = 225

        self.blur_radius = 5
        self.opening_radius = 5
        self.opening_iterations = 2

        self.OBJECT_POINTS = np.array([
            [-5.5, 4.25, 0],
            [5.5, 4.25, 0],
            [5.5, -4.25, 0],
            [-5.5, -4.25, 0],
        ], dtype='float')

    def is_valid(self, contour):
        if len(contour) <= 5 or cv2.contourArea(contour) < 100:
            return False
        elif len(polygon_from(contour)) != 4:
            return False
        else:
            return True

    def process_contours(self, image, contours):
        self.output_success = len(contours) > 0
        if self.output_success:
            contour = contours[0]
            rectangle = polygon_from(contour)
            rectangle = self.order_points(rectangle, center_of(rectangle))

            x, y, z, horizontal_angle, vertical_angle = self.simple_solve(
                self.OBJECT_POINTS,
                np.array([
                    rectangle[0][0],
                    rectangle[1][0],
                    rectangle[2][0],
                    rectangle[3][0],
                ], dtype='float'),
            )
            self.output_horizontal_angle = math.degrees(horizontal_angle)
            self.output_vertical_angle = math.degrees(vertical_angle)
            self.output_distance = math.sqrt(x**2 + y**2 + z**2)

            if self.view_modified:
                cv2.drawContours(image, [rectangle], 0, self.RED, 2)
                cv2.circle(image, tuple(center_of(rectangle)[0]), 5, self.RED, -1)
                cv2.circle(image, tuple(rectangle[0][0]), 5, self.BLACK, -1)
                cv2.circle(image, tuple(rectangle[1][0]), 5, self.WHITE, -1)
                cv2.circle(image, tuple(rectangle[2][0]), 5, self.GREEN, -1)
                cv2.circle(image, tuple(rectangle[3][0]), 5, self.YELLOW, -1)

                self.left_label(image, [
                    f'x: {x:.0f}',
                    f'y: {y:.0f}',
                    f'z: {z:.0f}',
                    f'd: {self.output_distance:.0f}',
                    f'h: {self.output_horizontal_angle:.0f}',
                    f'v: {self.output_vertical_angle:.0f}',
                ], width=75)
