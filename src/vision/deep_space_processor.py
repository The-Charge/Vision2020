import math

import numpy as np
import cv2

from .target_processor import *
from .contour_tools import *
from .filters import *
from .synced_values import *


__all__ = ['DeepSpaceProcessor']


class DeepSpaceProcessor(TargetProcessor):
    output_success = SyncedValue('Output/success', False)
    output_distance = SyncedValue('Output/distance', 0)
    output_horizontal_angle = SyncedValue('Output/horizontal_angle', 0)
    output_vertical_angle = SyncedValue('Output/vertical_angle', 0)

    min_solidity = SyncedValue('Pipeline/min_solidity')

    def __init__(self, size):
        super().__init__(size)

        self.min_solidity = 0.7
        self.opening_iterations = 2
        self.closing_iterations = 0
        self.blur_radius = 11

        self.distance_filter = MedianFilter(10)

        self.OBJECT_POINTS = np.array([
            [-19.6, 0, 0],
            [-9.8, -17, 0],
            [9.8, -17, 0],
            [19.6, 0, 0],
        ])

    def is_valid(self, contour):
        if len(contour) <= 5 or cv2.contourArea(contour) < 100:
            return False
        elif len(cv2.convexHull(contour)) != 4:
            return False
        elif hull_solidity_of(contour) < self.min_solidity:
            return False
        else:
            return True

    def process_contours(self, image, contours):
        self.output_success = len(contours) > 0
        if self.output_success:
            contour = contours[0]
            hull = cv2.convexHull(contour)

            points = np.array([
                hull[3][0],
                hull[2][0],
                hull[1][0],
                hull[0][0],
            ], dtype='float')

            _, rvec, tvec = cv2.solvePnP(
                self.OBJECT_POINTS,
                points,
                self.camera_matrix,
                self.dist_matrix,
            )

            x = tvec[0][0] + self.x_camera_offset
            y = tvec[1][0] + self.y_camera_offset
            z = tvec[2][0] + self.z_camera_offset

            self.output_distance = self.distance_filter.calculate(
                math.sqrt(x ** 2 + y ** 2 + z ** 2)
            )
            self.output_horizontal_angle = math.degrees(math.atan2(x, z)) + self.horizontal_camera_offset
            self.output_vertical_angle = -math.degrees(math.atan2(y, z)) + self.vertical_camera_offset

            if self.view_modified:
                cv2.drawContours(image, [contour], 0, self.BLUE, 2)
                mid = (
                    (hull[0][0][0] + hull[3][0][0]) // 2,
                    (hull[0][0][1] + hull[3][0][1]) // 2,
                )
                cv2.circle(image, mid, 3, self.RED, -1)
                cv2.circle(image, mid, 10, self.RED, 1)
                for point in hull:
                    center = point[0][0], point[0][1]
                    cv2.circle(image, center, 6, self.WHITE, -1)
                self.left_label(image, [
                    f'Distance: {self.output_distance:.0f}',
                    f'Horizontal: {self.output_horizontal_angle:.0f}',
                    f'Vertical: {self.output_vertical_angle:.0f}',
                    f'Avg. FPS: {self.average_fps:.0f}',
                ])
