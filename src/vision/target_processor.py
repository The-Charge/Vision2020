import math
from abc import abstractmethod
import time

import numpy as np
import cv2

from .processor import *
from .synced_values import *
from .filters import *


__all__ = ['TargetProcessor']


class TargetProcessor(Processor):
    # Debug values
    view_thresh = SyncedValue('Debug/view_thresh', False)
    view_modified = SyncedValue('Debug/view_modified', False)
    view_contours = SyncedValue('Debug/view_contours', False)
    instant_fps = SyncedValue('Debug/fps', 0)
    average_fps = SyncedValue('Debug/average_fps', 0)
    timestamp = SyncedValue('Debug/timestamp', 0)

    # Camera offsets for tuning. Needed when the camera is not mounted perfectly
    x_camera_offset = SyncedValue('Camera Offset/x', 0)
    y_camera_offset = SyncedValue('Camera Offset/y', 0)
    z_camera_offset = SyncedValue('Camera Offset/z', 0)
    horizontal_camera_offset = SyncedValue('Camera Offset/horizontal_angle', 0)
    vertical_camera_offset = SyncedValue('Camera Offset/vertical_angle', 0)

    # Threshold values
    lower_hue = SyncedValue('Pipeline/Threshold/Lower/hue', 0)
    lower_sat = SyncedValue('Pipeline/Threshold/Lower/sat', 0)
    lower_val = SyncedValue('Pipeline/Threshold/Lower/val', 0)
    upper_hue = SyncedValue('Pipeline/Threshold/Upper/hue', 255)
    upper_sat = SyncedValue('Pipeline/Threshold/Upper/sat', 255)
    upper_val = SyncedValue('Pipeline/Threshold/Upper/val', 255)

    # Pipeline values
    blur_radius = SyncedValue('Pipeline/blur_radius', 7)
    opening_radius = SyncedValue('Pipeline/opening_radius', 5)
    opening_iterations = SyncedValue('Pipeline/opening_iterations', 1)

    def __init__(self, size):
        super().__init__()
        self.fps_filter = MovingAverage(15)

        self.size = size
        self.center = (
            size[0] // 2,
            size[1] // 2,
        )

        # Estimate the values of the camera and distortion matrices. You could
        # do this officially with a checkerboard, but I've found unless your
        # camera has huge distortion, you'll be fine. Also, I was unable to
        # disable the auto-focus for the camera I used, and if you can't do that
        # then calibration is useless. Taken from https://bit.ly/3hzsGPN.
        self.camera_matrix = np.array([
            [self.size[0], 0, self.center[0]],
            [0, self.size[0], self.center[1]],
            [0, 0, 1]
        ], dtype='double')
        self.dist_matrix = np.zeros((4, 1))

    @abstractmethod
    def process_contours(self, image, contours):
        """Do something with the list of valid contours.

        Args:
            image: if you want to modify in place to add visuals
            contours: list of valid contours, sorted largest to smallest
        """
        pass

    @abstractmethod
    def is_valid(self, contour):
        """Given a contour, return true/false whether it's a valid target."""
        pass

    def process_image(self, image):
        """Take an input frame, process contours, (optionally) return a modified
        output frame."""
        start_time = time.time()

        # Apply blur
        if self.blur_radius > 1:
            image = cv2.blur(image, (int(self.blur_radius), int(self.blur_radius)))

        # Apply thresholding
        thresh = cv2.inRange(
            cv2.cvtColor(image, cv2.COLOR_BGR2HSV),
            (self.lower_hue, self.lower_sat, self.lower_val),
            (self.upper_hue, self.upper_sat, self.upper_val),
        )

        # Apply opening
        if self.opening_iterations > 0:
            kernel = np.ones((int(self.opening_radius), int(self.opening_radius)), np.uint8)
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=int(self.opening_iterations))

        if self.view_thresh:
            image = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)

        _, contours, _ = cv2.findContours(
            thresh,
            cv2.RETR_LIST,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        contours = sorted(
            [c for c in contours if self.is_valid(c)],  # filter by is_valid method
            key=lambda c: cv2.contourArea(c),  # sort by area
            reverse=True,  # largest to smallest
        )

        if self.view_modified and self.view_contours:
            cv2.drawContours(image, contours, -1, self.BLUE, 2)

        # Run user code to process contours
        image = self.process_contours(image, contours) or image

        if self.view_modified:
            self.draw_guidelines(image, self.YELLOW)

        self.timestamp = time.time()

        fps = 1 / (self.timestamp - start_time)
        self.instant_fps = round(fps)
        self.average_fps = round(self.fps_filter.calculate(fps))

        return image

    def simple_solve(self, object_points, image_points):
        """Use solvePnP to calculate the x, y, z distances as well as the angles
        to the target.

        The format of object_points and image_points is very important, they
        must be np arrays with the 'float' datatype; you cannot just pass the
        contour. Additionally, the two sets of points must be in the same order.
        """

        _, rvec, tvec = cv2.solvePnP(
            object_points,
            image_points,
            self.camera_matrix,
            self.dist_matrix,
        )

        x = tvec[0][0] + self.x_camera_offset
        y = -(tvec[1][0] + self.y_camera_offset)
        z = tvec[2][0] + self.z_camera_offset

        horizontal_angle = math.atan2(x, z) + self.horizontal_camera_offset
        vertical_angle = math.atan2(y, z) + self.vertical_camera_offset

        return x, y, z, horizontal_angle, vertical_angle
