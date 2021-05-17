"""The base vision processing class."""


from abc import ABC, abstractmethod
import math

import numpy as np
import cv2

from .synced_values import SyncedValue


__all__ = ['Processor']


class Processor(ABC):
    """Abstract base class for processors that provides convenience features
    for testing."""

    # OpenCV uses BGR instead of RGB
    GREEN = (0, 255, 0)
    RED = (0, 0, 255)
    BLUE = (255, 0, 0)
    WHITE = (255, 255, 255)
    BLACK = (0, 0, 0)
    YELLOW = (0, 255, 255)

    def __init__(self):
        # Create the NetworkTables listener for every SyncedValue
        for key in dir(self):
            getattr(self, key)

    @abstractmethod
    def process_image(self, image):
        """The main method of the class. When called it will return any
        calculated values and an image (either the original unmodified image or
        a modified one)."""
        pass

    @classmethod
    def left_label(cls, BGR_image, labels, width=175):
        """Draw a black box with text in the top-left on an image."""

        y = 20
        cv2.rectangle(
            BGR_image,
            (-10, -10),
            (width, len(labels) * y + 10),
            cls.BLACK,
            -1,
        )
        cv2.rectangle(
            BGR_image,
            (-10, -10),
            (width, len(labels) * y + 10),
            cls.WHITE,
            1,
        )

        for label in labels:
            cv2.putText(
                BGR_image,
                label,
                (10, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                .5,
                cls.WHITE,
                1,
                cv2.LINE_AA,
            )
            y += 20

    @classmethod
    def label_contour(cls, BGR_image, contour, text):
        """Add a text label directly above a given contour."""
        text = str(text)
        font = cv2.FONT_HERSHEY_SIMPLEX
        size, _ = cv2.getTextSize(text, font, 1, 1)

        left = tuple(contour[contour[:, :, 0].argmin()][0])
        top = tuple(contour[contour[:, :, 1].argmin()][0])
        point = left[0], top[1] - 10
        point_2 = point[0] + size[0], point[1] - size[1]

        cv2.rectangle(BGR_image, point, point_2, cls.BLACK, -1)
        cv2.putText(BGR_image, text, point, font, 1, cls.WHITE, 1, cv2.LINE_AA)

    @staticmethod
    def create_rotation_matrix(degrees):
        """Turn an angle in degrees to a 3D rotation matrix about the z-axis."""
        degrees = math.radians(degrees)
        return np.array([
            [math.cos(degrees), -math.sin(degrees), 0],
            [math.sin(degrees), math.cos(degrees), 0],
            [0, 0, 1],
        ])

    @staticmethod
    def draw_guidelines(image, color=None):
        """Draw a vertical and horizontal guideline on the center axes of the
        image."""
        size = image.shape[1], image.shape[0]
        center = size[0] // 2, size[1] // 2
        cv2.line(
            image,
            (center[0], 0),
            (center[0], size[1]),
            color,
            1,
        )
        cv2.line(
            image,
            (0, center[1]),
            (size[0], center[1]),
            color,
            1,
        )

    @staticmethod
    def order_points(points, center):
        """Sort points clockwise around a center point, starting from directly
        above the center."""
        return np.array(sorted(
            points,
            key=lambda p: math.atan2(p[0][1]-center[0][1], p[0][0]-center[0][0]),
        ))
