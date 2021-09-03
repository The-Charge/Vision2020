"""Some shortcuts to common OpenCV contour functions, mostly from
https://bit.ly/3eX3hNd."""


import math
import cv2
import numpy as np


__all__ = [
    'aspect_ratio_of',
    'extent_of',
    'hull_solidity_of',
    'circle_solidity_of',
    'equivalent_diameter_of',
    'orientation_of',
    'polygon_from',
    'center_of',
]


def aspect_ratio_of(contour):
    """Aspect ratio is the ratio of width to height of bounding rect of the object."""
    x, y, w, h = cv2.boundingRect(contour)
    return w / h


def extent_of(contour):
    """Extent is the ratio of contour area to bounding rectangle area."""
    x, y, w, h = cv2.boundingRect(contour)
    return cv2.contourArea(contour) / (w * h)


def hull_solidity_of(contour):
    """Solidity is the ratio of contour area to its convex hull area."""
    return cv2.contourArea(contour) / cv2.contourArea(cv2.convexHull(contour))


def circle_solidity_of(contour):
    """Solidity is the ratio of contour area to its bounding circle area."""
    _, radius = cv2.minEnclosingCircle(contour)
    return cv2.contourArea(contour) / (math.pi * radius**2)


def equivalent_diameter_of(contour):
    """Equivalent Diameter is the diameter of the circle whose area is same as
    the contour area."""
    return math.sqrt(4 * cv2.contourArea(contour) / math.pi)


def orientation_of(contour):
    """Orientation is the angle at which object is directed."""
    _, _, angle = cv2.fitEllipse(contour)
    return angle


def polygon_from(contour, tolerance=0.05):
    """Estimate a polygon from a contour. The tolerance is a percentage value.
    When OpenCV is creating the polygon, it calculates the distance between the
    current line and the next point in the contour. If that distance is greater
    than the tolerance times the length of the contour, it creates a new line.

    Note that the polygon returned from this is itself a contour, so functional
    with all these functions.

    Also there's no function here to get the convex hull/convex polygon, but
    that's just because it's 1 line normally: cv2.convexHull(contour)
    """
    return cv2.approxPolyDP(
        contour,
        tolerance * cv2.arcLength(contour, True),
        True,
    )


def center_of(contour):
    """Return the center of a contour, in the form np.array([[x, y]])."""
    M = cv2.moments(contour)
    return np.array([[int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])]])
