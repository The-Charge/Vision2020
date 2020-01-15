"""Some code I used in the off season to detect a different type of ball."""

import math

import numpy as np
import cv2 as cv
import imutils

from ProcessorBase import ProcessorBase


class BallProcessing(ProcessorBase):
    def __init__(self, camera_matrix, dist_matrix):
        super().__init__()
        self.camera_matrix = camera_matrix
        self.dist_matrix = dist_matrix
        self.ball_radius = 5.5  # units can be whatever you want
        self.obj_p = np.array([
            [self.ball_radius, self.ball_radius, 0],
            [0, self.ball_radius, 0],
            [self.ball_radius * 2, self.ball_radius, 0],
            [self.ball_radius, 0, 0],
            [self.ball_radius, self.ball_radius * 2, 0]
        ])

    def process_image(self, img):
        blurred = cv.GaussianBlur(img, (65, 65), 11)
        hsv = cv.cvtColor(blurred, cv.COLOR_BGR2HSV)

        # lower = (135, 85, 100)
        lower = (135, 100, 100)
        upper = (180, 200, 200)
        mask = cv.inRange(hsv, lower, upper)
        mask = cv.erode(mask, None, iterations=2)
        mask = cv.dilate(mask, None, iterations=2)

        cnts = cv.findContours(mask.copy(), cv.RETR_EXTERNAL,
                               cv.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        cnts = [cnt for cnt in cnts if cv.contourArea(cnt) > 1000]
        distance, angle1 = None, None
        if cnts is not None and len(cnts) is not 0:
            cnt = max(cnts, key=cv.contourArea)
            (x, y), radius = cv.minEnclosingCircle(cnt)
            vertices = np.array([
                (x, y),
                (x - radius, y),
                (x + radius, y),
                (x, y - radius),
                (x, y + radius)])

            cv.drawContours(img, [cnt], -1, (255, 0, 0), 1)

            cv.circle(img, (int(x), int(y)), int(radius), (0, 255, 0), 3)
            for v in vertices:
                cv.circle(img, (int(v[0]), int(v[1])), 2, (0, 0, 255), 3)

            _, rvec, tvec = cv.solvePnP(self.obj_p, vertices, self.camera_matrix,
                                        self.dist_matrix)
            x = tvec[0][0] + self.ball_radius
            x = tvec[1][0] + self.ball_radius
            z = tvec[2][0]
            distance = math.sqrt(x**2 + z**2)
            angle1 = math.degrees(math.atan2(x, z))
            labels = [
                ('Angle 1', angle1),
                ('Distance', distance)]
        else:
            labels = [('Ball Not Found')]

        self.draw_labels(img, labels)
        self.draw_center(img)

        # return (distance, angle1, None), img
        return (distance, angle1, None), img
