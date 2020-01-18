#!/usr/bin/env python3
"""
This file detects vision targets on a Raspberry Pi 3 and uploads them to NetworkTables.

"""

import json
import time
import sys
import math

from cscore import UsbCamera, MjpegServer, CvSink, CvSource, VideoMode
from networktables import NetworkTablesInstance
import ntcore
import numpy as np
import cv2

config_file = '/boot/frc.json'

team = None
server = None
camera_configs = []
cameras = []


class ProcessorBase:
    """A class that stores methods that could apply to any processor, such as drawing labels and creating matrices."""
    GREEN = (0, 255, 0)
    RED = (0, 0, 255)
    BLUE = (255, 0, 0)
    WHITE = (255, 255, 255)
    BLACK = (0, 0, 0)
    YELLOW = (0, 255, 255)

    def label(self, img, labels, width=175):
        """Draw a black box with text in the top-left on an image."""
        font = cv2.FONT_HERSHEY_SIMPLEX
        y = 20
        cv2.rectangle(img, (-10, -10), (width, len(labels)*y + 10),
                      self.BLACK, -1)
        cv2.rectangle(img, (-10, -10), (width, len(labels)*y + 10),
                      self.WHITE, 1)

        for label in labels:
            if type(label) is not tuple and type(label) is not list:
                label = [label]
            if len(label) == 1:
                text = label[0]
            if len(label) == 2:
                text = '{}: {}'.format(label[0], round(label[1]))
            if len(label) == 3:
                text = '{}: {}'.format(label[0], round(label[1], label[2]))
            cv2.putText(img, text, (10, y), font, .5,
                        self.WHITE, 1, cv2.LINE_AA)
            y += 20

    def create_rotation_matrix(self, angle):
        """Turn an angle in degrees to a 3D rotation matrix about the z-axis."""
        angle = math.radians(angle)
        R = np.array([[math.cos(angle), -math.sin(angle), 0],
                      [math.sin(angle), math.cos(angle), 0],
                      [0, 0, 1]])
        return R

    def rotate(self, img, angle):
        """Takes an image and rotates it clockwise."""
        if angle == 0:
            return img

        # I took the code from this tutorial:
        #  https://www.pyimagesearch.com/2017/01/02/rotate-images-correctly-with-opencv-and-python/
        h, w = img.shape[:2]
        cX, cY = w // 2, h // 2

        R = cv2.getRotationMatrix2D((cX, cY), -angle, 1)
        cos = np.abs(R[0, 0])
        sin = np.abs(R[0, 1])

        nW = int((h * sin) + (w * cos))
        nH = int((h * cos) + (w * sin))

        R[0, 2] += (nW / 2) - cX
        R[1, 2] += (nH / 2) - cY

        return cv2.warpAffine(img, R, (nW, nH))
    
    def label_cnt(self, img, cnt, text):
        text = str(text)
        font = cv2.FONT_HERSHEY_SIMPLEX
        size, _ = cv2.getTextSize(text, font, 1, 1)

        left = tuple(cnt[cnt[:, :, 0].argmin()][0])
        top = tuple(cnt[cnt[:, :, 1].argmin()][0])
        point = left[0], top[1] - 10
        point_2 = point[0] + size[0], point[1] - size[1]

        cv2.rectangle(img, point, point_2, self.BLACK, -1)
        cv2.putText(img, text, point, font, 1, self.WHITE, 1, cv2.LINE_AA)


class BallProcessing(ProcessorBase):
    """A processor class that would find a power core in an image. WIP."""
    pass


class TargetProcessing(ProcessorBase):
    """A process class to find a vision target in an image and return relevant location data.

    A class is used so it can be used on the Pi and testing on a
    computer. This class does not contain methods for fetching an image,
    only manipulating it. Really, the only method that should ever be
    called is the process_image method.
    """

    def __init__(self, camera_w, camera_h, draw_img=False):
        """Initialize camera and distrortion matrices and object cordinates."""

        # Create camera matrix
        self.dist_matrix = np.zeros((4, 1))
        focal_length = camera_w  # width of camera
        center = camera_w // 2, camera_h // 2
        self.camera_matrix = np.array(
            [[focal_length, 0, center[0]],
             [0, focal_length, center[1]],
             [0, 0, 1]], dtype='double')

        # Object coordinate setup. Coordinates are labeled starting with
        #  the top-left in a counter-clockwise order.
        self.outer_obj_points = np.array([
            [-19.6, 0, 0],
            [-9.8, -17, 0],
            [9.8, -17, 0],
            [19.6, 0, 0],
        ])
        self.inner_obj_points = np.array([
            [-19.6, 0, 29.25],
            [-9.8, -17, 29.25],
            [9.8, -17, 29.25],
            [19.6, 0, 29.25],
        ])
        self.outer_obj_points_8gon = np.array([
            [-19.6, 0, 0],
            [-9.8, -17, 0],
            [9.8, -17, 0],
            [19.6, 0, 0],
            [17.3, 0, 0],
            [8.6, -15, 0],
            [-8.6, -15, 0],
            [-17.3, 0, 0],
        ])
        self.inner_obj_points_8gon = np.array([
            [-19.6, 0, 29.25],
            [-9.8, -17, 29.25],
            [9.8, -17, 29.25],
            [19.6, 0, 29.25],
            [17.3, 0, 29.25],
            [8.6, -15, 29.25],
            [-8.6, -15, 29.25],
            [-17.3, 0, 29.25],
        ])

        # This should be the distance from the turret, in the same unit
        #  as obj_points(inches). x_offset is positive to the right and
        #  negative to the left. z_offset is negative inches from the
        #  camera to the turret.
        self.x_offset = 0
        self.z_offset = 0

        # Needed if the camera is rotated upside-down or something. It
        #  will rotate the camera x degrees clockwise. I havn't tested
        #  this before, and you may need to change the estimated
        #  parameters of the camera matrix to match the new dimensions.
        self.camera_rotation = 0

        # Wether to return a marked up image. Testing only.
        self.draw_img = draw_img

        # The threshold for checking images.
        self.lower_thresh = 40, 0, 90
        self.upper_thresh = 90, 255, 255
        self.home_lower_thresh = 20, 0, 15
        self.home_upper_thresh = 60, 255, 255
    
        # The minimum alignment angle needed to target the inner port.
        self.minimum_alignment = 20

        # Used when finding the approximate polygon.
        self.epsilon_adjust = 0.05

    def process_image(self, frame):
        """Take an image, isolate the target, and return the calculated values."""

        # This is untested, it's just set to 0 currently.
        frame = cv2.blur(frame, (3, 3))
        img = self.rotate(frame, self.camera_rotation)

        # The image returned by cvsink.getFrame() is already in BGR
        #  format, so the only thing that we need to do convert to HSV
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        thresh = cv2.inRange(img, self.lower_thresh, self.upper_thresh)
        _, cnts, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        # Filter contours
        cnts = self._get_valid_cnts(cnts)
        cnts = sorted(cnts, key=lambda x: cv2.contourArea(x), reverse=True)

        if len(cnts) > 0:
            # If a contour is present, use the largest/closest one
            cnt = cnts[0]

            # Maybe this will stop there being a fifth point in the middle of a straight line
            epsilon = 0.01 * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)

            hull = cv2.convexHull(approx)
            epsilon = self.epsilon_adjust * cv2.arcLength(hull, True)
            hull = cv2.approxPolyDP(hull, epsilon, True)
            points = np.array([
                hull[3][0],
                hull[2][0],
                hull[1][0],
                hull[0][0],
            ], dtype='float')

            _, rvec, tvec = cv2.solvePnP(self.outer_obj_points, points,
                                         self.camera_matrix, self.dist_matrix)
            tvec[2][0] += self.z_offset
            tvec[0][0] += self.x_offset
            distance, horizontal_angle, vertical_angle, alignment_angle = self._process_vecs(tvec, rvec)
            inner = 0  # whether coordinates refer to inner port

            if self.draw_img:
                cv2.drawContours(frame, [cnt], 0, self.BLUE, 1)

                mid = ((hull[0][0][0] + hull[1][0][0] + hull[2][0][0] + hull[3][0][0]) // 4,
                       (hull[0][0][1] + hull[3][0][1]) // 2)
                cv2.circle(frame, mid, 3, self.RED, -1)
                cv2.circle(frame, mid, 10, self.RED, 1)
                cv2.drawContours(frame, [hull], 0, self.RED, 1)
                cv2.drawContours(frame, [approx], 0, self.YELLOW, 2)

                for point in hull:
                    center = point[0][0], point[0][1]
                    cv2.circle(frame, center, 6, self.WHITE, -1)

                # Draw a vertical line down the image
                h, w, _ = frame.shape
                cv2.line(frame, (w // 2, 0), (w // 2, h), self.YELLOW, 1)

                labels = [
                    f'Distance: {round(distance)}"',
                    f'Horizontal: {round(horizontal_angle)}',
                    f'Vertical: {round(vertical_angle)}',
                    f'Alignment: {round(alignment_angle)}',
                ]
                self.label(frame, labels)

            # Return 1(success) and values. Return the frame that may or
            #  may not have been modified.
            result = (1, round(distance), round(horizontal_angle), round(vertical_angle), round(alignment_angle), inner)
            return result, frame

        # If no contours, return all zeros and original frame
        return (0, 0, 0, 0, 0, 0), frame

    def set_threshold(self, h_low, h_up, s_low, s_up, v_low, v_up):
        self.lower_thresh = h_low, s_low, v_low
        self.upper_thresh = h_up, s_up, v_up

    def _process_vecs(self, tvec, rvec):
        """Turn rvec and tvec into distance and angles."""
        x = tvec[0][0]
        y = tvec[1][0]
        z = tvec[2][0]

        # Straight line distance to target in 3 dimensions.
        distance = math.sqrt(x**2 + y**2 + z**2)

        # Angle the turret needs to turn/raise to align with the target.
        horizontal_angle = math.degrees(math.atan2(x, z))
        vertical_angle = math.degrees(math.atan2(y, z))

        # Angle of the robot from the targets perspective; if the robot
        #  is aligned dead-on with the target or lookin at it from an
        #  angle.
        R, _ = cv2.Rodrigues(rvec)
        points_at_origin = np.matmul(-R.T, tvec)
        alignment_angle = math.degrees(math.atan2(points_at_origin[0][0],
                                                  points_at_origin[2][0]))

        return distance, horizontal_angle, vertical_angle, alignment_angle

    def _get_valid_cnts(self, cnts_in):
        """Filter list to return only valid contours."""
        valid = []
        for cnt in cnts_in:
            # Eliminate contours that are too small
            if len(cnt) <= 5 or cv2.contourArea(cnt) < 100:
                continue
            
            # Eliminate contours whose convex hull does not have 4 sides
            hull = cv2.convexHull(cnt)
            epsilon = self.epsilon_adjust * cv2.arcLength(hull, True)
            hull = cv2.approxPolyDP(hull, epsilon, True)
            if len(hull) < 4:
                continue

            valid.append(cnt)
        return valid


# Copy-pasted from the example python program
def parse_error(e):
    """Report parse error."""
    print("config error in '{}': {}".format(config_file, e), file=sys.stderr)


# Copy-pasted from the example python program
def read_camera_config(config):
    """Read a single camera configuration."""
    class CameraConfig:
        pass
    cam = CameraConfig()

    print('now configuring:', config['name'])

    # name
    try:
        cam.name = config['name']
    except KeyError:
        parse_error('could not read camera name')
        return False

    # path
    try:
        cam.path = config['path']
    except KeyError:
        parse_error("camera '{}': could not read path".format(cam.name))
        return False

    # stream properties
    cam.streamConfig = config.get('stream')

    cam.config = config

    camera_configs.append(cam)
    return True


# Copy-pasted from the example python program
def read_config():
    """Read configuration file."""
    global team
    global server

    # parse file
    try:
        with open(config_file, 'rt', encoding='utf-8') as f:
            j = json.load(f)
    except OSError as err:
        print("could not open '{}': {}".format(config_file, err), file=sys.stderr)
        return False

    # top level must be an object
    if not isinstance(j, dict):
        parse_error('must be JSON object')
        return False

    # team number
    try:
        team = j['team']
    except KeyError:
        parse_error('could not read team number')
        return False

    # ntmode
    if 'ntmode' in j:
        string = j['ntmode']
        if string.lower() == 'client':
            server = False
        elif string.lower() == 'server':
            server = True
        else:
            parse_error("could not read ntmode value '{}'".format(string))

    # cameras
    try:
        cameras = j['cameras']
    except KeyError:
        parse_error('could not read cameras')
        return False
    for camera in cameras:
        if not read_camera_config(camera):
            return False

    # switched cameras
    if 'switched cameras' in j:
        for camera in j['switched cameras']:
            if not read_camera_config(camera):
                return False

    print(str(j).encode('utf-8'))

    return True


def camera_listener(fromobj, key, value, isNew):
    """Stream camera i in driver_cameras whenever value is changed."""
    global driver_cameras, mjpeg_server
    if value < len(driver_cameras):
        for _, sink, _ in driver_cameras:
            sink.setEnabled(False)
        cam, sink, _ = driver_cameras[value]
        sink.setEnabled(True)
        mjpeg_server.setSource(cam)


if __name__ == '__main__':
    # Testing only, this will tell the code to draw on the image
    #  received from the image camera, will lower FPS.
    USE_MODIFIED_IMAGE = True
    PAUSE = False
    USE_SMART_DASHBOARD = False

    if len(sys.argv) >= 2:
        config_file = sys.argv[1]

    # Read configuration and quit if error
    if not read_config():
        sys.exit(1)

    # Start NetworkTables
    ntinst = NetworkTablesInstance.getDefault()
    if server:
        print('Setting up NetworkTables server...')
        ntinst.startServer()
    else:
        print('Setting up NetworkTables client for team {}'.format(team))
        ntinst.startClientTeam(team)
    smart_dashboard = ntinst.getTable('SmartDashboard')
    # Add listener to control which camera is streaming
    if USE_SMART_DASHBOARD:
        smart_dashboard.getEntry('Vision/camera_id').addListener(
            camera_listener,
            ntcore.constants.NT_NOTIFY_IMMEDIATE
            | ntcore.constants.NT_NOTIFY_NEW
            | ntcore.constants.NT_NOTIFY_UPDATE
        )

    # Find primary and secondary cameras
    cameras = []
    for config in camera_configs:
        camera = UsbCamera(config.name, config.path)
        camera.setConfigJson(json.dumps(config.config))

        cvsink = CvSink('cvsink: ' + config.name)
        cvsink.setSource(camera)
        cvsink.setEnabled(False)  # disable frame fetching by default

        # Add 3 things to the list of cameras: the camera object itself,
        #  the cvsink object, and the camera configuration dictionary.
        #  The first two all the only ones actually needed. Two stream
        #  the camera you can use in server.setsource(cam), and the
        #  cvsink is used to enable fetching frames. config.config is
        #  only used to access configuration information like the name
        #  and resolution.
        cameras.append((camera, cvsink, config.config))

    vision_camera = None
    driver_cameras = []
    for camera in cameras:
        # The camera wanted for vision must be named 'Vision' exactly,
        #  because that's how I distinguish between them. The driver
        #  cameras can be named anything.
        if camera[2]['name'] == 'Vision':
            vision_camera = camera
        else:
            driver_cameras.append(camera)

    vision_camera_name = vision_camera[2]['name']
    driver_camera_list = ', '.join(
        [config['name'] for _, _, config in driver_cameras])
    print('Vision camera: ' + vision_camera_name)
    print('Driver cameras: ' + driver_camera_list)
    if USE_SMART_DASHBOARD:
        smart_dashboard.putString('Vision/vision_camera', vision_camera_name)
        smart_dashboard.putString('Vision/driver_cameras', driver_camera_list)

    # This is only for debugging, as once I print out a bunch of values
    #  it's hard to see that the initial setup went correctly.
    if PAUSE:
        print('Waiting...')
        time.sleep(5)
        print('Running!')

    # Start mjpeg_server
    port = 1181
    mjpeg_server = MjpegServer('Vision Server', port)
    # If you ever need to standardize resolutions between cameras, you
    #  can use the following. It's bad though, because then it
    #  uncompresses and resises all images, so it decreases FPS.
    # mjpeg_server.setResolution(640, 480)
    # mjpeg_server.setCompression(70)

    # Start camera loop
    driver_camera = driver_cameras[0]
    driver_camera[1].setEnabled(True)
    # mjpeg_server.setSource(driver_camera[0])

    camera, cvsink, config = vision_camera
    cvsink.setEnabled(True)
    # mjpeg_server.setSource(camera)

    # This code creates a cvsink which will push a modified image to the
    #  MJPEG stream. Testing only.
    if USE_MODIFIED_IMAGE:
        name = 'cvsource: ' + vision_camera[2]['name']
        width = vision_camera[2]['height']
        height = vision_camera[2]['width']
        fps = vision_camera[2]['fps']
        cvsource = CvSource(name, VideoMode.PixelFormat.kMJPEG, width, height, fps)
        mjpeg_server.setSource(cvsource)

    processor = TargetProcessing(config['width'], config['height'],
                                 draw_img=USE_MODIFIED_IMAGE)
    img = np.zeros(shape=(config['height'], config['width'], 3), dtype=np.uint8)
    while True:
        timestamp, img = cvsink.grabFrame(img)
        if timestamp == 0:
            error_msg = 'Error: ' + str(cvsink.getError())
            print(error_msg)
            if USE_SMART_DASHBOARD:
                smart_dashboard.putString('Vision/errors', error_msg)
            continue

        # The elapsed time is really only for testing to determine the
        #  optimal image resolution and speed.
        start_time = time.time()
        result, img = processor.process_image(img)
        if USE_MODIFIED_IMAGE:
            cvsource.putFrame(img)
        end_time = time.time()

        result = timestamp, *result, round(end_time - start_time, 5)
        # Result key:
        #  1. timestamp
        #  2. success (0/1)
        #  3. distance
        #  4. horizontal angle
        #  5. vertical angle
        #  6. alignment angle
        #  7. inner port (0/1)
        #  8. elapsed time
        print(result)
        if USE_SMART_DASHBOARD:
            smart_dashboard.putNumberArray('Vision/result', result)
        # Not sure if I need this. It might decrease the lag time of
        #  updating the NetworkTables, but it's untested.
        # ntinst.flush()
