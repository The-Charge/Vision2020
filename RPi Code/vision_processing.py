#!/usr/bin/env python3

import json
import time
import sys
import math

from cscore import UsbCamera, MjpegServer, CvSink, CvSource, VideoMode
from networktables import NetworkTablesInstance
import ntcore
import numpy as np
import cv2 as cv

config_file = '/boot/frc.json'

team = None
server = None
camera_configs = []
cameras = []


class TargetProcessing:
    """A class that stores all method to manipulate an image and run vision calculations.

    A class is used so it can be used on the Pi and testing on a
    computer. This class does not contain methods for fetching an image,
    only manipulating it. Really, the only method that should ever be
    called is the process_image method.
    """

    def __init__(self):
        """Initialize camera and distrortion matrices and object cordinates."""

        # Here's the deal with camera matrices: the official tutorial
        #  says it's necessary, but the other tutorial I used doesn't.
        #  It works if I just estimate the values, so that's what I've
        #  done. If we ever change to a camera with lots of distortion
        #  that may need to change. I think one of the issues may have
        #  been that the camera auto-focuses, which I think would change
        #  the camera matrix. If you do need to calibrate it, there's a
        #  calibration script included in the repository. Also, the
        #  camera matrices for the camera's I've been using are below.
        # Link to the tutorial that gave the estimation technique:
        #  https://www.learnopencv.com/head-pose-estimation-using-opencv-and-dlib/

        # For the old Microsoft testing camera:
        # self.camera_matrix = np.array(
        #     [[6.823213768214409356e+02, 0, 3.437010372208300737e+02],
        #      [0, 6.825785584524919614e+02, 2.349732921104409229e+02],
        #      [0, 0, 1]])
        # self.dist_matrix = np.array([
        #     1.438497706769043816e-01,
        #     -1.277730422370438879e+00,
        #     3.416375616850637984e-03,
        #     6.878034387843228554e-04,
        #     2.763647799620323475e+00])

        # For the new Logitech camera:
        # self.camera_matrix = np.array(
        #     [[1.381995930279219465e+03, 0, 9.869331650977278514e+02],
        #      [0, 1.384248229780748943e+03, 5.087657534211837174e+02],
        #      [0, 0, 1]])
        # self.dist_matrix = np.array([
        #     -4.643519073137976068e-04,
        #     3.765194233053799633e-01,
        #     -9.895872556314310592e-03,
        #     7.940727298754526181e-03,
        #     -7.882434187646156776e-01])

        self.dist_matrix = np.zeros((4, 1))
        focal_length = 800  # width of camera
        center = 400, 300  # (width / 2, height / 2)
        self.camera_matrix = np.array(
            [[focal_length, 0, center[0]],
             [0, focal_length, center[1]],
             [0, 0, 1]], dtype='double')

        # Object coordinate setup
        # The way I've done this is by taking a matrix with the exact
        #  values of a single piece of tape at origin, and then rotated
        #  and translated it to get the exact values.
        # For reference the targets are...
        #  - 5.5 x 2 inches
        #  - Rotated by ~14.5 degrees
        #  -  7 inches apart at their closests point
        #     (Should be 8 inches, but I messed up when I made them)
        tape_cords = np.array([[-1, 2.75,  0],
                               [-1, -2.75, 0],
                               [1,  -2.75, 0],
                               [1,  2.75,  0]])
        # Rotate matrix about origin
        left = tape_cords.dot(self.create_rotation_matrix(15))
        right = tape_cords.dot(self.create_rotation_matrix(-15))
        # In the left matrix, the last value is the offset from origin.
        #  we want the matrix to be centered at origin, so if we
        #  translate it over using that value it places it at origin.
        #  once it's at origin, we need to move it 3.5 inches in a
        #  direction to get the distance between them.
        origin_offset = left[3][0]
        translation = np.array([[3.5 + origin_offset, 0, 0],
                                [3.5 + origin_offset, 0, 0],
                                [3.5 + origin_offset, 0, 0],
                                [3.5 + origin_offset, 0, 0]])
        left -= translation
        right += translation
        self.obj_points = np.append(left, right, axis=0)

        # This should be the distance from the center of the robot, in
        #  the same unit as the object cordinates are(inches). x_offset
        #  should be positive for distance to the right from the center
        #  and negative for distance to the left from center. z_offset
        #  should be negative inches from the front bumper of the robot.
        self.x_offset = 0
        self.z_offset = 0
        # Needed if the camera is rotated upside-down or something. It
        #  will rotate the camera x degrees clockwise. I havn't tested
        #  this before, and you may need to change the estimated
        #  parameters of the camera matrix to match the new dimensions.
        self.camera_rotation = 0
        self.draw_img = True

    def draw_labels(self, img, labels, width=175):
        """Draw a black box with text in the top-left on an image."""
        font = cv.FONT_HERSHEY_SIMPLEX
        y = 20
        cv.rectangle(img, (-10, -10), (width, len(labels)*20 + 10),
                     self.colors['BLACK'], -1)
        cv.rectangle(img, (-10, -10), (width, len(labels)*20 + 10),
                     self.colors['WHITE'], 1)

        for label in labels:
            if type(label) is not tuple and type(label) is not list:
                label = [label]
            if len(label) == 1:
                text = label[0]
            if len(label) == 2:
                text = '{}: {}'.format(label[0], round(label[1]))
            if len(label) == 3:
                text = '{}: {}'.format(label[0], round(label[1], label[2]))
            cv.putText(img, text, (10, y), font, .5,
                       self.colors['WHITE'], 1, cv.LINE_AA)
            y += 20

    def create_rotation_matrix(self, angle):
        """Turn an angle in degrees to a 3D rotation matrix about the z-axis."""
        angle = math.radians(angle)
        array = np.array([[1, 0, 0],
                          [0, math.cos(angle), -math.sin(angle)],
                          [0, math.sin(angle), math.cos(angle)]])
        array = np.array([[math.cos(angle), -math.sin(angle), 0],
                          [math.sin(angle), math.cos(angle), 0],
                          [0, 0, 1]])
        return array

    def process_image(self, frame):
        """Take an image, isolate the target, and return the calculated values."""

        # I don't know if this will work, but we'll find out if we ever
        #  rotate the camera. (right now it's set to 0 degrees so it
        #  won't actually do anything)
        img = self.rotate(frame, self.camera_rotation)

        # The image returned by cvsink.getFrame() is already in BGR
        #  format, so the only thing that we need to do convert to HSV
        img = cv.cvtColor(img, cv.COLOR_BGR2HSV)
        lower_thresh = 40, 0, 90
        upper_thresh = 90, 255, 255
        thresh = cv.inRange(img, lower_thresh, upper_thresh)
        _, cnts, _ = cv.findContours(thresh, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)

        # Filter contours
        cnts = self.get_valid_cnts(cnts)
        cnts = self.get_cnt_info(cnts)
        cnts = sorted(cnts, key=lambda x: x['center'][0])
        pairs = self.find_pairs(cnts)
        # If there is more than one set, sort the largest to the front
        pairs = sorted(pairs, key=self.get_bounding_box_area, reverse=True)

        if len(pairs) > 0:
            # If a pair is present, default to the largest/closet pair
            pair = pairs[0]
            left = pair[0]
            right = pair[1]

            rect_l = left['rect']
            rect_r = right['rect']
            points = [rect_l[2], rect_l[1], rect_l[0], rect_l[3],
                      rect_r[1], rect_r[0], rect_r[3], rect_r[2]]
            points = np.array(points, dtype='float32')

            _, rvec, tvec = cv.solvePnP(self.obj_points, points,
                                        self.camera_matrix, self.dist_matrix)
            tvec[2][0] += self.z_offset
            tvec[0][0] += self.x_offset
            distance, angle1, angle2 = self.process_vecs(tvec, rvec)

            # Draw the bounding box, center, rectangles, and midline on
            #  an image. For testing use only.
            if self.draw_img:
                cv.drawContours(frame, [left['cnt'], right['cnt']], -1, (0, 255, 0))
                cv.polylines(frame, [left['rect'], right['rect']], True, (0, 0, 255))
                combined = np.vstack([left['rect'], right['rect']])
                bounding_rect = cv.minAreaRect(combined)
                bounding_rect = cv.boxPoints(bounding_rect)
                bounding_rect = np.int32(bounding_rect)
                cv.polylines(frame, [bounding_rect], True, (255, 0, 0))

                mid = ((left['rect'][0][0] + left['rect'][1][0] + left['rect'][2][0] + left['rect'][3][0] + right['rect'][0][0] + right['rect'][1][0] + right['rect'][2][0] + right['rect'][3][0]) // 8,
                       (left['rect'][0][1] + left['rect'][1][1] + left['rect'][2][1] + left['rect'][3][1] + right['rect'][0][1] + right['rect'][1][1] + right['rect'][2][1] + right['rect'][3][1]) // 8)
                cv.circle(frame, mid, 3, (255, 0, 0), -1)
                cv.circle(frame, mid, 10, (255, 0, 0), 1)

                img_height, img_width, _ = frame.shape
                cv.line(frame, (img_width // 2, 0), (img_width // 2, img_height), (0, 255, 255), 1)
            # Return 1(success) and values. Return the frame that may or
            #  may not have been modified.
            return (1, round(distance), round(angle1), round(angle2)), frame
        # If no contours, return all zeros and original frame
        return (0, 0, 0, 0), frame

    def process_vecs(self, tvec, rvec):
        """Turn rvec and tvec into distance and angles."""
        x = tvec[0][0]
        z = tvec[2][0]

        distance = math.sqrt(x**2 + z**2)

        angle1 = math.degrees(math.atan2(x, z))

        R, _ = cv.Rodrigues(rvec)
        points_at_origin = np.matmul(-R.T, tvec)
        angle2 = math.degrees(math.atan2(points_at_origin[0][0],
                                         points_at_origin[2][0]))

        return distance, angle1, angle2

    def find_pairs(self, cnts):
        """Filter a list of contours into pairs."""
        pairs = []
        for i in range(len(cnts) - 1):
            left = cnts[i]
            right = cnts[i+1]
            if left['angle'] > 0 and right['angle'] < 0:
                pairs.append((left, right))
        return pairs

    def get_cnt_info(self, cnts):
        """Turn list of contours into list of tuples with information about the contours."""
        cnt_info = []
        for cnt in cnts:
            _, _, angle = cv.fitEllipse(cnt)
            angle = angle - 180 if angle > 90 else angle

            rect = cv.minAreaRect(cnt)
            rect = cv.boxPoints(rect)
            rect = np.int32(rect)

            M = cv.moments(cnt)
            if M['m00'] != 0:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
            else:
                cx, cy = 0, 0

            info = {
                'cnt': cnt,
                'angle': angle,
                'rect': rect,
                'center': (cx, cy),
            }
            cnt_info.append(info)
        return cnt_info

    def get_valid_cnts(self, cnts_in):
        """Filter list to return only valid contours."""
        valid = []
        for cnt in cnts_in:
            # Eliminate contours that are too small
            if len(cnt) <= 5 or cv.contourArea(cnt) < 100:
                continue

            # Get approximate sides in contour shape, and only run if 4
            peri = cv.arcLength(cnt, True)
            approx = cv.approxPolyDP(cnt, 0.04 * peri, True)
            sides = len(approx)
            if sides != 4:
                continue

            # Get the similarity between the area of the bounding
            #  rectangle and the area of the contour, throw it out if
            #  it's less than 80%
            rect = cv.minAreaRect(cnt)
            rect = cv.boxPoints(rect)
            rect_area = self.get_rect_area(rect)
            cnt_area = cv.contourArea(cnt)
            similarity = min(rect_area, cnt_area) / max(rect_area, cnt_area)
            if similarity < .70:
                continue

            valid.append(cnt)
        return valid

    def get_bounding_box_area(self, pair):
        """Return the area of the bounding box of a contour pair."""
        left, right = pair
        combined = np.vstack([left['rect'], right['rect']])
        bounding_rect = cv.minAreaRect(combined)
        bounding_rect = cv.boxPoints(bounding_rect)
        area = self.get_rect_area(bounding_rect)
        return area

    def get_rect_area(self, rect):
        """Get area of rectangle."""
        d1 = math.sqrt((rect[0][0] - rect[1][0])**2
                       + (rect[0][1] - rect[1][1])**2)
        d2 = math.sqrt((rect[1][0] - rect[2][0])**2
                       + (rect[1][1] - rect[2][1])**2)
        return d1 * d2

    def rotate(self, img, angle):
        """Takes an image and rotates it clockwise."""
        if angle == 0:
            return img

        # I took the code from this tutorial:
        #  https://www.pyimagesearch.com/2017/01/02/rotate-images-correctly-with-opencv-and-python/
        h, w = img.shape[:2]
        cX, cY = w // 2, h // 2

        R = cv.getRotationMatrix2D((cX, cY), -angle, 1)
        cos = np.abs(R[0, 0])
        sin = np.abs(R[0, 1])

        nW = int((h * sin) + (w * cos))
        nH = int((h * cos) + (w * sin))

        R[0, 2] += (nW / 2) - cX
        R[1, 2] += (nH / 2) - cY

        return cv.warpAffine(img, R, (nW, nH))


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
    use_modified_image = True

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
    # smart_dashboard.getEntry('Vision/camera_id').addListener(
    #     camera_listener,
    #     ntcore.constants.NT_NOTIFY_IMMEDIATE
    #     | ntcore.constants.NT_NOTIFY_NEW
    #     | ntcore.constants.NT_NOTIFY_UPDATE
    # )

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
    # smart_dashboard.putString('Vision/vision_camera', vision_camera_name)
    # smart_dashboard.putString('Vision/driver_cameras', driver_camera_list)

    # This is only for debugging, as once I print out a bunch of values
    #  it's hard to see that the initial setup went correctly.
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
    mjpeg_server.setSource(camera)

    # This code creates a cvsink which will push a modified image to the
    #  MJPEG stream. Testing only.
    if use_modified_image:
        name = 'cvsource: ' + vision_camera[2]['name']
        width = vision_camera[2]['height']
        height = vision_camera[2]['width']
        fps = vision_camera[2]['fps']
        cvsource = CvSource(name, VideoMode.PixelFormat.kMJPEG, width, height, fps)
        mjpeg_server.setSource(cvsource)

    processor = TargetProcessing()
    # Allways preallocate, it runs a lot faster
    img = np.zeros(shape=(config['height'], config['width'], 3), dtype=np.uint8)
    while True:
        timestamp, img = cvsink.grabFrame(img)
        if timestamp == 0:
            print('Error:', cvsink.getError())
            continue

        # The elapsed time is really only for testing to determine the
        #  optimal image resolution and speed.
        start_time = time.time()
        result, img = processor.process_image(img)
        if use_modified_image:
            cvsource.putFrame(img)
        end_time = time.time()
        result = timestamp, *result, round(end_time - start_time, 5)
        # Result key:
        #  timestamp, success, distance, angle1, angle2, elapsedtime
        print(result)
        # smart_dashboard.putNumberArray('Vision/result', result)
        # ntinst.flush()
