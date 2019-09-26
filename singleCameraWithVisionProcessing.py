#!/usr/bin/env python3

import json
import time
import sys

from cscore import CameraServer, VideoSource, UsbCamera, MjpegServer, CvSink, VideoMode, CvSource
from networktables import NetworkTablesInstance
import ntcore
import numpy as np
import cv2 as cv

config_file = '/boot/frc.json'

team = None
server = None
camera_configs = []
cameras = []


def parse_error(e):
    """Report parse error."""
    print("config error in '{}': {}".format(config_file, e), file=sys.stderr)


def read_camera_config(config):
    """Read a single camera configuration."""
    class CameraConfig:
        pass
    cam = CameraConfig()

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

    # ntmode (optional) NEED TO LOOK AT THIS
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

    return True


def start_camera_with_compression(config, show_original=False):
    # Configure USB camera and CvSink
    camera = UsbCamera(config.name, config.path)
    camera.setConfigJson(json.dumps(config.config))
    cvsink = CvSink('cvsink')
    cvsink.setSource(camera)

    # Grab camera settings and create compression settings
    camera_height = config.config['height']
    camera_width = config.config['width']
    camera_fps = config.config['fps']
    compressed_height = 120
    compressed_width = 160

    # Configure CvSource
    cvSource = CvSource('cvsource', VideoMode.PixelFormat.kMJPEG,
                        compressed_width, compressed_height, camera_fps)

    # Create server to show original image
    if show_original:
        in_server_location = 1182
        in_server = MjpegServer('In Server', in_server_location)
        in_server.setSource(camera)
        print('in server at http://frcvision.local:{}'.format(in_server_location))

    # Create server to show modified image
    out_server_location = 1181
    out_server = MjpegServer('Out Server', out_server_location)
    out_server.setSource(cvSource)
    print('out server at http://frcvision.local:{}'.format(out_server_location))

    img = np.zeros(shape=(camera_height, camera_width, 3), dtype=np.uint8)
    compressed = np.zeros(shape=(compressed_height, compressed_width, 3), dtype=np.uint8)
    while True:
        t, img = cvsink.grabFrame(img)
        if t == 0:
            print('error:', cvsink.getError())
            continue

        # Vision code

        l = [x**2 for x in range(10000)]
        compressed = cv.resize(img, (compressed_width, compressed_height))
        cvSource.putFrame(compressed)


if __name__ == '__main__':
    if len(sys.argv) >= 2:
        config_file = sys.argv[1]

    # read configuration
    if not read_config():
        sys.exit(1)

    # start NetworkTables
    ntinst = NetworkTablesInstance.getDefault()
    if server:
        print('Setting up NetworkTables server')
        ntinst.startServer()
    else:
        print('Setting up NetworkTables client for team {}'.format(team))
        ntinst.startClientTeam(team)

    # start cameras
    camera = camera_configs[0]
    start_camera_with_compression(camera, show_original=True)
