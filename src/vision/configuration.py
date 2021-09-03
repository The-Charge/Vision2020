import json

from cscore import CameraServer, VideoSource, UsbCamera, MjpegServer
from networktables import NetworkTablesInstance

from . import get_ip

#   JSON format:
#   {
#       "team": <team number>,
#       "ntmode": <"client" or "server", "client" if unspecified>
#       "cameras": [
#           {
#               "name": <camera name>
#               "path": <path, e.g. "/dev/video0">
#               "pixel format": <"MJPEG", "YUYV", etc>   // optional
#               "width": <video mode width>              // optional
#               "height": <video mode height>            // optional
#               "fps": <video mode fps>                  // optional
#               "brightness": <percentage brightness>    // optional
#               "white balance": <"auto", "hold", value> // optional
#               "exposure": <"auto", "hold", value>      // optional
#               "properties": [                          // optional
#                   {
#                       "name": <property name>
#                       "value": <property value>
#                   }
#               ],
#               "stream": {                              // optional
#                   "properties": [
#                       {
#                           "name": <stream property name>
#                           "value": <stream property value>
#                       }
#                   ]
#               }
#           }
#       ]
#       "switched cameras": [
#           {
#               "name": <virtual camera name>
#               "key": <network table key used for selection>
#               // if NT value is a string, it's treated as a name
#               // if NT value is a double, it's treated as an integer index
#           }
#       ]
#   }


CONFIG_FILE = '/boot/frc.json'


def camera_from_config(config):
    """Create a camera from JSON configuration.

    Returns
        (USBCamera object,
        JSON stream config)
    """

    name = config.get('name', 'Unnamed')
    if name == 'Unnamed':
        print('Unnamed camera')

    path = config.get('path')
    if path is None:
        raise AttributeError(f'camera {name} has no path')

    camera = UsbCamera(name, path)
    camera.setConfigJson(json.dumps(config))
    camera.setConnectionStrategy(VideoSource.ConnectionStrategy.kKeepOpen)

    return camera, config.get('stream')


def start_camera(camera, stream_config=None):
    """Start an automatic camera capture and return the server."""
    print(f'Starting camera{camera.getInfo().name, camera.getInfo().path}')
    server = CameraServer.getInstance().startAutomaticCapture(camera=camera, return_server=True)
    if stream_config is not None:
        server.setConfigJson(json.dumps(stream_config))
    return server


def read_config():
    """Read the JSON configuration file and start NetworkTables and create
    USBCameras."""

    # Read JSON config file
    with open(CONFIG_FILE, 'rt', encoding='utf-8') as f:
        data = json.load(f)

    # Read team number
    team = data.get('team')
    if team is None:
        raise AttributeError('could not read team number')

    # Read NetworkTables mode
    mode = data.get('ntmode', 'client')
    if mode == 'server':
        server = True
    elif mode == 'client':
        server = False
    else:
        raise AttributeError(f'invalid ntmode \'{mode}\'')

    # Read and create cameras
    camera_configs = data.get('cameras', None)
    if camera_configs is None:
        print('zero cameras connected')
        cameras = []
        stream_configs = []
    else:
        cameras = [camera_from_config(config) for config in camera_configs]
        cameras, stream_configs = zip(*cameras)

    # Currently does not use switched cameras
    if 'switched cameras' in data:
        print('switched cameras currently not implemented')

    return team, server, cameras, stream_configs


def configure():
    team, server, cameras, stream_configs = read_config()

    for camera, config in zip(cameras, stream_configs):
        start_camera(camera, config)

    instance = NetworkTablesInstance.getDefault()
    if server:
        print(f'Setting up NetworkTables server on IP {get_ip()}')
        instance.startServer()
    else:
        print(f'Setting up NetworkTables client for team {team}')
        instance.startClientTeam(team)
        instance.startDSClient()

    return cameras, stream_configs
