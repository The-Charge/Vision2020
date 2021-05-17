import json
import traceback

import numpy as np

from cscore import UsbCamera, MjpegServer, CvSink, CvSource, VideoMode

from vision import *


VISION_CAMERA_NAME = 'Primary'
VISION_PORT = 1182


if __name__ == '__main__':
    cameras, stream_configs = configure()

    # Detect vision camera
    vision_camera, vision_stream_config = None, None
    for camera, stream_config in zip(cameras, stream_configs):
        if camera.getName().lower().strip() == VISION_CAMERA_NAME.lower().strip():
            vision_camera = camera
            vision_stream_config = stream_config
    if vision_camera is None:
        raise RuntimeError(f'no vision camera detected; must be named \'{VISION_CAMERA_NAME}\'')

    # Configure vision sink
    vision_sink = CvSink(f'{vision_camera.getName()} CVSink')
    vision_sink.setSource(vision_camera)
    vision_sink.setEnabled(True)

    # Get basic properties
    properties = json.loads(vision_camera.getConfigJson())
    width = properties['width']
    height = properties['height']
    fps = properties['fps']

    # Configure vision source
    vision_source = CvSource(f'{vision_camera.getName()} CVSource', VideoMode.PixelFormat.kMJPEG, width, height, fps)
    server = MjpegServer(f'{vision_camera.getName()} Stream', VISION_PORT)
    server.setSource(vision_source)

    # Create the processor
    processor = ExampleProcessor((width, height))

    frame = np.zeros(shape=(height, width, 3), dtype=np.uint8)
    while True:
        timestamp, frame = vision_sink.grabFrame(frame)
        if timestamp == 0:
            print(f'Error: {vision_sink.getError()}')
            continue

        try:
            frame = processor.process_image(frame)
        except:
            print(traceback.format_exc())
            continue

        vision_source.putFrame(frame)
