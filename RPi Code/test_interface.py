import numpy as np
import cv2
from vision_processing_2020 import BallProcessing


def run_with_image(num):
    img = cv2.imread(f'C:\\workspace\\Vision2020\\Power Cell Images\\frame_{num}.png')
    height, width = img.shape[:2]
    processor = BallProcessing(width, height)
    _, img = processor.process_image(img)
    cv2.imshow('Image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def run_with_video(num):
    processor = BallProcessing()
    cap = cv2.VideoCapture(f'C:\\workspace\\Vision2020\\Power Cell Images\\IMG_{num}.mov')
    if cap.isOpened() is False:
        print('Error opening video stream file')
    else:
        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        scale = width / 500
        width = int(width / scale)
        height = int(height / scale)

        while cap.isOpened():
            ret, frame = cap.read()
            if ret:
                frame = cv2.resize(frame, (width, height))
                frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
                _, frame = processor.process_image(frame)

                cv2.imshow('Video', frame)
                if cv2.waitKey(25) & 0xFF == ord('q'):
                    break
            else:
                break

        cap.release()
        cv2.destroyAllWindows()


run_with_image('00050')
# run_with_video('5602')
