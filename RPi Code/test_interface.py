import numpy as np
import cv2
from vision_processing_2020 import BallProcessing


img = cv2.imread('C:\\workspace\\Vision2020\\Power Cell Images\\frame_00050.png')

processor = BallProcessing()
img = processor.process_image(img)

cv2.imshow('Image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
