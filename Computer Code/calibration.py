import numpy as np
import cv2 as cv
import os

# Calibration images from:
#  https://markhedleyjones.com/storage/checkerboards/Checkerboard-A4-25mm-8x6.pdf

def calibrate(path_to_images, dimensions):
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    x, y = dimensions
    images = os.listdir(path_to_images)
    images = [path_to_images + '\\' + path for path in images
              if path.endswith('.png') or path.endswith('.jpg')]

    objp = np.zeros((x*y, 3), np.float32)
    objp[:, :2] = np.mgrid[0:y, 0:x].T.reshape(-1, 2)
    objpoints = []
    imgpoints = []

    for fname in images:
        img = cv.imread(fname)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        ret, corners = cv.findChessboardCorners(gray, (y, x), None)

        if ret:
            objpoints.append(objp)
            corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)

    ret, M, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    return M, dist


if __name__ == '__main__':
    path_to_calibration_images = ''
    chessboard_dimensions = (6, 8)  # number of inner corners
    save_to_file = False

    M, dist = calibrate(path_to_calibration_images, chessboard_dimensions)
    if save_to_file:
        np.savetxt('camera_matrix.txt', M)
        np.savetxt('dist_matrix.txt', dist)

    print('Calibration successful!')
    print('\nCamera Matrix:')
    print(M)
    print('\nDistortion Matrix:')
    print(dist)
