import cv2
import numpy as np
import glob
import os

# Create output folders if they don't exist
calibrated_output_folder = './calibrated_images/'
undistorted_output_folder = './undistorted_images/'

if not os.path.exists(calibrated_output_folder):
    os.makedirs(calibrated_output_folder)

if not os.path.exists(undistorted_output_folder):
    os.makedirs(undistorted_output_folder)

x_y = (6, 8)
# Vector for 3D points
threedpoints = []

# Vector for 2D points
twodpoints = []

objectp3d = np.zeros((1, x_y[0] * x_y[1], 3), np.float32)
objectp3d[-1][:, :2] = np.mgrid[0:x_y[0], 0:x_y[1]].T.reshape(-1, 2)

prev_img_shape = None

cv_img = []
graylist = []
for img in glob.glob('./imge' + '*.jpg'):
    nimg = cv2.imread(img)
    cv_img.append(nimg)
    gray = cv2.cvtColor(nimg, cv2.COLOR_BGR2GRAY)
    graylist.append(gray)
    ret, corners = cv2.findChessboardCorners(gray, x_y, None)

    if ret == True:
        threedpoints.append(objectp3d)
        twodpoints.append(corners)

        cv2.drawChessboardCorners(nimg, x_y, corners, ret)
        cv2.imshow('corners', nimg)
        cv2.waitKey(200)

cv2.destroyAllWindows()

# Calibration
ret, matrix, distortion, r_vecs, t_vecs = cv2.calibrateCamera(
    threedpoints, twodpoints, graylist[0].shape[::-1], None, None
)

print("Camera calibration:", ret)
print("Camera matrix:", matrix)
print("Distortion:", distortion)
print("Rotation vectors:", r_vecs)
print("Translation vectors:", t_vecs)

# Undistortion
#newcameramtx, roi = cv2.getOptimalNewCameraMatrix(matrix, distortion, graylist[0].shape[::-1], 1, graylist[0].shape[::-1])

# Loop over images, calibrate and undistort
for img_path, cv_img_gray in zip(glob.glob('./imge' + '*.jpg'), graylist):
    img = cv2.imread(img_path)
    undistorted_img = cv2.undistort(cv_img_gray, matrix, distortion, None, newcameramtx)

    # Save the calibrated image
    calibrated_output_path = calibrated_output_folder + 'calibrated_' + img_path.split('/')[-1]
    cv2.imwrite(calibrated_output_path, img)

    # Save the undistorted image
    output_file = os.path.join(os.path.join('./imge/', 'undistorted_images'), 'undistorted_' + os.path.basename(img_path))
    cv2.imwrite(output_file, undistorted_img)

