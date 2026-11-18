import cv2
import numpy as np
import glob
import os

PATTERN_SIZE = (9, 6)
SQUARE_SIZE_MM = 23.33

IMAGES_DIR = os.path.join(os.path.dirname(__file__), "..", "media", "calibration")
SAMPLE_IMAGE_NAME = "calibration_image_18.png"
OUTPUT_YAML = os.path.join(os.path.dirname(__file__), "camera_parameters.yaml")
DISTORTED_OUT = os.path.join(os.path.dirname(__file__), "..", "media", "calibration", "distorted_image.png")
RECTIFIED_OUT = os.path.join(os.path.dirname(__file__), "..", "media", "calibration", "rectified_image.png")

CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

def main():

    objp = np.zeros((PATTERN_SIZE[0] * PATTERN_SIZE[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:PATTERN_SIZE[0], 0:PATTERN_SIZE[1]].T.reshape(-1, 2)
    objp *= SQUARE_SIZE_MM

    objpoints = []
    imgpoints = []

    images = sorted(glob.glob(os.path.join(IMAGES_DIR, "calibration_image_*.png")))
    if not images:
        raise RuntimeError(f"Keine Kalibrierbilder in {IMAGES_DIR} gefunden.")

    img_shape = None
    used = 0
    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_shape = gray.shape[::-1]

        found, corners = cv2.findChessboardCorners(gray, PATTERN_SIZE, None)
        if found:
            corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), CRITERIA)
            objpoints.append(objp)
            imgpoints.append(corners_refined)
            used += 1
        else:
            print(f"[WARN] Schachbrett nicht gefunden in: {os.path.basename(fname)}")

    print(f"{used}/{len(images)} Bilder erfolgreich fuer Kalibrierung verwendet.")
    if used < 2:
        raise RuntimeError("Zu wenige gueltige Kalibrierbilder gefunden.")

    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, img_shape, None, None
    )

    total_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], camera_matrix, dist_coeffs)
        imgpoints_i = np.asarray(imgpoints[i], dtype=np.float32).reshape(-1, 1, 2)
        imgpoints2 = np.asarray(imgpoints2, dtype=np.float32).reshape(-1, 1, 2)
        error = cv2.norm(imgpoints_i, imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        total_error += error
    mean_error = total_error / len(objpoints)
    print(f"Mittlerer Reprojection Error: {mean_error:.4f} px")

    fs = cv2.FileStorage(OUTPUT_YAML, cv2.FILE_STORAGE_WRITE)
    fs.write("camera_matrix", camera_matrix)
    fs.write("distortion_coefficients", dist_coeffs)
    fs.write("reprojection_error", mean_error)
    fs.write("pattern_size", np.array(PATTERN_SIZE))
    fs.write("square_size_mm", SQUARE_SIZE_MM)
    fs.release()
    print(f"Kameraparameter gespeichert in {OUTPUT_YAML}")

    sample_img_path = os.path.join(IMAGES_DIR, SAMPLE_IMAGE_NAME)
    if not os.path.isfile(sample_img_path):
        print(f"[WARN] {SAMPLE_IMAGE_NAME} nicht gefunden, verwende erstes Bild stattdessen.")
        sample_img_path = images[0]
    sample_img = cv2.imread(sample_img_path)
    undistorted = cv2.undistort(sample_img, camera_matrix, dist_coeffs)

    cv2.imwrite(DISTORTED_OUT, sample_img)
    cv2.imwrite(RECTIFIED_OUT, undistorted)
    print(f"Verzerrtes Bild: {DISTORTED_OUT}")
    print(f"Entzerrtes Bild: {RECTIFIED_OUT}")

if __name__ == "__main__":
    main()
