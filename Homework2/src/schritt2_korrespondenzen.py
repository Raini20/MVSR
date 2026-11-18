"""
MVSR Hausübung 2 – Schritt 2: Geeignete Korrespondenzen zwischen Referenzbild und Kamerabild wählen

Baut auf Schritt 1 auf: verwendet die gespeicherten Referenzdaten
(data/reference_features.yaml) und dieselben SIFT-Parameter (create_sift).

Ablauf (entspricht Punkt 2 der Angabe):
  2.1  Referenzdaten aus Schritt 1 laden
  2.2  Kameraparameter bestimmen (Kalibrierung mit dem Flyer) und speichern
  2.3  Einige Kamerabilder aus dem Video lesen
  2.4  SIFT-Merkmale im Kamerabild extrahieren (gleiche Parameter wie Schritt 1)
  2.5  Deskriptoren mit dem Brute-Force-Matcher vergleichen
  2.6  Schlechte / mehrdeutige Matches mit dem Ratio-Test verwerfen
  2.7  Akzeptierte Korrespondenzen visualisieren

Aufruf (im Ordner src/):
  python schritt2_korrespondenzen.py
"""

from pathlib import Path

import cv2
import numpy as np

from schritt1_referenzmerkmale import REFERENCE_IMAGE, REFERENCE_YAML, create_sift

# ---------------------------------------------------------------------------
# Einstellungen
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent

CAMERA_FILE = ROOT / "config" / "camera_parameters.yaml"
VIDEO_FILE = ROOT / "media" / "video" / "camera_video.mp4"
RESULTS_DIR = ROOT / "media" / "results" / "schritt2"

# Frames aus dem Video, an denen das Matching gezeigt wird (unterschiedliche Ansichten)
TEST_FRAMES = [100, 600, 1000]

# Ratio-Test nach Lowe: bestes Match muss deutlich besser sein als das zweitbeste
RATIO_THRESHOLD = 0.75

# Kalibrierung mit dem Flyer
CALIBRATION_FRAME_STEP = 25      # jeden 25. Frame verwenden (ca. 50 Ansichten)
CALIBRATION_MIN_POINTS = 25      # Frames mit weniger guten Punkten werden ausgelassen


# ---------------------------------------------------------------------------
# Funktionen
# ---------------------------------------------------------------------------
def load_reference():
    """Lädt die in Schritt 1 gespeicherten Referenzdaten."""
    fs = cv2.FileStorage(str(REFERENCE_YAML), cv2.FILE_STORAGE_READ)
    assert fs.isOpened(), "Referenzdaten fehlen -> zuerst schritt1_referenzmerkmale.py ausführen"
    keypoint_data = fs.getNode("keypoints_u_v_size_angle_response_octave").mat()
    reference = {
        "keypoints": [cv2.KeyPoint(float(u), float(v), float(size), float(angle), float(response), int(octave))
                      for u, v, size, angle, response, octave in keypoint_data],
        "descriptors": fs.getNode("descriptors").mat().astype(np.float32),
        "object_points": fs.getNode("object_points_mm").mat(),
        "width_mm": fs.getNode("object_width_mm").real(),
        "height_mm": fs.getNode("object_height_mm").real(),
        "image": cv2.imread(str(REFERENCE_IMAGE)),
    }
    fs.release()
    return reference


def read_frame(index):
    """Liest einen einzelnen Frame aus dem Video."""
    video = cv2.VideoCapture(str(VIDEO_FILE))
    video.set(cv2.CAP_PROP_POS_FRAMES, index)
    ok, frame = video.read()
    video.release()
    assert ok, f"Frame {index} konnte nicht gelesen werden"
    return frame


def match_features(reference_descriptors, frame_descriptors):
    """Brute-Force-Matching mit Ratio-Test.

    Für jedes Referenzmerkmal werden die zwei ähnlichsten Merkmale im
    Kamerabild gesucht (k = 2). Das Match wird nur akzeptiert, wenn das beste
    deutlich besser ist als das zweitbeste: d1 < RATIO_THRESHOLD * d2.
    Rückgabe: (akzeptierte Matches, alle besten Matches ohne Filter)
    """
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    knn_matches = matcher.knnMatch(reference_descriptors, frame_descriptors, k=2)

    all_matches = []
    good_matches = []
    for pair in knn_matches:
        if len(pair) < 2:
            continue
        best, second = pair
        all_matches.append(best)
        if best.distance < RATIO_THRESHOLD * second.distance:
            good_matches.append(best)

    return good_matches, all_matches


def calibrate_camera_with_object(reference, sift):
    """Kamerakalibrierung nach Zhang – wie in HÜ 1, aber mit dem Flyer statt Schachbrett.

    Warum: Das Video wurde mit einer anderen Optik bzw. einem anderen
    Bildausschnitt aufgenommen als die Kalibrierbilder aus HÜ 1, daher passen
    die Parameter aus HÜ 1 nicht (siehe README).

    Der Flyer ist ebenso ein ebenes Objekt mit bekannten 3D-Punkten (Z = 0)
    wie ein Schachbrett. Für viele Frames werden die Matches bestimmt; falsche
    Matches werden mit einer Homographie + RANSAC entfernt (für eine Ebene
    gilt Bild = H * Objekt). Mit den restlichen Punkten wird calibrateCamera
    aufgerufen.
    """
    video = cv2.VideoCapture(str(VIDEO_FILE))
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    image_size = (int(video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    all_object_points = []
    all_image_points = []
    for index in range(0, frame_count, CALIBRATION_FRAME_STEP):
        video.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = video.read()
        if not ok:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = sift.detectAndCompute(gray, None)
        matches, _ = match_features(reference["descriptors"], descriptors)
        if len(matches) < CALIBRATION_MIN_POINTS:
            continue

        object_points = np.float32([reference["object_points"][m.queryIdx] for m in matches])
        image_points = np.float32([keypoints[m.trainIdx].pt for m in matches])

        _, mask = cv2.findHomography(object_points[:, :2], image_points, cv2.RANSAC, 3.0)
        mask = mask.ravel() == 1
        if mask.sum() < CALIBRATION_MIN_POINTS:
            continue
        all_object_points.append(object_points[mask])
        all_image_points.append(image_points[mask])
    video.release()

    # k3 wird festgehalten (= 0): bei wenig Verzeichnung nicht zuverlässig bestimmbar
    rms, camera_matrix, dist_coeffs, _, _ = cv2.calibrateCamera(
        all_object_points, all_image_points, image_size, None, None, flags=cv2.CALIB_FIX_K3)
    return camera_matrix, dist_coeffs, rms, len(all_object_points), image_size


def save_camera_parameters(camera_matrix, dist_coeffs, rms, image_size):
    fs = cv2.FileStorage(str(CAMERA_FILE), cv2.FILE_STORAGE_WRITE)
    fs.write("camera_matrix", camera_matrix)
    fs.write("distortion_coefficients", dist_coeffs)
    fs.write("reprojection_error", rms)
    fs.write("image_width", image_size[0])
    fs.write("image_height", image_size[1])
    fs.release()


def load_camera_parameters():
    fs = cv2.FileStorage(str(CAMERA_FILE), cv2.FILE_STORAGE_READ)
    assert fs.isOpened(), "Kameraparameter fehlen -> zuerst schritt2_korrespondenzen.py ausführen"
    camera_matrix = fs.getNode("camera_matrix").mat()
    dist_coeffs = fs.getNode("distortion_coefficients").mat()
    fs.release()
    return camera_matrix, dist_coeffs


def draw_correspondences(reference, frame, frame_keypoints, matches, inlier_mask=None):
    """Referenzbild links, Kamerabild rechts, Linien zwischen den Matches.

    Ohne inlier_mask werden alle Matches grün gezeichnet. Mit inlier_mask
    (aus Schritt 3) sind Inlier grün und Outlier rot.
    """
    # Referenzbild auf die Höhe des Kamerabilds skalieren
    scale = frame.shape[0] / reference["image"].shape[0]
    reference_image = cv2.resize(reference["image"], None, fx=scale, fy=scale)
    reference_keypoints = [cv2.KeyPoint(kp.pt[0] * scale, kp.pt[1] * scale, kp.size * scale)
                           for kp in reference["keypoints"]]

    if inlier_mask is None:
        inlier_mask = [True] * len(matches)
    inliers = [m for m, ok in zip(matches, inlier_mask) if ok]
    outliers = [m for m, ok in zip(matches, inlier_mask) if not ok]

    image = cv2.drawMatches(reference_image, reference_keypoints, frame, frame_keypoints,
                            outliers, None, matchColor=(0, 0, 255), singlePointColor=(0, 0, 255),
                            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    cv2.drawMatches(reference_image, reference_keypoints, frame, frame_keypoints,
                    inliers, image, matchColor=(0, 255, 0), singlePointColor=(0, 255, 0),
                    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS | cv2.DrawMatchesFlags_DRAW_OVER_OUTIMG)
    return image


# ---------------------------------------------------------------------------
# Hauptprogramm
# ---------------------------------------------------------------------------
def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    sift = create_sift()  # dieselben Parameter wie in Schritt 1

    # 2.1 Referenzdaten aus Schritt 1 ----------------------------------------
    reference = load_reference()
    print("2.1 Referenzmerkmale aus Schritt 1:", len(reference["keypoints"]))

    # 2.2 Kameraparameter bestimmen ------------------------------------------
    print("2.2 Kamerakalibrierung mit dem Flyer (dauert ca. 1 Minute) ...")
    camera_matrix, dist_coeffs, rms, n_frames, image_size = calibrate_camera_with_object(reference, sift)
    save_camera_parameters(camera_matrix, dist_coeffs, rms, image_size)
    print("    Verwendete Frames:", n_frames, "| Bildgröße:", image_size)
    print("    Kameramatrix K:")
    print(np.round(camera_matrix, 1))
    print("    Verzeichnung:", np.round(dist_coeffs.ravel(), 4))
    print("    RMS-Reprojektionsfehler:", round(rms, 3), "px")
    print("    -> Das Bild wird NICHT entzerrt, die Verzeichnung wird direkt in")
    print("       solvePnP berücksichtigt (Schritt 3).")

    for frame_index in TEST_FRAMES:
        # 2.3 Kamerabild lesen -----------------------------------------------
        frame = read_frame(frame_index)
        print(f"\nFrame {frame_index}:")

        # 2.4 SIFT im Kamerabild ---------------------------------------------
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_keypoints, frame_descriptors = sift.detectAndCompute(gray, None)
        print("2.4 SIFT-Merkmale im Kamerabild:", len(frame_keypoints))

        # 2.5 + 2.6 Matching und Ratio-Test ----------------------------------
        good_matches, all_matches = match_features(reference["descriptors"], frame_descriptors)
        print("2.5 Matches ohne Filter:       ", len(all_matches))
        print("2.6 Matches nach Ratio-Test:   ", len(good_matches))
        # Alle Referenzmerkmale besitzen eine 3D-Koordinate (Schritt 1),
        # daher ist jedes akzeptierte Match für die Pose Estimation verwendbar.

        # 2.7 Visualisierung -------------------------------------------------
        image_good = draw_correspondences(reference, frame, frame_keypoints, good_matches)
        cv2.imwrite(str(RESULTS_DIR / f"korrespondenzen_frame_{frame_index:04d}.jpg"), image_good)

        if frame_index == TEST_FRAMES[0]:
            # Vergleich: ohne Ratio-Test (zeigt, warum gefiltert werden muss)
            image_all = draw_correspondences(reference, frame, frame_keypoints, all_matches)
            cv2.imwrite(str(RESULTS_DIR / f"ohne_filter_frame_{frame_index:04d}.jpg"), image_all)

    print("\nBilder in:", RESULTS_DIR)


if __name__ == "__main__":
    main()
