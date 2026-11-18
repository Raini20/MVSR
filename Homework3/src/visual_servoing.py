"""
MVSR Hausübung 3 – Visual Servoing (bildbasiert, IBVS)

Baut auf der Pose Estimation aus HÜ 2 auf (SIFT, kNN-Matching mit Ratio-Test,
solvePnPRansac + IPPE + LM) und erweitert sie um Regelfehler und Geschwindigkeit.

Ablauf (entspricht den Punkten der Angabe):
  1.  Soll- und Istzustand bestimmen
      1.1  Referenzbild laden, SIFT-Merkmale extrahieren und auswählen (3D-Punkte, Z = 0)
      1.2  Sollzustand: Sollpose der Kamera und daraus die Soll-Bildpunkte s*
      1.3  Istzustand je Frame: Matching, Pose (PnP) und die gemessenen Bildpunkte s
  2.  Regelfehler e = s - s* und Geschwindigkeit v = -lambda * L^+ * e
  3.  Anwendung auf das ganze Video: Ausgabe, Visualisierung, Auswertung

Aufruf (im Ordner src/):
  python visual_servoing.py -reference ../media/reference/reference_image.png -video ../media/video/camera_video.mp4
  Optional: --no-gui (ohne Anzeigefenster), -camera <yaml>, -object_size <Breite> <Höhe> [mm]
"""

import argparse
import csv
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# Einstellungen
# ---------------------------------------------------------------------------
# SIFT (Standardwerte nach Lowe, wie in HÜ 2)
SIFT_N_FEATURES = 0
SIFT_N_OCTAVE_LAYERS = 3
SIFT_CONTRAST_THRESHOLD = 0.04
SIFT_EDGE_THRESHOLD = 10
SIFT_SIGMA = 1.6

# Auswahl der Referenzmerkmale (wie in HÜ 2)
BORDER_MARGIN_MM = 3.0           # Mindestabstand zum Objektrand
MIN_SELF_DISTANCE = 200.0        # Mindest-Deskriptordistanz zu anderen Merkmalen

# Matching und Pose (wie in HÜ 2)
RATIO_THRESHOLD = 0.75
RANSAC_ITERATIONS = 500
RANSAC_REPROJECTION_ERROR = 4.0  # [px]
RANSAC_CONFIDENCE = 0.99
MIN_INLIERS = 15                 # weniger Inlier -> kein verlässlicher Zustand

# Visual Servoing
GAIN = 1.0                       # Reglerverstärkung lambda [1/s]
FRAME_STEP = 2                   # jeden 2. Frame verarbeiten (Video hat 60 fps)

# Visualisierung
V_MAX_TRANS = 150.0              # Vollausschlag der Balken: Translation [mm/s]
V_MAX_ROT = 0.5                  # Vollausschlag der Balken: Rotation [rad/s]
SNAPSHOT_FRAMES = [100, 300, 600, 800, 1000, 1200]


# ---------------------------------------------------------------------------
# 1.1  Referenz: Merkmale und 3D-Punkte
# ---------------------------------------------------------------------------
def create_sift():
    return cv2.SIFT_create(nfeatures=SIFT_N_FEATURES, nOctaveLayers=SIFT_N_OCTAVE_LAYERS,
                           contrastThreshold=SIFT_CONTRAST_THRESHOLD,
                           edgeThreshold=SIFT_EDGE_THRESHOLD, sigma=SIFT_SIGMA)


def load_reference(path, width_mm, height_mm, sift):
    """Referenzbild laden und geeignete SIFT-Merkmale mit 3D-Punkten bestimmen.

    Das Referenzbild ist eine frontale Ansicht des Objekts, die genau das
    Objekt ausfüllt (Ränder = Objektränder). Daraus folgt der Maßstab
    (Pixel pro mm) und für jedes Merkmal bei Pixel (u, v) der 3D-Punkt
        X = u / s,  Y = H - v / s,  Z = 0    [mm]
    (Objektkoordinatensystem: Ursprung links unten, X rechts, Y oben,
    Z aus der Oberfläche heraus).
    Die Auswahl der Merkmale ist dieselbe wie in HÜ 2 (Rand, Eindeutigkeit).
    """
    image = cv2.imread(str(path))
    assert image is not None, f"Referenzbild nicht gefunden: {path}"
    height_px, width_px = image.shape[:2]
    px_per_mm = width_px / width_mm
    assert abs(height_px / px_per_mm - height_mm) < 0.02 * height_mm, \
        "Seitenverhältnis des Referenzbilds passt nicht zu -object_size"

    keypoints, descriptors = sift.detectAndCompute(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), None)

    margin_px = BORDER_MARGIN_MM * px_per_mm
    positions = np.array([kp.pt for kp in keypoints])
    self_matches = cv2.BFMatcher(cv2.NORM_L2).knnMatch(descriptors, descriptors, k=5)

    selected = []
    for i, kp in enumerate(keypoints):
        x, y = kp.pt
        border = max(margin_px, kp.size / 2)
        if x < border or y < border or x > width_px - border or y > height_px - border:
            continue                                   # zu nah am Rand
        nearest_other = np.inf
        for match in self_matches[i]:
            if np.linalg.norm(positions[match.trainIdx] - positions[i]) > 5.0:
                nearest_other = match.distance
                break
        if nearest_other >= MIN_SELF_DISTANCE:         # eindeutig
            selected.append(i)

    pixels = positions[selected]
    object_points = np.zeros((len(selected), 3))
    object_points[:, 0] = pixels[:, 0] / px_per_mm
    object_points[:, 1] = height_mm - pixels[:, 1] / px_per_mm

    return {"image": image, "keypoints": [keypoints[i] for i in selected],
            "descriptors": descriptors[selected], "object_points": object_points,
            "width_mm": width_mm, "height_mm": height_mm, "px_per_mm": px_per_mm}


def load_camera_parameters(path):
    fs = cv2.FileStorage(str(path), cv2.FILE_STORAGE_READ)
    assert fs.isOpened(), f"Kameraparameter nicht gefunden: {path}"
    camera_matrix = fs.getNode("camera_matrix").mat()
    dist_coeffs = fs.getNode("distortion_coefficients").mat()
    fs.release()
    return camera_matrix, dist_coeffs


# ---------------------------------------------------------------------------
# 1.2  Sollzustand
# ---------------------------------------------------------------------------
def desired_pose(reference, camera_matrix):
    """Sollpose des Objekts im Kamerakoordinatensystem (R*, t*).

    Das Referenzbild beschreibt die Sollansicht: Die Kamera blickt frontal auf
    das Objekt, dieses ist im Bild zentriert und wird mit dem Maßstab des
    Referenzbilds abgebildet. Daraus folgt der Sollabstand
        Z* = f / (Pixel pro mm).

    Kamerakoordinatensystem (OpenCV): x rechts, y nach unten, z in Blickrichtung.
    Frontal heißt: Kamera-x = Objekt-X, Kamera-y = -Objekt-Y (Bild-y zeigt nach
    unten, Objekt-Y nach oben), Kamera-z = -Objekt-Z (Kamera blickt auf die Fläche):
        R* = diag(1, -1, -1)
    Das Objektzentrum c = (W/2, H/2, 0) soll auf der optischen Achse bei Z* liegen:
        R* c + t* = (0, 0, Z*)   ->   t* = (0, 0, Z*) - R* c
    """
    focal = (camera_matrix[0, 0] + camera_matrix[1, 1]) / 2
    z_desired = focal / reference["px_per_mm"]
    R = np.diag([1.0, -1.0, -1.0])
    center = np.array([reference["width_mm"] / 2, reference["height_mm"] / 2, 0.0])
    t = np.array([0.0, 0.0, z_desired]) - R @ center
    return R, t.reshape(3, 1)


# ---------------------------------------------------------------------------
# 1.3  Istzustand
# ---------------------------------------------------------------------------
def match_features(reference_descriptors, frame_descriptors):
    """Brute-Force-Matching (kNN, k = 2) mit Ratio-Test nach Lowe."""
    knn_matches = cv2.BFMatcher(cv2.NORM_L2).knnMatch(reference_descriptors, frame_descriptors, k=2)
    return [pair[0] for pair in knn_matches
            if len(pair) == 2 and pair[0].distance < RATIO_THRESHOLD * pair[1].distance]


def estimate_pose(object_points, image_points, camera_matrix, dist_coeffs):
    """Pose wie in HÜ 2: a) RANSAC (AP3P) für die Inlier, b) solvePnP mit IPPE
    aus allen Inliern, c) Verfeinerung mit solvePnPRefineLM.
    Rückgabe: rvec, tvec, Inlier-Maske, mittlerer Reprojektionsfehler [px] (oder None)."""
    if len(object_points) < 6:
        return None

    success, _, _, inliers = cv2.solvePnPRansac(
        object_points, image_points, camera_matrix, dist_coeffs,
        iterationsCount=RANSAC_ITERATIONS, reprojectionError=RANSAC_REPROJECTION_ERROR,
        confidence=RANSAC_CONFIDENCE, flags=cv2.SOLVEPNP_AP3P)
    if not success or inliers is None or len(inliers) < MIN_INLIERS:
        return None
    inliers = inliers.ravel()

    success, rvec, tvec = cv2.solvePnP(object_points[inliers], image_points[inliers],
                                       camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_IPPE)
    if not success:
        return None
    rvec, tvec = cv2.solvePnPRefineLM(object_points[inliers], image_points[inliers],
                                      camera_matrix, dist_coeffs, rvec, tvec)

    projected, _ = cv2.projectPoints(object_points, rvec, tvec, camera_matrix, dist_coeffs)
    errors = np.linalg.norm(projected.reshape(-1, 2) - image_points, axis=1)
    inlier_mask = errors < RANSAC_REPROJECTION_ERROR
    if inlier_mask.sum() < MIN_INLIERS:
        return None
    return rvec, tvec, inlier_mask, float(errors[inlier_mask].mean())


# ---------------------------------------------------------------------------
# 2.  Regelfehler und Geschwindigkeit (IBVS)
# ---------------------------------------------------------------------------
def to_normalized(image_points, camera_matrix, dist_coeffs):
    """Pixel -> normierte Bildkoordinaten (x, y) = (X/Z, Y/Z) im Kamerasystem.
    Die Verzeichnung wird dabei entfernt (undistortPoints)."""
    return cv2.undistortPoints(image_points.reshape(-1, 1, 2), camera_matrix, dist_coeffs).reshape(-1, 2)


def interaction_matrix(xy, z):
    """Bild-Jacobi-Matrix L (2N x 6) für N Punkte in normierten Koordinaten.

    Für einen Punkt (x, y) mit Tiefe Z gilt (Chaumette & Hutchinson):
        [x'] = [-1/Z   0    x/Z   x*y     -(1+x²)   y ]
        [y']   [ 0   -1/Z   y/Z   1+y²    -x*y     -x ]  * (vx vy vz wx wy wz)^T
    """
    x, y = xy[:, 0], xy[:, 1]
    L = np.zeros((2 * len(xy), 6))
    L[0::2, 0] = -1 / z
    L[0::2, 2] = x / z
    L[0::2, 3] = x * y
    L[0::2, 4] = -(1 + x ** 2)
    L[0::2, 5] = y
    L[1::2, 1] = -1 / z
    L[1::2, 2] = y / z
    L[1::2, 3] = 1 + y ** 2
    L[1::2, 4] = -x * y
    L[1::2, 5] = -x
    return L


def compute_velocity(object_points, image_points, rvec, tvec, desired, camera_matrix, dist_coeffs):
    """Regelfehler und Geschwindigkeit aus den Inlier-Punkten eines Frames.

    s  : gemessene Bildpunkte (normiert)        -> Istzustand
    s* : Bildpunkte der Sollpose (normiert)     -> Sollzustand
    e  = s - s*                                 -> Regelfehler (2N Werte)
    Z  : Tiefe der Punkte im Kamerasystem aus der aktuellen Pose (HÜ 2)
    v  = -lambda * L^+ * e                      -> Regelgesetz
    Rückgabe: v (6,), mittlerer Bildfehler [px], s*-Pixelpunkte
    """
    R_d, t_d = desired

    s = to_normalized(image_points, camera_matrix, dist_coeffs)
    points_desired = (R_d @ object_points.T + t_d).T
    s_desired = points_desired[:, :2] / points_desired[:, 2:3]

    R, _ = cv2.Rodrigues(rvec)
    z = (R @ object_points.T + tvec).T[:, 2]

    e = (s - s_desired).reshape(-1)
    L = interaction_matrix(s, z)
    v = -GAIN * np.linalg.pinv(L) @ e

    # Fehler in Pixel (zur Anschauung): Abstand der Punkte zu ihrer Sollposition
    focal = (camera_matrix[0, 0] + camera_matrix[1, 1]) / 2
    error_px = float(np.mean(np.linalg.norm((s - s_desired), axis=1)) * focal)
    return v, error_px


def pose_error(rvec, tvec, desired):
    """Zur Beurteilung: Wie weit ist die Kamera von der Sollpose entfernt?

    Rückgabe: Verschiebung (mm) vom aktuellen Kameraursprung zum Soll-Kameraursprung,
    ausgedrückt im aktuellen Kamerasystem, und Drehwinkel [Grad] zwischen beiden Lagen.
    """
    R, _ = cv2.Rodrigues(rvec)
    R_d, t_d = desired
    R_rel = R_d @ R.T                                   # aktuelle Kamera -> Soll-Kamera
    t_rel = t_d - R_rel @ tvec
    shift = -R_rel.T @ t_rel                            # Soll-Kameraursprung im aktuellen Kamerasystem
    angle = np.degrees(np.arccos(np.clip((np.trace(R_rel) - 1) / 2, -1, 1)))
    return shift.ravel(), float(angle)


# ---------------------------------------------------------------------------
# Visualisierung
# ---------------------------------------------------------------------------
def draw_text(image, lines, x=20, y0=45, color=(255, 255, 255), scale=1.0):
    for i, line in enumerate(lines):
        y = y0 + int(i * 42 * scale)
        cv2.putText(image, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 5)
        cv2.putText(image, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2)


def draw_outline(image, reference, rvec, tvec, camera_matrix, dist_coeffs, color):
    w, h = reference["width_mm"], reference["height_mm"]
    outline = np.array([[0, 0, 0], [w, 0, 0], [w, h, 0], [0, h, 0]], dtype=np.float64)
    points, _ = cv2.projectPoints(outline, rvec, tvec, camera_matrix, dist_coeffs)
    cv2.polylines(image, [np.int32(points)], True, color, 3)


def draw_velocity(image, v):
    """Bewegung einzeichnen: Pfeil für (vx, vy) in der Bildmitte und sechs Balken."""
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    tip = (int(center[0] + np.clip(v[0] / V_MAX_TRANS, -1, 1) * 250),
           int(center[1] + np.clip(v[1] / V_MAX_TRANS, -1, 1) * 250))
    cv2.circle(image, center, 8, (0, 0, 0), -1)
    cv2.arrowedLine(image, center, tip, (0, 0, 0), 12, tipLength=0.25)
    cv2.arrowedLine(image, center, tip, (0, 215, 255), 6, tipLength=0.25)

    names = ["vx", "vy", "vz", "wx", "wy", "wz"]
    for i in range(6):
        y = h - 40 - (5 - i) * 42
        full = V_MAX_TRANS if i < 3 else V_MAX_ROT
        length = int(np.clip(v[i] / full, -1, 1) * 150)
        x0 = 120 + 150
        cv2.putText(image, names[i], (20, y + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 5)
        cv2.putText(image, names[i], (20, y + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        cv2.rectangle(image, (x0 - 150, y - 12), (x0 + 150, y + 16), (60, 60, 60), -1)
        cv2.rectangle(image, (x0, y - 12), (x0 + length, y + 16), (0, 215, 255), -1)
        cv2.line(image, (x0, y - 16), (x0, y + 20), (255, 255, 255), 2)


def draw_state(frame, reference, rvec, tvec, desired, camera_matrix, dist_coeffs,
               image_points, object_points, v, info):
    """Ist- und Sollzustand im Kamerabild: aktueller Umriss (gelb), Sollumriss (grün),
    gemessene Punkte s (rot) mit Linie zur Sollposition s* (grün)."""
    image = frame.copy()
    R_d, t_d = desired
    rvec_d, _ = cv2.Rodrigues(R_d)

    draw_outline(image, reference, rvec, tvec, camera_matrix, dist_coeffs, (0, 255, 255))
    draw_outline(image, reference, rvec_d, t_d, camera_matrix, dist_coeffs, (0, 200, 0))

    target, _ = cv2.projectPoints(object_points, rvec_d, t_d, camera_matrix, dist_coeffs)
    for p, q in zip(image_points, target.reshape(-1, 2)):
        cv2.line(image, tuple(np.int32(p)), tuple(np.int32(q)), (255, 255, 255), 2)
        cv2.circle(image, tuple(np.int32(q)), 6, (0, 200, 0), -1)
        cv2.circle(image, tuple(np.int32(p)), 6, (0, 0, 255), -1)

    draw_velocity(image, v)
    draw_text(image, info)
    return image


# ---------------------------------------------------------------------------
# Hauptprogramm
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="MVSR HÜ3 – Visual Servoing (IBVS)")
    parser.add_argument("-reference", "--reference", required=True, help="Referenzbild (Sollansicht, frontal)")
    parser.add_argument("-video", "--video", required=True, help="Video (Istansichten)")
    parser.add_argument("-camera", "--camera", default=str(ROOT / "config" / "camera_parameters.yaml"),
                        help="Kameraparameter (.yaml)")
    parser.add_argument("-object_size", "--object_size", type=float, nargs=2, default=[101.0, 210.0],
                        metavar=("BREITE", "HOEHE"), help="Objektgröße in mm (Standard: 101 210)")
    parser.add_argument("-output", "--output", default=str(ROOT / "media" / "results"),
                        help="Ordner für Ergebnisse")
    parser.add_argument("--no-gui", action="store_true", help="kein Anzeigefenster")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    sift = create_sift()

    # 1.1 Referenz ---------------------------------------------------------------
    reference = load_reference(args.reference, args.object_size[0], args.object_size[1], sift)
    camera_matrix, dist_coeffs = load_camera_parameters(args.camera)
    print(f"Referenz: {len(reference['keypoints'])} Merkmale, Maßstab {reference['px_per_mm']:.2f} px/mm")

    # 1.2 Sollzustand --------------------------------------------------------------
    desired = desired_pose(reference, camera_matrix)
    print(f"Sollpose: Kamera frontal vor dem Objekt, Sollabstand Z* = {desired[1][2, 0]:.0f} mm")
    print(f"Regelverstärkung lambda = {GAIN} 1/s | v: Translation [mm/s], Rotation [rad/s], Kamerasystem")
    print()

    video = cv2.VideoCapture(args.video)
    assert video.isOpened(), f"Video nicht gefunden: {args.video}"
    fps = video.get(cv2.CAP_PROP_FPS)

    output_video = None
    results = []

    print(" Frame |   vx      vy      vz  [mm/s] |   wx      wy      wz [rad/s] | Fehler [px] | Inlier")
    print("-" * 98)

    frame_index = -1
    while True:
        ok, frame = video.read()
        if not ok:
            break
        frame_index += 1
        if frame_index % FRAME_STEP != 0:
            continue

        # 1.3 Istzustand: Merkmale, Matching, Pose ------------------------------------
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_keypoints, frame_descriptors = sift.detectAndCompute(gray, None)
        matches = match_features(reference["descriptors"], frame_descriptors)
        object_points = np.array([reference["object_points"][m.queryIdx] for m in matches])
        image_points = np.array([frame_keypoints[m.trainIdx].pt for m in matches])
        pose = estimate_pose(object_points, image_points, camera_matrix, dist_coeffs)

        if pose is not None:
            rvec, tvec, inlier_mask, _ = pose
            object_points, image_points = object_points[inlier_mask], image_points[inlier_mask]
            n_inliers = int(inlier_mask.sum())

            # 2. Regelfehler und Geschwindigkeit ------------------------------------------
            v, error_px = compute_velocity(object_points, image_points, rvec, tvec, desired,
                                           camera_matrix, dist_coeffs)
            shift, angle = pose_error(rvec, tvec, desired)

            print(f"{frame_index:6d} | {v[0]:7.1f} {v[1]:7.1f} {v[2]:7.1f}    | "
                  f"{v[3]:7.3f} {v[4]:7.3f} {v[5]:7.3f}    | {error_px:8.1f}    | {n_inliers:4d}")
            results.append([frame_index, 1, n_inliers, error_px, *v, *shift, angle])

            info = [f"Frame {frame_index}",
                    f"v = [{v[0]:.0f} {v[1]:.0f} {v[2]:.0f}] mm/s",
                    f"w = [{v[3]:.2f} {v[4]:.2f} {v[5]:.2f}] rad/s",
                    f"Bildfehler {error_px:.0f} px | Inlier {n_inliers}"]
            image = draw_state(frame, reference, rvec, tvec, desired, camera_matrix, dist_coeffs,
                               image_points, object_points, v, info)
        else:
            # Kein verlässlicher Zustand: es wird KEINE Geschwindigkeit aus dem Frame berechnet.
            # Als Sicherheitsbefehl gilt v = 0 (Kamera bleibt stehen).
            v = np.zeros(6)
            print(f"{frame_index:6d} | keine verlässliche Pose ({len(matches)} Matches)  ->  v = 0 (Stopp)")
            results.append([frame_index, 0, 0] + [np.nan] * 11)
            image = frame.copy()
            draw_text(image, [f"Frame {frame_index}", "kein verlaesslicher Zustand", "v = 0 (Stopp)"],
                      color=(0, 0, 255))

        if frame_index in SNAPSHOT_FRAMES:
            cv2.imwrite(str(output_dir / f"visual_servoing_frame_{frame_index:04d}.jpg"), image)

        # Ergebnisvideo: verkleinert und mit 15 fps, damit die Datei klein bleibt
        if frame_index % (2 * FRAME_STEP) == 0:
            small = cv2.resize(image, None, fx=0.5, fy=0.5)
            if output_video is None:
                output_video = cv2.VideoWriter(str(output_dir / "visual_servoing_video.mp4"),
                                               cv2.VideoWriter_fourcc(*"mp4v"), fps / (2 * FRAME_STEP),
                                               (small.shape[1], small.shape[0]))
            output_video.write(small)

        if not args.no_gui:
            cv2.imshow("Visual Servoing (q = Ende)", cv2.resize(image, None, fx=0.5, fy=0.5))
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    video.release()
    if output_video is not None:
        output_video.release()
    cv2.destroyAllWindows()

    # 3. Auswertung --------------------------------------------------------------------
    columns = ["frame", "zustand_ok", "inlier", "bildfehler_px", "vx_mm_s", "vy_mm_s", "vz_mm_s",
               "wx_rad_s", "wy_rad_s", "wz_rad_s", "dx_mm", "dy_mm", "dz_mm", "drehfehler_deg"]
    with open(output_dir / "geschwindigkeiten.csv", "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file, delimiter=";")
        writer.writerow(columns)
        for row in results:
            writer.writerow([round(float(x), 4) for x in row])

    data = np.array(results, dtype=np.float64)
    valid = data[:, 1] == 1
    print("-" * 98)
    print(f"Verarbeitete Frames: {len(data)}, davon mit Zustand: {valid.sum()} ({100 * valid.mean():.1f} %)")
    print(f"Bildfehler: Median {np.nanmedian(data[:, 3]):.0f} px, Minimum {np.nanmin(data[:, 3]):.0f} px")
    print(f"|v_trans|: Median {np.nanmedian(np.linalg.norm(data[:, 4:7], axis=1)):.0f} mm/s, "
          f"|w|: Median {np.nanmedian(np.linalg.norm(data[:, 7:10], axis=1)):.2f} rad/s")

    fig, axes = plt.subplots(4, 1, figsize=(10, 11), sharex=True)
    for i, name in enumerate(["vx", "vy", "vz"]):
        axes[0].plot(data[:, 0], data[:, 4 + i], label=name)
    axes[0].set_ylabel("Translation [mm/s]")
    for i, name in enumerate(["wx", "wy", "wz"]):
        axes[1].plot(data[:, 0], data[:, 7 + i], label=name)
    axes[1].set_ylabel("Rotation [rad/s]")
    axes[2].plot(data[:, 0], data[:, 3], color="black")
    axes[2].set_ylabel("mittlerer Bildfehler [px]")
    axes[3].plot(data[:, 0], np.linalg.norm(data[:, 10:13], axis=1), label="Verschiebung [mm]")
    axes[3].plot(data[:, 0], data[:, 13], label="Drehung [Grad]")
    axes[3].set_ylabel("Abstand zur Sollpose")
    axes[3].set_xlabel("Frame")
    for ax in axes:
        ax.grid(True)
        ax.legend(loc="upper right") if ax is not axes[2] else None
        for frame_nr in data[~valid, 0]:
            ax.axvspan(frame_nr - FRAME_STEP / 2, frame_nr + FRAME_STEP / 2, color="red", alpha=0.15, lw=0)
    axes[0].set_title("Visual Servoing über das Video (rot hinterlegt: kein verlässlicher Zustand)")
    fig.tight_layout()
    fig.savefig(output_dir / "auswertung.png", dpi=110)
    print("Ergebnisse in:", output_dir)


if __name__ == "__main__":
    main()
