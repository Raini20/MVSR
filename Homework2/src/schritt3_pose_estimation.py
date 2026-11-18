"""
MVSR Hausübung 2 – Schritt 3: 2D–3D-Korrespondenzen aufbauen und Pose schätzen

Baut auf Schritt 1 und 2 auf: verwendet die Referenzdaten aus Schritt 1 sowie
Kameraparameter, Matching und Visualisierung aus Schritt 2 – jetzt für jedes
Bild des Videos.

Ablauf pro Frame (entspricht Punkt 3 der Angabe):
  3.1  Merkmale extrahieren und matchen (wie Schritt 2)
  3.2  2D–3D-Korrespondenzen aufbauen (zwei Listen in gleicher Reihenfolge)
  3.3  Pose mit solvePnPRansac schätzen und mit solvePnPRefineLM verfeinern
  3.4  Translation und Rotation ausgeben
  3.5  Ergebnis visualisieren (Objektkoordinatensystem + Korrespondenzen)
Danach:
  3.6  Auswertung über das ganze Video (Wann funktioniert es, wann nicht?)

Aufruf (im Ordner src/):
  python schritt3_pose_estimation.py            mit Anzeigefenster (q = Ende)
  python schritt3_pose_estimation.py --no-gui   ohne Fenster
"""

import csv
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

from schritt1_referenzmerkmale import create_sift
from schritt2_korrespondenzen import (VIDEO_FILE, draw_correspondences, load_camera_parameters,
                                      load_reference, match_features)

# ---------------------------------------------------------------------------
# Einstellungen
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "media" / "results" / "schritt3"

FRAME_STEP = 2                  # jeden 2. Frame verarbeiten (Video hat 60 fps)
SHOW_WINDOW = "--no-gui" not in sys.argv

# RANSAC-Parameter für solvePnPRansac
RANSAC_ITERATIONS = 500         # Anzahl getesteter Pose-Hypothesen
RANSAC_REPROJECTION_ERROR = 4.0 # max. Abstand eines Inliers zur Reprojektion [px]
RANSAC_CONFIDENCE = 0.99

MIN_INLIERS = 15                # weniger Inlier -> Pose gilt als unzuverlässig

# Frames, die zusätzlich als Ergebnisbild gespeichert werden (verschiedene Ansichten)
SNAPSHOT_FRAMES = [100, 300, 600, 800, 1000, 1200]


# ---------------------------------------------------------------------------
# Funktionen
# ---------------------------------------------------------------------------
def build_correspondences(matches, reference, frame_keypoints):
    """Erzeugt die Liste der 3D-Objektpunkte und der 2D-Bildpunkte.

    Eintrag i beider Listen gehört zum selben Match:
      object_points[i] = 3D-Punkt des Referenzmerkmals   matches[i].queryIdx
      image_points[i]  = Bildpunkt des Kameramerkmals    matches[i].trainIdx
    """
    object_points = np.array([reference["object_points"][m.queryIdx] for m in matches], dtype=np.float64)
    image_points = np.array([frame_keypoints[m.trainIdx].pt for m in matches], dtype=np.float64)
    return object_points, image_points


def estimate_pose(object_points, image_points, camera_matrix, dist_coeffs):
    """Robuste Pose in drei Teilschritten:

    a) RANSAC (solvePnPRansac) trennt Inlier von Outliern. Jede Hypothese wird
       mit SOLVEPNP_AP3P aus nur 4 Punkten berechnet. Mit den anderen Flags
       verwendet OpenCV intern EPnP mit 5 Punkten, das bei ebenen Objekten
       (alle Z = 0) instabil ist und viel weniger Inlier findet (siehe README).
    b) Mit ALLEN Inliern wird die Pose neu berechnet: solvePnP mit
       SOLVEPNP_IPPE, einem Verfahren speziell für ebene Objekte.
       (Die von solvePnPRansac zurückgegebene Pose wird nicht verwendet,
       da OpenCV 4.x sie bei ebenen Objekten teilweise falsch berechnet.)
    c) Verfeinerung mit Levenberg-Marquardt (solvePnPRefineLM).
    Rückgabe: rvec, tvec, Inlier-Maske, mittlerer Reprojektionsfehler
              (oder None, wenn keine zuverlässige Pose gefunden wurde)
    """
    if len(object_points) < 6:
        return None

    # a) Inlier mit RANSAC bestimmen
    success, _, _, inliers = cv2.solvePnPRansac(
        object_points,
        image_points,
        camera_matrix,
        dist_coeffs,
        iterationsCount=RANSAC_ITERATIONS,
        reprojectionError=RANSAC_REPROJECTION_ERROR,
        confidence=RANSAC_CONFIDENCE,
        flags=cv2.SOLVEPNP_AP3P,
    )
    if not success or inliers is None or len(inliers) < MIN_INLIERS:
        return None

    inliers = inliers.ravel()

    # b) Pose aus allen Inliern berechnen
    success, rvec, tvec = cv2.solvePnP(object_points[inliers], image_points[inliers],
                                       camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_IPPE)
    if not success:
        return None

    # c) Verfeinerung
    rvec, tvec = cv2.solvePnPRefineLM(object_points[inliers], image_points[inliers],
                                      camera_matrix, dist_coeffs, rvec, tvec)

    # Reprojektionsfehler aller Punkte mit der verfeinerten Pose
    projected, _ = cv2.projectPoints(object_points, rvec, tvec, camera_matrix, dist_coeffs)
    errors = np.linalg.norm(projected.reshape(-1, 2) - image_points, axis=1)
    inlier_mask = errors < RANSAC_REPROJECTION_ERROR
    if inlier_mask.sum() < MIN_INLIERS:
        return None

    return rvec, tvec, inlier_mask, float(errors[inlier_mask].mean())


def draw_pose(frame, rvec, tvec, camera_matrix, dist_coeffs, reference):
    """Objektumriss und Objektkoordinatensystem in das Kamerabild zeichnen."""
    image = frame.copy()
    w, h = reference["width_mm"], reference["height_mm"]

    outline = np.array([[0, 0, 0], [w, 0, 0], [w, h, 0], [0, h, 0]], dtype=np.float64)
    outline_2d, _ = cv2.projectPoints(outline, rvec, tvec, camera_matrix, dist_coeffs)
    cv2.polylines(image, [np.int32(outline_2d)], True, (255, 255, 0), 3)

    # X rot, Y grün, Z blau; Achsenlänge 60 mm
    cv2.drawFrameAxes(image, camera_matrix, dist_coeffs, rvec, tvec, 60, 5)
    return image


def write_text(image, lines, color=(255, 255, 255)):
    for i, line in enumerate(lines):
        y = 45 + i * 45
        cv2.putText(image, line, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 0), 6)
        cv2.putText(image, line, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 1.1, color, 2)


# ---------------------------------------------------------------------------
# Hauptprogramm
# ---------------------------------------------------------------------------
def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    reference = load_reference()                         # Schritt 1
    sift = create_sift()                                 # Schritt 1

    video = cv2.VideoCapture(str(VIDEO_FILE))
    fps = video.get(cv2.CAP_PROP_FPS)
    camera_matrix, dist_coeffs = load_camera_parameters()  # in Schritt 2 bestimmt

    output_video = None
    results = []

    print(" Frame |  tx [mm]  ty [mm]  tz [mm] |       rvec [rad]        | Inlier/Matches | Fehler")
    print("-" * 92)

    frame_index = -1
    while True:
        ok, frame = video.read()
        if not ok:
            break
        frame_index += 1
        if frame_index % FRAME_STEP != 0:
            continue

        # 3.1 Merkmale und Matching (wie Schritt 2) ---------------------------
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_keypoints, frame_descriptors = sift.detectAndCompute(gray, None)
        matches, _ = match_features(reference["descriptors"], frame_descriptors)

        # 3.2 2D–3D-Korrespondenzen --------------------------------------------
        object_points, image_points = build_correspondences(matches, reference, frame_keypoints)

        # 3.3 Pose schätzen -----------------------------------------------------
        pose = estimate_pose(object_points, image_points, camera_matrix, dist_coeffs)

        # 3.4 Ausgabe -------------------------------------------------------------
        if pose is not None:
            rvec, tvec, inlier_mask, mean_error = pose
            t = tvec.ravel()
            r = rvec.ravel()
            n_inliers = int(inlier_mask.sum())
            print(f"{frame_index:6d} | {t[0]:8.1f} {t[1]:8.1f} {t[2]:8.1f} | "
                  f"[{r[0]:6.3f} {r[1]:6.3f} {r[2]:6.3f}] | {n_inliers:5d} / {len(matches):<6d} | {mean_error:.2f} px")

            # Neigung: Winkel zwischen Objektnormale (Z-Achse) und Blickrichtung der Kamera
            rotation_matrix, _ = cv2.Rodrigues(rvec)
            tilt = np.degrees(np.arccos(np.clip(-rotation_matrix[2, 2], -1, 1)))
            results.append([frame_index, 1, len(matches), n_inliers, mean_error, *t, *r,
                            np.linalg.norm(t), tilt])

            # 3.5 Visualisierung
            image_pose = draw_pose(frame, rvec, tvec, camera_matrix, dist_coeffs, reference)
            write_text(image_pose, [f"Frame {frame_index}",
                                    f"t = [{t[0]:.0f}, {t[1]:.0f}, {t[2]:.0f}] mm",
                                    f"Inlier {n_inliers}/{len(matches)}"])
        else:
            print(f"{frame_index:6d} | {'keine zuverlässige Pose':^26} | {'':23} | {'-':>5} / {len(matches):<6d} |")
            results.append([frame_index, 0, len(matches)] + [np.nan] * 10)
            inlier_mask = None
            image_pose = frame.copy()
            write_text(image_pose, [f"Frame {frame_index}", "keine zuverlaessige Pose"], color=(0, 0, 255))

        # Korrespondenzen (Inlier grün, Outlier rot) neben dem Ergebnis
        image_result = draw_correspondences(reference, image_pose, frame_keypoints, matches, inlier_mask)

        if frame_index in SNAPSHOT_FRAMES:
            cv2.imwrite(str(RESULTS_DIR / f"pose_frame_{frame_index:04d}.jpg"), image_pose)
            cv2.imwrite(str(RESULTS_DIR / f"korrespondenzen_pose_frame_{frame_index:04d}.jpg"), image_result)

        # Ergebnisvideo: verkleinert und mit 15 fps, damit die Datei für die Abgabe klein bleibt
        if frame_index % (2 * FRAME_STEP) == 0:
            image_small = cv2.resize(image_result, None, fx=0.3, fy=0.3)
            if output_video is None:
                output_video = cv2.VideoWriter(str(RESULTS_DIR / "pose_video.mp4"), cv2.VideoWriter_fourcc(*"mp4v"),
                                               fps / (2 * FRAME_STEP), (image_small.shape[1], image_small.shape[0]))
            output_video.write(image_small)

        if SHOW_WINDOW:
            cv2.imshow("Pose Estimation (q = Ende)", cv2.resize(image_result, None, fx=0.5, fy=0.5))
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    video.release()
    output_video.release()
    cv2.destroyAllWindows()

    # 3.6 Auswertung ---------------------------------------------------------------
    columns = ["frame", "pose_ok", "matches", "inlier", "fehler_px", "tx_mm", "ty_mm", "tz_mm",
               "rx_rad", "ry_rad", "rz_rad", "abstand_mm", "neigung_deg"]
    with open(RESULTS_DIR / "posen.csv", "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file, delimiter=";")
        writer.writerow(columns)
        for row in results:
            writer.writerow([round(v, 4) if isinstance(v, float) else v for v in row])

    data = np.array(results, dtype=np.float64)
    ok = data[:, 1] == 1
    print("-" * 92)
    print(f"Verarbeitete Frames: {len(data)}, davon mit Pose: {ok.sum()} ({100 * ok.mean():.1f} %)")
    print(f"Abstand mit Pose:  {np.nanmin(data[:, 11]):.0f} bis {np.nanmax(data[:, 11]):.0f} mm")
    print(f"Neigung mit Pose:  {np.nanmin(data[:, 12]):.0f} bis {np.nanmax(data[:, 12]):.0f} Grad")
    print(f"Mittlerer Reprojektionsfehler: {np.nanmean(data[:, 4]):.2f} px")

    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    axes[0].plot(data[:, 0], data[:, 5], label="tx")
    axes[0].plot(data[:, 0], data[:, 6], label="ty")
    axes[0].plot(data[:, 0], data[:, 7], label="tz")
    axes[0].set_ylabel("Translation [mm]")
    axes[0].legend()
    axes[1].plot(data[:, 0], data[:, 2], label="Matches nach Ratio-Test")
    axes[1].plot(data[:, 0], data[:, 3], label="RANSAC-Inlier")
    axes[1].axhline(MIN_INLIERS, color="red", linestyle="--", label="MIN_INLIERS")
    axes[1].set_ylabel("Anzahl")
    axes[1].legend()
    axes[2].plot(data[:, 0], data[:, 12], color="purple")
    axes[2].set_ylabel("Neigung [Grad]")
    axes[2].set_xlabel("Frame")
    for ax in axes:
        ax.grid(True)
        for frame_nr in data[~ok, 0]:
            ax.axvspan(frame_nr - FRAME_STEP / 2, frame_nr + FRAME_STEP / 2, color="red", alpha=0.15, lw=0)
    axes[0].set_title("Pose über das Video (rot hinterlegt: keine zuverlässige Pose)")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "auswertung.png", dpi=110)

    print("Ergebnisse in:", RESULTS_DIR)


if __name__ == "__main__":
    main()
