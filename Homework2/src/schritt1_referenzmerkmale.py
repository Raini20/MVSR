"""
MVSR Hausübung 2 – Schritt 1: Features und Deskriptoren lokaler Bildmerkmale extrahieren

Ablauf (entspricht Punkt 1 der Angabe):
  1.1  Referenzfoto laden
  1.2  Scharfes, frontales Referenzbild des Objekts erstellen
  1.3  Objektkoordinatensystem festlegen
  1.4  SIFT-Merkmale und Deskriptoren extrahieren
  1.5  Merkmale visualisieren
  1.6  Geeignete Referenzmerkmale auswählen
  1.7  3D-Koordinaten der ausgewählten Merkmale bestimmen
  1.8  Referenzdaten speichern (.yaml und .csv)

Aufruf (im Ordner src/):
  python schritt1_referenzmerkmale.py
"""

import csv
from pathlib import Path

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Einstellungen
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent

REFERENCE_PHOTO = ROOT / "media" / "reference" / "reference_photo.jpg"
REFERENCE_IMAGE = ROOT / "media" / "reference" / "reference_image.png"
REFERENCE_YAML = ROOT / "data" / "reference_features.yaml"
REFERENCE_CSV = ROOT / "data" / "reference_points.csv"
RESULTS_DIR = ROOT / "media" / "results" / "schritt1"

# Reale Größe des Objekts (Flyer), mit dem Lineal gemessen
OBJECT_WIDTH_MM = 101.0
OBJECT_HEIGHT_MM = 210.0

# Pixelkoordinaten der vier Objektecken im Referenzfoto
# Reihenfolge: links-oben, rechts-oben, rechts-unten, links-unten
OBJECT_CORNERS_PX = np.float32([
    [829.8, 726.5],
    [2263.2, 661.0],
    [2242.4, 3483.4],
    [969.6, 3506.6],
])

# Auflösung des Referenzbilds: 5 Pixel pro Millimeter
# -> Objekt wird 505 x 1050 px groß, also ungefähr so groß wie im Video
PX_PER_MM = 5.0

# SIFT-Parameter (Standardwerte nach Lowe). Schritt 2 und 3 verwenden über
# create_sift() exakt dieselben Werte.
SIFT_N_FEATURES = 0              # keine Begrenzung, Auswahl erfolgt in 1.6
SIFT_N_OCTAVE_LAYERS = 3         # Skalenstufen pro Oktave
SIFT_CONTRAST_THRESHOLD = 0.04   # kontrastarme (rauschanfällige) Punkte verwerfen
SIFT_EDGE_THRESHOLD = 10         # Punkte auf Kanten verwerfen
SIFT_SIGMA = 1.6                 # Glättung der ersten Oktave

# Auswahl der Referenzmerkmale
BORDER_MARGIN_MM = 3.0           # Mindestabstand zum Objektrand
MIN_SELF_DISTANCE = 200.0        # Mindest-Deskriptordistanz zu anderen Merkmalen


# ---------------------------------------------------------------------------
# Funktionen
# ---------------------------------------------------------------------------
def create_sift():
    """SIFT-Detektor mit den Parametern dieser Hausübung."""
    return cv2.SIFT_create(
        nfeatures=SIFT_N_FEATURES,
        nOctaveLayers=SIFT_N_OCTAVE_LAYERS,
        contrastThreshold=SIFT_CONTRAST_THRESHOLD,
        edgeThreshold=SIFT_EDGE_THRESHOLD,
        sigma=SIFT_SIGMA,
    )


def pixel_to_object(points_px):
    """Pixel (u, v) im Referenzbild -> Objektkoordinaten (X, Y, Z) in mm.

    Ursprung: linke untere Ecke, X nach rechts, Y nach oben, Z = 0.
    Die Bildzeile v zählt nach unten, Y nach oben, daher Y = H - v / s.
    """
    points_px = np.asarray(points_px, dtype=np.float64).reshape(-1, 2)
    object_points = np.zeros((len(points_px), 3), dtype=np.float64)
    object_points[:, 0] = points_px[:, 0] / PX_PER_MM
    object_points[:, 1] = OBJECT_HEIGHT_MM - points_px[:, 1] / PX_PER_MM
    return object_points


def draw_coordinate_system(image):
    """Ursprung und Achsen des Objektkoordinatensystems einzeichnen."""
    height = image.shape[0]
    origin = (0, height - 1)
    length = int(40 * PX_PER_MM)  # 40 mm lange Achsen

    cv2.arrowedLine(image, origin, (length, height - 1), (0, 0, 255), 6, tipLength=0.1)
    cv2.arrowedLine(image, origin, (0, height - 1 - length), (0, 255, 0), 6, tipLength=0.1)
    cv2.circle(image, origin, 12, (255, 0, 0), -1)
    cv2.putText(image, "X", (length - 25, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
    cv2.putText(image, "Y", (12, height - length + 25), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
    return image


# ---------------------------------------------------------------------------
# Hauptprogramm
# ---------------------------------------------------------------------------
def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    REFERENCE_YAML.parent.mkdir(parents=True, exist_ok=True)

    # 1.1 Referenzfoto laden ------------------------------------------------
    photo = cv2.imread(str(REFERENCE_PHOTO))
    assert photo is not None, f"Referenzfoto nicht gefunden: {REFERENCE_PHOTO}"
    print("1.1 Referenzfoto:", photo.shape[1], "x", photo.shape[0], "px")

    # 1.2 Frontales Referenzbild erstellen ----------------------------------
    # Das Foto ist leicht schräg aufgenommen. Über die vier Ecken wird eine
    # Homographie berechnet, die das Objekt frontal und mit festem Maßstab
    # (PX_PER_MM) darstellt. Danach entspricht jeder Pixel einer bekannten
    # Position am Objekt.
    width_px = int(OBJECT_WIDTH_MM * PX_PER_MM)
    height_px = int(OBJECT_HEIGHT_MM * PX_PER_MM)
    target_corners = np.float32([
        [0, 0],
        [width_px, 0],
        [width_px, height_px],
        [0, height_px],
    ])
    homography = cv2.getPerspectiveTransform(OBJECT_CORNERS_PX, target_corners)
    reference_image = cv2.warpPerspective(photo, homography, (width_px, height_px), flags=cv2.INTER_AREA)
    cv2.imwrite(str(REFERENCE_IMAGE), reference_image)
    print("1.2 Referenzbild:", width_px, "x", height_px, "px ->", REFERENCE_IMAGE.name)

    # 1.3 Objektkoordinatensystem -------------------------------------------
    print("1.3 Objektkoordinatensystem: Ursprung links unten, X rechts, Y oben, "
          "Z aus der Oberfläche heraus, Einheit mm")
    print("    Ecke rechts oben (Pixel", (width_px, 0), ") ->",
          pixel_to_object([[width_px, 0]])[0], "mm")

    # 1.4 SIFT-Merkmale extrahieren -----------------------------------------
    gray = cv2.cvtColor(reference_image, cv2.COLOR_BGR2GRAY)
    sift = create_sift()
    keypoints, descriptors = sift.detectAndCompute(gray, None)
    print("1.4 Gefundene SIFT-Merkmale:", len(keypoints))
    print("    Form der Deskriptor-Matrix:", descriptors.shape)

    # 1.5 Alle Merkmale visualisieren ---------------------------------------
    image_all = cv2.drawKeypoints(reference_image, keypoints, None,
                                  flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv2.imwrite(str(RESULTS_DIR / "1_alle_merkmale.png"), image_all)

    # 1.6 Geeignete Referenzmerkmale auswählen ------------------------------
    # Kriterium a) Abstand zum Objektrand:
    #   Liegt ein Merkmal zu nahe am Rand, enthält seine Umgebung im Kamerabild
    #   auch Hintergrund -> Deskriptor passt nicht mehr.
    # Kriterium b) Eindeutigkeit:
    #   Hat ein Merkmal an einer ANDEREN Stelle des Objekts einen sehr ähnlichen
    #   Deskriptor (z.B. gleiche Buchstaben, wiederholende Muster), ist die
    #   Zuordnung im Kamerabild nicht eindeutig -> verwerfen.
    margin_px = BORDER_MARGIN_MM * PX_PER_MM
    positions = np.array([kp.pt for kp in keypoints])

    matcher = cv2.BFMatcher(cv2.NORM_L2)
    self_matches = matcher.knnMatch(descriptors, descriptors, k=5)

    selected = []
    rejected_border = []
    rejected_ambiguous = []

    for i, kp in enumerate(keypoints):
        x, y = kp.pt
        border = max(margin_px, kp.size / 2)
        if x < border or y < border or x > width_px - border or y > height_px - border:
            rejected_border.append(i)
            continue

        # ähnlichstes Merkmal an einer anderen Stelle suchen
        # (Punkte an derselben Stelle, z.B. zweite Orientierung, zählen nicht)
        nearest_other_distance = np.inf
        for match in self_matches[i]:
            if np.linalg.norm(positions[match.trainIdx] - positions[i]) > 5.0:
                nearest_other_distance = match.distance
                break

        if nearest_other_distance < MIN_SELF_DISTANCE:
            rejected_ambiguous.append(i)
        else:
            selected.append(i)

    print("1.6 Auswahl der Referenzmerkmale:")
    print("    ausgewählt:          ", len(selected))
    print("    verworfen (Rand):    ", len(rejected_border))
    print("    verworfen (mehrdeutig):", len(rejected_ambiguous))

    image_selected = reference_image.copy()
    for i in rejected_border:
        cv2.circle(image_selected, tuple(np.int32(keypoints[i].pt)), 4, (0, 0, 255), -1)
    for i in rejected_ambiguous:
        cv2.circle(image_selected, tuple(np.int32(keypoints[i].pt)), 4, (255, 0, 255), -1)
    for i in selected:
        cv2.circle(image_selected, tuple(np.int32(keypoints[i].pt)), 4, (0, 255, 0), -1)
    draw_coordinate_system(image_selected)
    cv2.imwrite(str(RESULTS_DIR / "2_ausgewaehlte_merkmale.png"), image_selected)

    selected_keypoints = [keypoints[i] for i in selected]
    selected_descriptors = descriptors[selected]

    # 1.7 3D-Koordinaten bestimmen ------------------------------------------
    object_points = pixel_to_object([kp.pt for kp in selected_keypoints])
    print("1.7 Beispiel: Merkmal 0 bei Pixel", np.round(selected_keypoints[0].pt, 1),
          "-> 3D-Punkt", np.round(object_points[0], 2), "mm")

    # 1.8 Referenzdaten speichern -------------------------------------------
    # Zeile i in keypoints, descriptors und object_points gehört immer zum
    # Merkmal mit der ID i -> eindeutige Zuordnung bleibt erhalten.
    keypoint_data = np.array([[kp.pt[0], kp.pt[1], kp.size, kp.angle, kp.response, kp.octave]
                              for kp in selected_keypoints], dtype=np.float64)

    fs = cv2.FileStorage(str(REFERENCE_YAML), cv2.FILE_STORAGE_WRITE)
    fs.write("object_width_mm", OBJECT_WIDTH_MM)
    fs.write("object_height_mm", OBJECT_HEIGHT_MM)
    fs.write("px_per_mm", PX_PER_MM)
    fs.write("keypoints_u_v_size_angle_response_octave", keypoint_data)
    fs.write("descriptors", selected_descriptors)
    fs.write("object_points_mm", object_points)
    fs.release()

    with open(REFERENCE_CSV, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file, delimiter=";")
        writer.writerow(["id", "u_px", "v_px", "X_mm", "Y_mm", "Z_mm"])
        for i, (kp, point) in enumerate(zip(selected_keypoints, object_points)):
            writer.writerow([i, round(kp.pt[0], 2), round(kp.pt[1], 2),
                             round(point[0], 3), round(point[1], 3), round(point[2], 3)])

    print("1.8 Gespeichert:", REFERENCE_YAML.name, "und", REFERENCE_CSV.name)
    print("    Bilder in:", RESULTS_DIR)


if __name__ == "__main__":
    main()
