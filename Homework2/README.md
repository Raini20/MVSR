# MVSR Hausübung 2 – Pose Estimation

Rainhard Wipp · Python 3, OpenCV 4.12

Das Programm schätzt die Pose eines flachen Objekts (Flyer, 101 × 210 mm) relativ zur Kamera. Es besteht aus drei Skripten, die den drei Punkten der Angabe entsprechen und aufeinander aufbauen:

| Skript | Angabe | verwendet | erzeugt |
|---|---|---|---|
| `schritt1_referenzmerkmale.py` | 1. Features und Deskriptoren extrahieren | Referenzfoto | Referenzbild, `reference_features.yaml`, `reference_points.csv` |
| `schritt2_korrespondenzen.py` | 2. Korrespondenzen wählen | Ergebnis Schritt 1, Video | Kameraparameter, Bilder der Korrespondenzen |
| `schritt3_pose_estimation.py` | 3. 2D–3D-Korrespondenzen und Pose | Funktionen aus Schritt 1 + 2, Video | Pose pro Frame, Ergebnisbilder, Ergebnisvideo, Auswertung |

## Ordnerstruktur

```
MVSR_HÜ2_Wipp-Rainhard/
├── README.md
├── requirements.txt
├── src/
│   ├── schritt1_referenzmerkmale.py
│   ├── schritt2_korrespondenzen.py
│   └── schritt3_pose_estimation.py
├── config/
│   └── camera_parameters.yaml         Kameraparameter (in Schritt 2 bestimmt)
├── data/
│   ├── reference_features.yaml        Keypoints, Deskriptoren und 3D-Punkte
│   └── reference_points.csv           Zuordnung id ↔ Pixel (u, v) ↔ 3D-Punkt (X, Y, Z)
└── media/
    ├── reference/
    │   ├── reference_photo.jpg        Originalfoto des Objekts
    │   └── reference_image.png        frontales Referenzbild (aus Schritt 1)
    ├── video/
    │   └── camera_video.mp4           Kamerabild
    └── results/
        ├── schritt1/                  Merkmale im Referenzbild
        ├── schritt2/                  Korrespondenzen Referenz ↔ Kamerabild
        └── schritt3/                  Pose, Ergebnisvideo, Auswertung
```

## Verwendung

```
pip install -r requirements.txt
cd src
python schritt1_referenzmerkmale.py
python schritt2_korrespondenzen.py
python schritt3_pose_estimation.py
```

Die Skripte werden der Reihe nach ausgeführt. Alle Pfade und Parameter stehen als Konstanten am Anfang der jeweiligen Datei. Schritt 3 zeigt das Ergebnis live in einem Fenster an (`q` beendet das Programm, `--no-gui` schaltet das Fenster ab) und gibt pro Frame die Translation in der Konsole aus:

```
 Frame |  tx [mm]  ty [mm]  tz [mm] |       rvec [rad]        | Inlier/Matches | Fehler
  1232 |    -27.2     79.9    228.6 | [-3.104  0.015  0.085] |    96 / 107    | 0.95 px
```

---

## Schritt 1 – Referenzmerkmale

**Referenzbild.** Der Flyer wurde auf einer dunklen Unterlage fotografiert. Das Foto ist leicht schräg aufgenommen, deshalb wird über die vier Ecken des Flyers eine Homographie berechnet (`getPerspectiveTransform`, `warpPerspective`). Das Ergebnis ist ein frontales Referenzbild mit festem Maßstab von 5 px/mm (505 × 1050 px). Der Maßstab ist so gewählt, dass das Objekt ungefähr so groß ist wie im Video.

**Objektkoordinatensystem.** Ursprung in der linken unteren Ecke, X nach rechts, Y nach oben, Z aus der Oberfläche heraus. Einheit ist mm, und alle Punkte haben Z = 0. Durch das frontale Referenzbild gilt für jedes Merkmal bei Pixel (u, v):

  X = u / 5,  Y = 210 − v / 5,  Z = 0

**Merkmale: SIFT.** SIFT ist skalierungs- und rotationsinvariant. Das ist hier wichtig, weil sich Abstand und Drehung der Kamera im Video ständig ändern. Verwendet werden die Standardparameter nach Lowe, die über `create_sift()` auch in Schritt 2 und 3 identisch zum Einsatz kommen:

| Parameter | Wert | Begründung |
|---|---|---|
| `nfeatures` | 0 | keine Begrenzung, die Auswahl erfolgt gezielt (siehe unten) |
| `nOctaveLayers` | 3 | Skalenstufen pro Oktave, Lowes Kompromiss aus Wiederholbarkeit und Rechenzeit |
| `contrastThreshold` | 0.04 | verwirft kontrastarme Punkte, die stark vom Rauschen abhängen |
| `edgeThreshold` | 10 | verwirft Punkte auf Kanten, weil diese entlang der Kante schlecht lokalisierbar sind |
| `sigma` | 1.6 | Glättung der ersten Oktave |

**Auswahl geeigneter Merkmale** (`2_ausgewaehlte_merkmale.png`: grün = ausgewählt, rot = Rand, magenta = mehrdeutig):
- *Rand:* Merkmale näher als 3 mm (bzw. ihr eigener Radius) am Objektrand werden verworfen, weil ihre Umgebung im Kamerabild Hintergrund enthält.
- *Eindeutigkeit:* Jedes Merkmal wird mit allen anderen Merkmalen des Referenzbilds verglichen. Gibt es an einer anderen Stelle ein sehr ähnliches Merkmal (Deskriptordistanz < 200), etwa durch gleiche Buchstaben oder die gleichförmige Kontur des Drachen, wird es verworfen. Es könnte im Kamerabild nicht eindeutig zugeordnet werden.

Ergebnis: Von 920 SIFT-Merkmalen werden 588 ausgewählt, 23 liegen zu nah am Rand und 309 sind mehrdeutig.

**Speichern.** In `reference_features.yaml` gehört Zeile *i* von Keypoints, Deskriptoren und 3D-Punkten immer zum selben Merkmal. Die `reference_points.csv` zeigt diese Zuordnung lesbar an.

## Schritt 2 – Korrespondenzen

**Kameraparameter.** Die Kalibrierung aus HÜ 1 ist für dieses Video nicht verwendbar, aus zwei Gründen:
- Das Video wurde mit einem anderen Bildausschnitt aufgenommen als die Fotos. Im Videomodus ließ sich die 0,6×-Optik nicht wählen. Laut EXIF stammt das Referenzfoto vom Ultraweitwinkel (13 mm KB-äquivalent), das Video dagegen nicht.
- Die 20 Kalibrierbilder aus HÜ 1 haben unterschiedliche Pixelgrößen (2026 bis 2369 px breit). `calibrateCamera` nimmt aber eine gemeinsame Bildgröße an.

Umgerechnet auf das Video ergibt die HÜ-1-Kalibrierung f ≈ 628 px. Die Kalibrierung mit dem Flyer liefert dagegen f ≈ 970 px. Mit den HÜ-1-Werten wären alle Abstände um rund ein Drittel zu klein, und es gäbe weniger gültige Posen.

Deshalb wird die Kamera in Abschnitt 2.2 **mit dem Flyer selbst kalibriert** (Verfahren nach Zhang mit `calibrateCamera`, genau wie in HÜ 1). Statt der Schachbrettecken dienen die Referenzmerkmale mit ihren bekannten 3D-Punkten (Z = 0) als Kalibrierobjekt. Aus jedem 25. Frame werden die Matches bestimmt. Fehlzuordnungen werden mit einer Homographie samt RANSAC entfernt, denn für eine Ebene muss jede richtige Korrespondenz die Gleichung Bild = H · Objekt erfüllen. Verwendet werden nur Frames mit mindestens 25 Punkten. k3 bleibt fixiert, da die Verzeichnung gering ist und k3 sonst schlecht bestimmbar wäre.

| | Wert |
|---|---|
| verwendete Frames | 47 (unterschiedliche Abstände und Neigungen) |
| fx, fy | 970.4 px, 977.3 px |
| cx, cy | 533.7 px, 719.4 px (Bildmitte: 540, 720) |
| Verzeichnung k1, k2, p1, p2 | −0.0040, −0.0108, −0.0015, −0.0003 |
| RMS-Reprojektionsfehler | 1.34 px |

Der Hauptpunkt liegt nahe der Bildmitte und die Verzeichnung ist sehr klein, beides ist für eine Smartphone-Hauptkamera mit softwareseitiger Korrektur plausibel. Der RMS-Fehler ist höher als beim Schachbrett (HÜ 1: 0.19 px), weil SIFT-Punkte weniger genau lokalisiert sind als subpixelgenaue Schachbrettecken.

Das Kamerabild wird **nicht** entzerrt. Die Verzeichnungskoeffizienten werden direkt an `solvePnPRansac`, `solvePnP` und `projectPoints` übergeben.

**Matching.** `BFMatcher` mit `NORM_L2`, das ist die passende Distanz für SIFT-Deskriptoren. Mit `knnMatch(k=2)` werden für jedes Referenzmerkmal die zwei ähnlichsten Merkmale im Kamerabild gesucht.

**Auswahl der Korrespondenzen: Ratio-Test nach Lowe.** Ein Match wird nur akzeptiert, wenn d₁ < 0.75 · d₂ gilt, wenn also das beste Match deutlich besser ist als das zweitbeste. Eine feste Distanzschwelle hängt stark von Bild und Beleuchtung ab, das Verhältnis dagegen kaum. `ohne_filter_frame_0100.jpg` zeigt im Vergleich, wie viele falsche Zuordnungen ohne Filter entstehen. Weil in Schritt 1 nur Merkmale mit bekannter 3D-Koordinate gespeichert wurden, gehört jedes akzeptierte Match zu einem bekannten 3D-Punkt.

## Schritt 3 – Pose Estimation

1. Für jeden Frame (jeder 2. Frame des 60-fps-Videos) werden Merkmale extrahiert und wie in Schritt 2 gematcht.
2. **2D–3D-Korrespondenzen:** Aus den Matches entstehen zwei Listen in gleicher Reihenfolge, `object_points[i]` (3D-Punkt des Referenzmerkmals) und `image_points[i]` (Bildpunkt im Kamerabild).
3. **Pose** in drei Teilschritten (wie in der Vorlesung: RANSAC → Neuberechnung mit allen Inliern → Verfeinerung):
   - **a) RANSAC:** `solvePnPRansac` trennt Inlier von Outliern, also falsche Matches, die den Ratio-Test überstanden haben. Parameter: Inlier-Schwelle 4 px, 500 Iterationen, Konfidenz 0.99.
   - **b) Neuberechnung:** Mit allen Inliern wird die Pose mit `solvePnP` und `SOLVEPNP_IPPE` neu berechnet, einem Verfahren speziell für ebene Objekte.
   - **c) Verfeinerung:** Levenberg–Marquardt mit `solvePnPRefineLM`, das den Reprojektionsfehler minimiert.

   **Warum `SOLVEPNP_AP3P` in RANSAC?** Jede RANSAC-Hypothese wird aus einer kleinen Punktmenge berechnet. Mit AP3P sind das 4 Punkte. Bei allen anderen Flags (ITERATIVE, EPNP, SQPNP, IPPE) verwendet OpenCV dafür intern EPnP mit 5 Punkten, und das ist bei ebenen Objekten instabil. Anzahl gefundener Inlier im Test:

   | Frame | Matches | Homographie-Inlier (Vergleich) | RANSAC mit IPPE/ITERATIVE | RANSAC mit AP3P |
   |---|---|---|---|---|
   | 100 | 67 | 57 | 17 | 53 |
   | 450 | 112 | 89 | 14 | 91 |
   | 500 | 88 | 66 | 10 | 57 |
   | 1000 | 68 | 48 | 46 | 47 |

   **Warum Teilschritt b)?** In OpenCV 4.12 berechnet `solvePnPRansac` mit AP3P zwar die richtigen Inlier, die zurückgegebene Pose passt aber teilweise nicht dazu. In Frame 300 meldete RANSAC 79 Inlier, die zurückgegebene Pose erklärte davon nur 3. Wird die Pose stattdessen selbst aus allen Inliern berechnet, steigt der Anteil gültiger Posen im Video von 47 % auf 97 %.
4. **Prüfung:** Bei weniger als 15 Inliern gilt die Pose als unzuverlässig und wird nicht ausgegeben.
5. **Ausgabe:** Translation (tx, ty, tz) in mm sowie die Rotation als Rotationsvektor `rvec` (Achse · Winkel in rad, mit `cv2.Rodrigues` in die Rotationsmatrix umrechenbar).
6. **Visualisierung:** Objektumriss und Objektkoordinatensystem (X rot, Y grün, Z blau) im Kamerabild, daneben das Referenzbild mit den Korrespondenzen (grün = Inlier, rot = Outlier).

## Ergebnisse und Beurteilung

Video: 1233 Frames (60 fps, 1080 × 1440 px), davon wird jeder 2. Frame verarbeitet, also 617 Frames. Alle Werte gelten für OpenCV 4.12.

| Kennzahl | Ergebnis |
|---|---|
| Frames mit gültiger Pose | 601 von 617 (97 %) |
| Abstand Kamera – Objektursprung | 139 bis 293 mm |
| Neigung (Objektnormale zu Blickrichtung) | 1° bis 44° |
| RANSAC-Inlier pro Frame | Median 72, Inlier-Anteil Median 82 % |
| mittlerer Reprojektionsfehler der Inlier | 0.8 bis 2.2 px, Median 1.3 px |

`auswertung.png` zeigt Translation, Anzahl der Matches und Inlier sowie die Neigung über das ganze Video. Bereiche ohne gültige Pose sind rot hinterlegt.

**Wann funktioniert die Posenschätzung zuverlässig?**
- Über fast das ganze Video, bei Abständen von 14 bis 29 cm und Neigungen bis etwa 40°. Translation und Neigung verlaufen glatt und ohne Sprünge, obwohl jeder Frame unabhängig von den vorherigen berechnet wird.
- Ein Reprojektionsfehler um 1 px zeigt, dass Pose und Kameramodell gut zu den Messungen passen. In den Ergebnisbildern liegt der eingezeichnete Objektumriss genau auf dem Flyer.
- Die Bewegungen im Video (näher / weiter weg, Kippen) sind in tz und in der Neigung klar nachvollziehbar.

**Wann schlägt sie fehl?**
- Nur in Frame 764–796 (und einzeln in Frame 844) gibt es keine Pose. Dort treffen mehrere ungünstige Bedingungen zusammen: Der Flyer ist mit etwa 44° am stärksten geneigt, die Kamera ist nah dran, ein Teil des Flyers liegt außerhalb des Bildes, und der Flyer ist überbelichtet.
- **Starke Neigung:** Die lokalen Bildbereiche sind perspektivisch stark verzerrt. SIFT-Deskriptoren sind nur gegenüber Rotation und Skalierung invariant, nicht gegenüber starker perspektivischer Verzerrung. Die Zahl der Matches nach dem Ratio-Test sinkt mit zunehmender Neigung deutlich (siehe `auswertung.png`) und fällt in diesem Abschnitt auf 17 bis 33.
- **Überbelichtung:** Die automatische Belichtung stellt auf den dunklen Stoff ein, wodurch der helle Flyer ausbrennt und die rosa Drachentextur an Kontrast verliert.
- **Objekt teilweise außerhalb des Bildes:** Fehlende Bereiche bedeuten weniger Korrespondenzen.
- Bleiben weniger als 15 Inlier übrig, gibt das Programm bewusst keine Pose aus, statt eine unzuverlässige anzuzeigen.

**Verbesserungsmöglichkeiten:** Belichtung sperren, um das Ausbrennen zu vermeiden. Für starke Neigungen zusätzliche, schräge Referenzansichten oder affin-invariante Merkmale (z. B. ASIFT) einsetzen. Mit einem Schachbrett im Videomodus kalibrieren, um noch genauere Kameraparameter zu erhalten.
