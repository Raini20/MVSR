# MVSR Hausübung 3 – Visual Servoing

Rainhard Wipp · Python 3, OpenCV 4.12

Das Programm führt die Kamera eines Videos in Richtung einer Sollansicht. Für jeden Frame berechnet es die Geschwindigkeit `v = (vx vy vz wx wy wz)`, mit der sich die Kamera bewegen müsste. Verwendet wird **bildbasiertes Visual Servoing (IBVS)** auf dem Flyer aus HÜ 2. Die Pose Estimation aus HÜ 2 liefert dafür die Tiefe der Merkmale. Das Programm ist eigenständig, alle benötigten Daten liegen in diesem Ordner.

## Ordnerstruktur

```
Homework3/
├── README.md
├── requirements.txt
├── src/
│   └── visual_servoing.py
├── config/
│   └── camera_parameters.yaml        Kameraparameter (aus HÜ 2 für dieses Video bestimmt)
└── media/
    ├── reference/
    │   └── reference_image.png       Referenzbild = Sollansicht (frontal, 505 × 1050 px)
    ├── video/
    │   └── camera_video.mp4          Testvideo (1080 × 1440 px, 60 fps, 1233 Frames)
    └── results/
        ├── visual_servoing_video.mp4        Ergebnisvideo
        ├── visual_servoing_frame_*.jpg      Einzelbilder
        ├── geschwindigkeiten.csv            v, Fehler und Abstand zur Sollpose je Frame
        └── auswertung.png                   Verlauf über das Video
```

## Verwendung

```
pip install -r requirements.txt
cd src
python visual_servoing.py -reference ../media/reference/reference_image.png -video ../media/video/camera_video.mp4
```

| Parameter | Bedeutung |
|---|---|
| `-reference <Bild>` | Referenzbild = Sollansicht (Pflicht) |
| `-video <Video>` | Video mit den Istansichten (Pflicht) |
| `-camera <yaml>` | Kameraparameter, Standard `config/camera_parameters.yaml` |
| `-object_size <B> <H>` | Objektgröße in mm, Standard `101 210` (Flyer) |
| `-output <Ordner>` | Ergebnisordner, Standard `media/results` |
| `--no-gui` | kein Anzeigefenster (mit Fenster beendet `q` das Programm) |

**Anforderungen an das Referenzbild:** Es zeigt das Objekt frontal und füllt es genau aus (Bildränder = Objektränder). Daraus ergeben sich Maßstab und 3D-Punkte (siehe Schritt 1). Das Bild `reference_image.png` ist das frontale Bild aus HÜ 2. Referenzbild und Video sind austauschbar, solange das Objekt flach ist und `-object_size` sowie die Kameraparameter dazu passen.

Pro Frame (jeder 2. des 60-fps-Videos) wird im Terminal eine Zeile ausgegeben:

```
 Frame |   vx      vy      vz  [mm/s] |   wx      wy      wz [rad/s] | Fehler [px] | Inlier
   600 |    28.4    -2.9    11.2    |   0.157  -0.103   0.008    |    185.0    |   62
```

## Verarbeitungskette

| Schritt | Angabe | Inhalt |
|---|---|---|
| 1.1 | Sollzustand | Referenzbild laden, SIFT-Merkmale extrahieren, geeignete auswählen (Rand, Eindeutigkeit), 3D-Punkte bestimmen |
| 1.2 | Sollzustand | Sollpose der Kamera festlegen, daraus die Soll-Bildpunkte s\* |
| 1.3 | Istzustand | je Frame: SIFT, Matching mit Ratio-Test, Pose (PnP) und gemessene Bildpunkte s |
| 2 | Regelfehler, Geschwindigkeit | e = s − s\*, v = −λ L⁺ e |
| 3 | Video | Ausgabe, Visualisierung, Auswertung, Frames ohne Zustand |

Die Schritte 1.1 und 1.3 sind die Pose Estimation aus HÜ 2 (SIFT-Parameter nach Lowe, `BFMatcher` mit kNN und Ratio 0.75, `solvePnPRansac` mit AP3P für die Inlier, `solvePnP` mit IPPE aus allen Inliern, `solvePnPRefineLM`, mindestens 15 Inlier). Der Unterschied: Die Referenzmerkmale werden beim Start direkt aus dem Referenzbild berechnet, damit das Bild ein Parameter sein kann. Die Kamerakalibrierung aus HÜ 1 passt nicht zum Video, deshalb liegt hier die Kalibrierung aus HÜ 2.

## Koordinatensysteme

- **Objektkoordinatensystem:** Ursprung links unten am Flyer, X nach rechts, Y nach oben, Z aus der Oberfläche heraus, Einheit mm, alle Merkmale haben Z = 0. Für ein Merkmal bei Pixel (u, v) im Referenzbild gilt X = u / s, Y = H − v / s mit s = Pixel pro mm (hier 5).
- **Kamerakoordinatensystem (OpenCV):** Ursprung im Projektionszentrum, x nach rechts, y nach unten, z in Blickrichtung. Die Pose (R, t) aus `solvePnP` beschreibt das Objekt im Kamerasystem: X_cam = R · X_obj + t.
- Alle Geschwindigkeiten sind im **Kamerakoordinatensystem** angegeben.

## Sollzustand

Das Referenzbild beschreibt die gewünschte Ansicht: Die Kamera blickt **frontal** auf den Flyer, dieser ist im Bild **zentriert**, und er erscheint mit dem Maßstab des Referenzbilds (5 px/mm). Daraus folgt die Sollpose des Objekts im Kamerasystem

- R\* = diag(1, −1, −1) (Kamera-x = Objekt-X, Kamera-y = −Objekt-Y, Kamera-z = −Objekt-Z)
- Sollabstand Z\* = f / s = 975 px / 5 px/mm = **195 mm** (f = Mittel aus fx und fy)
- t\* so, dass das Objektzentrum auf der optischen Achse bei Z\* liegt: t\* = (0, 0, Z\*) − R\* · (W/2, H/2, 0)

Die Soll-Bildpunkte s\* sind die Projektion der 3D-Punkte der Merkmale mit dieser Sollpose. Sie zeigen, wo die Merkmale im Bild liegen würden, wenn die Kamera die Sollpose hätte.

## Verfahren und Regelfehler: IBVS

Verwendet werden alle RANSAC-Inlier eines Frames (typisch 60 bis 120 Punkte). Für jeden Punkt i werden normierte Bildkoordinaten (x, y) = (X/Z, Y/Z) im Kamerasystem benutzt. Die Verzeichnung wird dafür mit `undistortPoints` entfernt.

- **Istzustand** s: gemessene Bildpunkte der Inlier.
- **Sollzustand** s\*: Bildpunkte derselben 3D-Punkte bei der Sollpose.
- **Regelfehler** e = s − s\*, ein Vektor mit 2N Werten (je Punkt ein x- und y-Fehler, dimensionslos). Zur Anschauung wird der mittlere Abstand der Punkte zu ihrer Sollposition in Pixel ausgegeben (`Fehler [px]`, Mittel über alle Inlier mal Brennweite).
- **Bild-Jacobi-Matrix** L (2N × 6): je Punkt zwei Zeilen

  ```
  [ -1/Z    0     x/Z    x·y     -(1+x²)    y ]
  [  0    -1/Z    y/Z   1+y²     -x·y      -x ]
  ```

  Sie verknüpft die Bildbewegung ṡ = L · v mit der Kamerageschwindigkeit v. Die Tiefe Z jedes Punkts folgt aus der aktuellen Pose (Z = dritte Komponente von R · X_obj + t). Das ist die Verbindung zur Pose Estimation aus HÜ 2: Beim bildbasierten Verfahren ist die Tiefe der einzige Wert, der aus der Pose benötigt wird.
- **Regelgesetz** (exponentieller Abfall des Fehlers): **v = −λ · L⁺ · e**, mit der Pseudoinversen L⁺ (`np.linalg.pinv`, Kleinste-Quadrate-Lösung über alle Punkte) und λ = 1 s⁻¹.

Weil e im Sollzustand null wird, geht auch v gegen null.

**Warum IBVS?** Der Fehler wird direkt im Bild definiert. Die Sollposition der Merkmale ist bekannt, die Pose wird nur für die Tiefe benötigt, und ihr Fehler wirkt sich deshalb nur schwach auf v aus. Außerdem bleiben die Merkmale beim Regeln tendenziell im Bild. Dass dazu viele Punkte verwendet werden, mittelt das Rauschen der einzelnen SIFT-Punkte.

Zur Beurteilung (nicht Teil des Regelgesetzes) wird außerdem der Abstand der aktuellen Kamera zur Sollpose berechnet: die Verschiebung dx, dy, dz in mm (vom aktuellen Kameraursprung zum Soll-Kameraursprung, im aktuellen Kamerasystem) und der Drehwinkel in Grad. Sie stehen in `geschwindigkeiten.csv` und in `auswertung.png`.

## Ausgabe: Bedeutung, Einheit, Vorzeichen von v

`v = (vx vy vz wx wy wz)` ist die **Geschwindigkeit der Kamera im Kamerakoordinatensystem** (x rechts, y unten, z nach vorn), mit der sich die Kamera der Sollansicht nähert, wenn man sie damit bewegt.

| Komponente | Einheit | positives Vorzeichen bedeutet |
|---|---|---|
| vx | mm/s | Kamera bewegt sich nach rechts (im Bild nach rechts) |
| vy | mm/s | Kamera bewegt sich nach unten (im Bild nach unten) |
| vz | mm/s | Kamera bewegt sich in Blickrichtung, also zum Objekt hin |
| wx | rad/s | Drehung um die x-Achse (Rechte-Hand-Regel: Blickrichtung kippt nach oben) |
| wy | rad/s | Drehung um die y-Achse (Blick schwenkt nach rechts) |
| wz | rad/s | Drehung um die optische Achse (Kamera rollt im Uhrzeigersinn, wie das Bild gesehen) |

Die Translation ist in mm/s, weil die Tiefe Z in mm in L eingeht. Die Rotation ist in rad/s. Da das Video nur abgespielt wird und keine Kamera bewegt, ist v der Befehl, den ein Regler an den Roboter geben würde. Die tatsächliche Größe hängt von λ ab. Ein größeres λ regelt schneller (v wächst proportional), λ = 1 s⁻¹ entspricht in etwa „Abstand zur Sollposition in 1 s ausgleichen“.

**Plausibilität:** Die Translation von v zeigt zur Sollposition: Der Kosinus des Winkels zwischen v und der Verschiebung zur Sollpose hat über alle Frames mit Zustand den Median 0.93 und das Minimum 0.73. Im Video kommt die Kamera dem Sollzustand um Frame 1140 am nächsten (Bildfehler 13 px). Dort sinkt v auf wenige mm/s bzw. 0.1 rad/s, die Geschwindigkeit geht in der Nähe des Sollzustands also gegen null.

## Visualisierung

Im Ergebnisvideo und in den Einzelbildern `visual_servoing_frame_*.jpg`:

- **gelber Umriss** und Text: aktuell geschätzte Pose des Flyers (Istzustand)
- **grüner Umriss:** Flyer in der Sollpose (Sollzustand), projiziert in das aktuelle Bild
- **rote Punkte:** gemessene Merkmale s, **grüne Punkte:** Sollpositionen s\*, **weiße Linien:** Regelfehler je Merkmal
- **gelber Pfeil** in der Bildmitte: Translation (vx, vy) als Bewegungsrichtung der Kamera im Bild
- **sechs Balken** unten links: vx, vy, vz (Vollausschlag ±150 mm/s) und wx, wy, wz (Vollausschlag ±0.5 rad/s)
- Text: Frame, v in mm/s, w in rad/s, Bildfehler in px, Anzahl der Inlier

`auswertung.png` zeigt den Verlauf von v, den Bildfehler und den Abstand zur Sollpose über das ganze Video.

## Frames ohne verlässlichen Zustand

Gibt es weniger als 15 RANSAC-Inlier, wird die Pose und damit auch die Tiefe verworfen und es wird **keine Geschwindigkeit aus diesem Frame berechnet**. Das Terminal zeigt `keine verlässliche Pose ... -> v = 0 (Stopp)`. Als Sicherheitsbefehl gilt in diesem Fall v = 0, die Kamera bleibt also stehen. Im Bild steht „v = 0 (Stopp)“ in Rot, und der Frame ist in `geschwindigkeiten.csv` mit `zustand_ok = 0` markiert (Werte leer) und in `auswertung.png` rot hinterlegt.

## Ergebnisse und Beurteilung

Video: 1233 Frames, verarbeitet wird jeder 2. Frame, also 617 Frames. Alle Werte gelten für OpenCV 4.12.

| Kennzahl | Ergebnis |
|---|---|
| Frames mit verlässlichem Zustand | 601 von 617 (97.4 %) |
| Frames ohne Zustand | 764 bis 796 und 844 (v = 0) |
| mittlerer Bildfehler | Median 137 px, Minimum 13 px (Frame 1140) |
| Translationsgeschwindigkeit \|v\| | Median 73 mm/s |
| Rotationsgeschwindigkeit \|w\| | Median 0.31 rad/s |

**Wann funktioniert das Verfahren zuverlässig?**
- Wenn die Pose zuverlässig ist: Abstand 14 bis 29 cm, Neigung bis etwa 40°, mindestens 15 Inlier. Dann verlaufen v und der Fehler glatt, obwohl jeder Frame unabhängig berechnet wird.
- Nahe am Sollzustand (Frame 1130 bis 1150): v wird klein, und vz wechselt das Vorzeichen, wenn die Kamera am Sollzustand vorbeifährt.
- Mit vielen Punkten ist das Ergebnis robust gegenüber einzelnen falschen Matches.

**Wo gibt es Schwierigkeiten?**
- **Starke Neigung, Überbelichtung, Objekt angeschnitten** (Frame 764 bis 796, 844): zu wenige Matches und Inlier, es gibt keinen Zustand. Das ist dieselbe Ursache wie in HÜ 2.
- **Tiefe aus der Pose:** Ein Posenfehler (vor allem bei starker Neigung) verfälscht Z und damit L. v wird dann falsch skaliert. Die Richtung hängt dagegen hauptsächlich vom Bildfehler e ab.
- **Rauschen:** v wird nicht gefiltert. Bei wenigen Inliern (zum Beispiel um Frame 500) schwankt v sichtbar von Frame zu Frame. Ein Tiefpass auf v wäre ein möglicher Ausbau.
- **Große Fehler:** Bei einer großen Drehung zwischen Ist- und Sollansicht (hier bis 45°) liefert IBVS keine geradlinige Kamerabewegung mehr, weil L nur lokal gilt. Die Translationsrichtung von v passt trotzdem (Kosinus ≥ 0.73, siehe oben).
- **Ebenes Objekt:** Alle Punkte liegen in einer Ebene. Die Bewegung ist damit gut bestimmt, solange die Ebene nicht fast parallel zur Blickrichtung steht.
- **Keine Regelschleife:** Das Video ist aufgezeichnet, die berechnete Bewegung wirkt nicht auf die Kamera zurück. Sie zeigt, wie ein Roboter die Kamera bewegen würde.
