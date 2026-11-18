# helloWorld – Anleitung

Teil der OpenCV-Inbetriebnahme (MVSR Hausübung 1). Lädt ein als Parameter übergebenes
Bild und zeigt es in einem Fenster an, um eine lauffähige OpenCV-Installation nachzuweisen.

## Voraussetzungen
- Python 3
- OpenCV 4.12 (`pip install opencv-python`)

## Verwendung
```
python main.py <pfad_zum_bild>
```

Mit dem mitgelieferten helloWorld-Bild:
```
python main.py ../media/hello_world/hello_world.png
```

Das Programm öffnet ein in der Größe veränderbares Fenster mit dem übergebenen Bild.
Ein Tastendruck im Fenster beendet das Programm; der zugehörige Tastencode wird in der
Konsole ausgegeben.
