"""
OpenCV Hello World.
Aufruf: python main.py <pfad_zum_bild>
"""
import sys
import cv2

def main():
    if len(sys.argv) != 2:
        print("Verwendung: python main.py <pfad_zum_bild>")
        sys.exit(1)

    image_path = sys.argv[1]
    img = cv2.imread(image_path)
    if img is None:
        print(f"Fehler: Bild konnte nicht geladen werden: {image_path}")
        sys.exit(1)

    cv2.namedWindow("helloWorld", 0)
    cv2.imshow("helloWorld", img)
    key = cv2.waitKey(0)
    print(f"Taste gedrueckt: {key}")

if __name__ == "__main__":
    main()
