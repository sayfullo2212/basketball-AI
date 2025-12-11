from ultralytics import YOLO
import cv2

# Modelni yuklash
model = YOLO("basketball.pt")   # <-- o'z model faylingiz nomi

# Video yo'li
video_path = r"C:\Users\Sayfullo\Videos\Captures\match3.mp4"

# Video ochish
cap = cv2.VideoCapture(video_path)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Model orqali frame'ni tahlil qilish
    results = model(frame)

    # Annotated frame (chizilgan bounding-box bilan)
    annotated_frame = results[0].plot()

    # Ekranga chiqarish
    cv2.imshow("Basketball Detection", annotated_frame)

    # Esc bosilsa chiqish
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
