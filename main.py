from ultralytics import YOLO
import cv2
import os

# =========================
# 1. Modelni yuklash
# =========================
model = YOLO("basketball.pt")  # O'qitilgan model fayl

# =========================
# 2. Video faylini tanlash
# =========================
video_path = r"C:\Users\Sayfullo\Videos\Captures\match3.mp4"  # O'zgartiring
if not os.path.exists(video_path):
    print("Video topilmadi!")
    exit()

cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("Video ochilmadi!")
    exit()

# =========================
# 3. Video o'lchami va fps
# =========================
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# =========================
# 4. Avtomatik nomlash
# =========================
base_name = os.path.splitext(os.path.basename(video_path))[0]  # masalan: 'mat3'
output_name = f"{base_name}_trajectory.mp4"
counter = 2
while os.path.exists(output_name):
    output_name = f"{base_name}_trajectory{counter}.mp4"
    counter += 1

print(f"Trayektoriya video saqlanadi: {output_name}")

# =========================
# 5. VideoWriter yaratish
# =========================
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_name, fourcc, fps, (width, height))

# =========================
# 6. Ball markazlari ro'yxati
# =========================
points = []

# =========================
# 7. Video frame'larini o'qish va trayektoriyani chizish
# =========================
while True:
    ret, frame = cap.read()
    if not ret or frame is None:
        break

    # YOLO detection
    results = model(frame)[0]

    # Ballni aniqlash va markazini saqlash
    for box in results.boxes:
        cls = int(box.cls[0])
        if cls == 2:  # faraz: ball class index = 2
            x1, y1, x2, y2 = box.xyxy[0]
            cx = int((x1 + x2)/2)
            cy = int((y1 + y2)/2)
            points.append((cx, cy))
            cv2.circle(frame, (cx, cy), 5, (0,0,255), -1)  # qizil nuqta

    # Trayektoriyani chizish
    traj_frame = frame.copy()  # original frame fon bilan
    for i in range(1, len(points)):
        cv2.line(traj_frame, points[i-1], points[i], (255,0,0), 2)  # moviy chiziq

    # Frame'ni ekranga chiqarish
    cv2.imshow("Ball Trajectory", traj_frame)

    # Frame'ni video faylga yozish
    out.write(traj_frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC bosilsa chiqish
        break

# =========================
# 8. Resurslarni yopish
# =========================
cap.release()
out.release()
cv2.destroyAllWindows()

print("Trayektoriya video saqlandi ✅")
