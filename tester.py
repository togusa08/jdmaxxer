import cv2
from ultralytics import YOLO

video_input = "Vid6.mp4"
video_output = "YOLO_TEST.mp4"

model = YOLO("yolov8n.pt")

cap = cv2.VideoCapture(video_input)

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*"mp4v")

writer = cv2.VideoWriter(
    video_output,
    fourcc,
    fps,
    (width, height)
)

frame_idx = 0

while True:

    ret, frame = cap.read()

    if not ret:
        break

    results = model(frame, verbose=False)[0]

    car_count = 0

    if results.boxes is not None:

        for box in results.boxes:

            cls_id = int(box.cls[0])
            conf = float(box.conf[0])

            # COCO car = 2
            if cls_id == 2 and conf >= 0.4:

                car_count += 1

                x1, y1, x2, y2 = map(
                    int,
                    box.xyxy[0]
                )

                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2
                )

    writer.write(frame)

    frame_idx += 1

    if frame_idx % 30 == 0:
        print(
            f"Frame {frame_idx}: "
            f"{car_count} cars detected"
        )

cap.release()
writer.release()

print("DONE")
print("Created:", video_output)