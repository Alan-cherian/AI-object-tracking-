import cv2
from ultralytics import YOLO

# Load YOLOv8 model
model = YOLO('best.pt')
names = model.names

# Debug mouse position
def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        print(f"Mouse moved to: [{x}, {y}]")

cv2.namedWindow("RGB")
cv2.setMouseCallback("RGB", RGB)

# Load video
cap = cv2.VideoCapture("box.mp4")

# Get video properties
frame_width = 1020
frame_height = 500
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use 'avc1' or 'mp4v' for .mp4 files
out = cv2.VideoWriter("output.mp4", fourcc, fps//3, (frame_width, frame_height))

frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    if frame_count % 3 != 0:
        continue

    frame = cv2.resize(frame, (frame_width, frame_height))
    results = model.track(frame, persist=True, classes=[0, 2])  # Only tracking specific classes

    if results[0].boxes.id is not None:
        ids = results[0].boxes.id.cpu().numpy().astype(int)
        boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
        class_ids = results[0].boxes.cls.int().cpu().tolist()

        for track_id, box, class_id in zip(ids, boxes, class_ids):
            x1, y1, x2, y2 = box
            label = names[class_id]
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(frame, label, (x1 + 3, y1 - 7),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

    # Write the frame to output video
    out.write(frame)

    # Display the frame
    cv2.imshow("RGB", frame)
    
    # Press ESC to quit
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()

