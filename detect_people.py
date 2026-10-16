
from ultralytics import YOLO
import cv2
from pathlib import Path

# Load the existing YOLO model
model = YOLO("yolo11n.pt")

# Select your input video

from ultralytics import YOLO
import cv2
from pathlib import Path

# Project folder containing this Python file
BASE_DIR = Path(__file__).resolve().parent

# Load the YOLO model
model = YOLO(str(BASE_DIR / "yolo11n.pt"))

# All three input videos
video_files = [
    "Copy of Copy of Hafidz 2 Store-Picking and Returning-6.mp4",
    "Copy of Copy of Hafidz 2 Store-Turning to Shelf-8.mp4",
    "Copy of Copy of Hafidz 2 Store-Viewing-9.mp4"
]

output_dir = BASE_DIR / "outputs"
output_dir.mkdir(exist_ok=True)

# Process each video separately
for video_name in video_files:
    video_path = BASE_DIR / video_name

    print(f"\nProcessing: {video_name}")

    if not video_path.exists():
        print("ERROR: Video not found:", video_path)
        continue

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        print("ERROR: Cannot open video:", video_name)
        continue

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if fps <= 0 or width <= 0 or height <= 0:
        print("ERROR: Invalid video properties:", video_name)
        cap.release()
        continue

    # Create a unique output filename for each input
    output_path = output_dir / f"{video_path.stem}_detected.mp4"

    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height)
    )

    if not writer.isOpened():
        print("ERROR: Cannot create output video:", output_path)
        cap.release()
        continue

    person_ids = set()
    frames_processed = 0

    try:
        while True:
            ret, frame = cap.read()

            if not ret:
                break

            # Detect and track people (YOLO class 0 = person)
            results = model.track(
                frame,
                persist=True,
                classes=[0],
                verbose=False
            )

            annotated_frame = results[0].plot()
            writer.write(annotated_frame)
            frames_processed += 1

            boxes = results[0].boxes
            if boxes is not None and boxes.id is not None:
                person_ids.update(boxes.id.int().cpu().tolist())

            if frames_processed % 100 == 0:
                print("Frames processed:", frames_processed)

    finally:
        cap.release()
        writer.release()

    print("Completed:", video_name)
    print("Frames:", frames_processed)
    print("Tracking IDs observed:", sorted(person_ids))
    print("Saved to:", output_path)

print("\nAll videos processed!")


# Open the video
cap = cv2.VideoCapture(str(video_path))

if not cap.isOpened():
    print("Error: Cannot open video.")
    raise SystemExit

# Create output folder
output_dir = Path("outputs")
output_dir.mkdir(exist_ok=True)

output_path = output_dir / "detected_people.mp4"

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

if fps <= 0 or width <= 0 or height <= 0:
    cap.release()
    raise SystemExit("Error: Invalid video properties.")


output_path = output_dir / "detected_people.mp4"

writer = cv2.VideoWriter(
    str(output_path),
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (width, height)
)

if not writer.isOpened():
    print("Error: Could not create output video.")
    cap.release()
    raise SystemExit


person_ids = set()
frames_processed = 0

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # Detect people only (COCO class 0 = person)
    results = model.track(
        frame,
        persist=True,
        classes=[0],
        verbose=False
    )

    annotated_frame = results[0].plot()
    writer.write(annotated_frame)
    frames_processed += 1

    if results[0].boxes is not None:
        boxes = results[0].boxes

        if boxes.id is not None:
            for person_id in boxes.id.int().cpu().tolist():
                person_ids.add(person_id)

    if frames_processed % 100 == 0:
        print("Frames processed:", frames_processed)

cap.release()
writer.release()

print("\nDetection completed!")
print("Frames processed:", frames_processed)
print("Tracking IDs observed:", sorted(person_ids))
print("Output video:", output_path.resolve())
