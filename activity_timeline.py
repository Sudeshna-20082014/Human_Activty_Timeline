

from ultralytics import YOLO
import cv2
import csv
import math
from pathlib import Path
from collections import deque

# Folder containing this Python file
BASE_DIR = Path(__file__).resolve().parent

# Input videos
video_files = [
    "Copy of Copy of Hafidz 2 Store-Picking and Returning-6.mp4",
    "Copy of Copy of Hafidz 2 Store-Turning to Shelf-8.mp4",
    "Copy of Copy of Hafidz 2 Store-Viewing-9.mp4"
]

# Output folder and CSV file
output_dir = BASE_DIR / "outputs"
output_dir.mkdir(exist_ok=True)
csv_path = output_dir / "activity_timeline.csv"

all_events = []

# Settings
MOVEMENT_THRESHOLD = 30
HISTORY_LENGTH = 10
CONFIRMATION_FRAMES = 5


def format_time(seconds):
    """Convert seconds to HH:MM:SS."""
    seconds = max(0, int(seconds))
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60
    return f"{hours:02}:{minutes:02}:{secs:02}"


def save_event(video_name, person_id, activity,
               start_frame, end_frame, fps):
    """Save one activity interval."""
    start_seconds = start_frame / fps
    end_seconds = max(start_seconds, end_frame / fps)

    all_events.append({
        "video": video_name,
        "person_id": person_id,
        "activity": activity,
        "start_time": format_time(start_seconds),
        "end_time": format_time(end_seconds),
        "start_seconds": round(start_seconds, 2),
        "end_seconds": round(end_seconds, 2)
    })


def process_video(video_name):
    video_path = BASE_DIR / video_name

    if not video_path.exists():
        print("Video not found:", video_path)
        return

    print("\nAnalyzing:", video_name)

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        print("Cannot open video:", video_name)
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0

    model = YOLO(str(BASE_DIR / "yolo11n.pt"))

    people = {}
    frame_number = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model.track(
            frame,
            persist=True,
            classes=[0],
            verbose=False
        )

        boxes = results[0].boxes

        if boxes is not None and boxes.id is not None:
            coordinates = boxes.xyxy.cpu().tolist()
            ids = boxes.id.int().cpu().tolist()

            for box, person_id in zip(coordinates, ids):
                x1, y1, x2, y2 = box

                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2

                if person_id not in people:
                    people[person_id] = {
                        "history": deque(maxlen=HISTORY_LENGTH),
                        "activity": None,
                        "candidate_activity": None,
                        "candidate_count": 0,
                        "start_frame": frame_number,
                        "last_seen": frame_number
                    }

                person = people[person_id]
                history = person["history"]

                # Estimate movement across several frames
                moving = False

                if len(history) >= 5:
                    old_x, old_y = history[0]

                    distance = math.hypot(
                        center_x - old_x,
                        center_y - old_y
                    )

                    moving = distance > MOVEMENT_THRESHOLD

                new_activity = (
                    "Walking" if moving else "Standing/Stopped"
                )

                # Initialize the person's first activity
                if person["activity"] is None:
                    person["activity"] = new_activity
                    person["start_frame"] = frame_number

                # Activity matches the current confirmed activity
                elif new_activity == person["activity"]:
                    person["candidate_activity"] = None
                    person["candidate_count"] = 0

                # Check whether a different activity persists
                else:
                    if person["candidate_activity"] == new_activity:
                        person["candidate_count"] += 1
                    else:
                        person["candidate_activity"] = new_activity
                        person["candidate_count"] = 1

                    # Confirm change after consecutive detections
                    if person["candidate_count"] >= CONFIRMATION_FRAMES:
                        save_event(
                            video_name,
                            person_id,
                            person["activity"],
                            person["start_frame"],
                            frame_number,
                            fps
                        )

                        person["activity"] = new_activity
                        person["start_frame"] = frame_number
                        person["candidate_activity"] = None
                        person["candidate_count"] = 0

                person["last_seen"] = frame_number
                history.append((center_x, center_y))

        frame_number += 1

        if frame_number % 100 == 0:
            print("Frames analyzed:", frame_number)

    # Save the final confirmed activity for each person
    for person_id, person in people.items():
        if person["activity"] is not None:
            save_event(
                video_name,
                person_id,
                person["activity"],
                person["start_frame"],
                person["last_seen"],
                fps
            )

    cap.release()

    print("Video analyzed:", video_name)
    print("Frames analyzed:", frame_number)
    print("People tracked:", len(people))


# Analyze all videos
for video_name in video_files:
    process_video(video_name)

# Sort by video, timestamp, and person
all_events.sort(
    key=lambda event: (
        event["video"],
        event["start_seconds"],
        event["person_id"]
    )
)

# Export timeline to CSV
fieldnames = [
    "video",
    "person_id",
    "activity",
    "start_time",
    "end_time",
    "start_seconds",
    "end_seconds"
]

with open(csv_path, "w", newline="", encoding="utf-8-sig") as file:
    writer = csv.DictWriter(file, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(all_events)

print("\nActivity timeline generated!")
print("Total timeline events:", len(all_events))
print("Saved to:", csv_path)

print("\nFirst 15 timeline events:")

for event in all_events[:15]:
    print(
        event["start_time"],
        "| Person", event["person_id"],
        "|", event["activity"],
        "|", event["video"]
    )
