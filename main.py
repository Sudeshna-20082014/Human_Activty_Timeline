import cv2
from ultralytics import YOLO
from fastapi.responses import FileResponse
from fastapi import HTTPException
from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, File, HTTPException, UploadFile

app = FastAPI(title="AI Human Activity Timeline Generator")

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

ALLOWED_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}
MAX_SIZE = 100 * 1024 * 1024  # 100 MB


@app.get("/")
def home():
    return {"message": "Our AI project backend is running!"}


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/videos/upload")
async def upload_video(file: UploadFile = File(...)):
    extension = Path(file.filename or "").suffix.lower()

    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail="Please upload an MP4, AVI, MOV, or MKV video."
        )

    video_id = str(uuid4())
    video_path = UPLOAD_DIR / f"{video_id}{extension}"
    size = 0

    try:
        with video_path.open("wb") as output:
            while True:
                chunk = await file.read(1024 * 1024)

                if not chunk:
                    break

                size += len(chunk)

                if size > MAX_SIZE:
                    raise HTTPException(
                        status_code=413,
                        detail="Video must be smaller than 100 MB."
                    )

                output.write(chunk)

        if size == 0:
            video_path.unlink(missing_ok=True)
            raise HTTPException(
                status_code=400,
                detail="The uploaded video is empty."
            )

        return {
            "video_id": video_id,
            "filename": file.filename,
            "size_bytes": size,
            "status": "uploaded",
            "message": "Video uploaded successfully!"
        }

    except HTTPException:
        video_path.unlink(missing_ok=True)
        raise

    except Exception:
        video_path.unlink(missing_ok=True)
        raise HTTPException(
            status_code=500,
            detail="Could not save the uploaded video."
        )

    finally:
        await file.close()


from pathlib import Path

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)


@app.post("/videos/{video_id}/detect")
def detect_people(video_id: str):
    # Locate the previously uploaded video.
    matches = list(UPLOAD_DIR.glob(f"{video_id}.*"))

    if not matches:
        raise HTTPException(
            status_code=404,
            detail="Uploaded video not found."
        )

    input_path = matches[0]
    output_path = OUTPUT_DIR / f"{video_id}_detected.mp4"

    cap = cv2.VideoCapture(str(input_path))

    if not cap.isOpened():
        raise HTTPException(
            status_code=400,
            detail="Could not open the uploaded video."
        )

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if not fps or fps <= 0 or width <= 0 or height <= 0:
        cap.release()
        raise HTTPException(
            status_code=400,
            detail="Invalid video properties."
        )

    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height)
    )

    if not writer.isOpened():
        cap.release()
        raise HTTPException(
            status_code=500,
            detail="Could not create output video."
        )

    model = YOLO("yolo11n.pt")
    people_seen = set()
    frame_count = 0

    try:
        while True:
            success, frame = cap.read()

            if not success:
                break

            frame_count += 1

            result = model.track(
                frame,
                persist=True,
                classes=[0],
                tracker="bytetrack.yaml",
                verbose=False
            )[0]

            if result.boxes is not None:
                if result.boxes.id is not None:
                    boxes = result.boxes.xyxy.cpu().tolist()
                    ids = result.boxes.id.int().cpu().tolist()

                    for box, person_id in zip(boxes, ids):
                        x1, y1, x2, y2 = map(int, box)
                        people_seen.add(person_id)

                        # Draw a box and tracking ID.
                        cv2.rectangle(
                            frame,
                            (x1, y1),
                            (x2, y2),
                            (0, 255, 0),
                            3
                        )

                        cv2.putText(
                            frame,
                            f"Person {person_id}",
                            (x1, max(25, y1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,
                            (0, 255, 0),
                            2
                        )

            writer.write(frame)

    finally:
        cap.release()
        writer.release()

    if frame_count == 0:
        output_path.unlink(missing_ok=True)
        raise HTTPException(
            status_code=400,
            detail="The video contains no readable frames."
        )

    return {
        "video_id": video_id,
        "frames_processed": frame_count,
        "person_ids_detected": sorted(people_seen),
        "highlighted_video_url": (
            f"/videos/{video_id}/detected-video"
        ),
        "message": "Person detection completed."
    }


@app.get("/videos/{video_id}/detected-video")
def get_detected_video(video_id: str):
    output_path = OUTPUT_DIR / f"{video_id}_detected.mp4"

    if not output_path.is_file():
        raise HTTPException(
            status_code=404,
            detail="Run person detection first."
        )

    return FileResponse(
        path=str(output_path),
        media_type="video/mp4",
        filename="detected_video.mp4"
    )

import csv
from pathlib import Path
from fastapi import HTTPException

BASE_DIR = Path(__file__).resolve().parent
CSV_PATH = BASE_DIR / "outputs" / "activity_timeline.csv"


@app.get("/timeline")
def get_timeline():
    if not CSV_PATH.exists():
        raise HTTPException(
            status_code=404,
            detail="Activity timeline CSV not found. Run activity_timeline.py first."
        )

    with open(CSV_PATH, "r", newline="", encoding="utf-8-sig") as file:
        return list(csv.DictReader(file))