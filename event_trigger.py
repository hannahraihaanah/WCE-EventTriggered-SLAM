from ultralytics import YOLO
import cv2
import os

# ---------------------------------------------------
# 1. Setup
# ---------------------------------------------------
ENDOSLAM_EVENT_DIR = os.path.expanduser("~/EndoSLAM/event_frames")
os.makedirs(ENDOSLAM_EVENT_DIR, exist_ok=True)

model = YOLO("./#Model_Weights/best.pt")

CONF_THRESHOLD = 0.60
EVENT_WINDOW = 15

frame_buffer = []
event_frames = []
frame_index = 0

cap = cv2.VideoCapture("capsule_video.mp4")
print("\n📌 EVENT TRIGGER STARTED\n")

# ---------------------------------------------------
# 2. Scan video until first bleeding detected
# ---------------------------------------------------
triggered = False

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_index += 1
    frame_buffer.append(frame.copy())

    if len(frame_buffer) > EVENT_WINDOW:
        frame_buffer.pop(0)

    results = model(frame, verbose=False)

    for box in results[0].boxes:
        conf = float(box.conf)
        if conf > CONF_THRESHOLD:
            print(f"🚨 FIRST BLEEDING at frame {frame_index} | conf={conf:.2f}")
            triggered = True
            break

    if triggered:
        # Save pre-event frames
        event_frames.extend(frame_buffer)

        # Collect next 15 frames
        for _ in range(EVENT_WINDOW):
            ret2, next_frame = cap.read()
            if ret2:
                event_frames.append(next_frame)
        break

cap.release()

# ---------------------------------------------------
# 3. Re-scan ALL 21 frames for bleeding
# ---------------------------------------------------
print(f"\n📊 Total frames collected: {len(event_frames)}")
print("\n🔍 Re-scanning all frames for bleeding...\n")

all_detections = []
lesion_frames = []

for i, ef in enumerate(event_frames):
    results = model(ef, verbose=False)
    for box in results[0].boxes:
        conf = float(box.conf)
        if conf > CONF_THRESHOLD:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            all_detections.append({
                'frame_idx': i,
                'conf': conf,
                'bbox': (x1, y1, x2, y2)
            })
            if i not in lesion_frames:
                lesion_frames.append(i)
            print(f"🚨 Frame {i:02d} | conf={conf:.2f} | bbox=({x1},{y1},{x2},{y2})")

print(f"\n📍 Total bleeding detections: {len(all_detections)}")
print(f"📍 Bleeding frames: {lesion_frames}")

# ---------------------------------------------------
# 4. Save frames to EndoSLAM
# ---------------------------------------------------
for i, ef in enumerate(event_frames):
    out_path = os.path.join(ENDOSLAM_EVENT_DIR, f"{i:06d}.png")
    cv2.imwrite(out_path, ef)

# Save lesion frame indexes
lesion_file = os.path.join(ENDOSLAM_EVENT_DIR, "lesion_frames.txt")
with open(lesion_file, "w") as f:
    f.write(" ".join(map(str, lesion_frames)))
print("✔ Saved lesion_frames.txt")

# Save ALL detections
all_bbox_file = os.path.join(ENDOSLAM_EVENT_DIR, "all_detections.txt")
with open(all_bbox_file, "w") as f:
    for d in all_detections:
        x1, y1, x2, y2 = d['bbox']
        f.write(f"{d['frame_idx']} {x1} {y1} {x2} {y2} {d['conf']:.4f}\n")
print("✔ Saved all_detections.txt")

# Save BEST detection (highest confidence)
if all_detections:
    best = max(all_detections, key=lambda d: d['conf'])
    bbox_file = os.path.join(ENDOSLAM_EVENT_DIR, "lesion_bbox.txt")
    with open(bbox_file, "w") as f:
        x1, y1, x2, y2 = best['bbox']
        f.write(f"{x1} {y1} {x2} {y2} {best['conf']:.4f} {best['frame_idx']}\n")
    print(f"✔ Saved lesion_bbox.txt: best=frame {best['frame_idx']} conf={best['conf']:.2f}")

print("✔ Saved clean RGB frames")
print("\n🎯 READY FOR ENDOSLAM\n")
