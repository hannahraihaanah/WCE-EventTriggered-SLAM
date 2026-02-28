import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

RGB_DIR = "event_frames"
DEPTH_FILE = "results/predictions.npy"
DETECTIONS_FILE = "event_frames/all_detections.txt"

# Load RGB frames
rgb_files = sorted([f for f in os.listdir(RGB_DIR) if f.endswith(".png")])
print("RGB files:", len(rgb_files))

# Load Depth
depth = np.load(DEPTH_FILE)
print("Depth maps:", depth.shape[0])

# Fix mismatch
if depth.shape[0] > len(rgb_files):
    depth = depth[:len(rgb_files)]
elif len(rgb_files) > depth.shape[0]:
    rgb_files = rgb_files[:depth.shape[0]]

N = len(rgb_files)

# Load all detections
detections = {}
with open(DETECTIONS_FILE, "r") as f:
    for line in f:
        vals = line.strip().split()
        if len(vals) >= 6:
            idx = int(vals[0])
            x1, y1, x2, y2 = int(vals[1]), int(vals[2]), int(vals[3]), int(vals[4])
            conf = float(vals[5])
            if idx not in detections:
                detections[idx] = []
            detections[idx].append((x1, y1, x2, y2, conf))

print(f"Bleeding frames: {sorted(detections.keys())}")

# Create figure
fig, axes = plt.subplots(N, 3, figsize=(14, 4 * N))
plt.subplots_adjust(hspace=0.4)

for i, fname in enumerate(rgb_files):
    # Load RGB
    rgb = cv2.imread(os.path.join(RGB_DIR, fname))
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

    # Load Depth
    d = depth[i]
    d_norm = (d - d.min()) / (d.max() - d.min() + 1e-6)
    d_resized = cv2.resize(d_norm, (rgb.shape[1], rgb.shape[0]))
    d_color = cv2.applyColorMap((d_resized * 255).astype("uint8"), cv2.COLORMAP_JET)
    d_color = cv2.cvtColor(d_color, cv2.COLOR_BGR2RGB)

    # Overlay
    overlay = cv2.addWeighted(rgb, 0.6, d_color, 0.4, 0)

    # Draw bounding boxes on RGB copy
    rgb_annotated = rgb.copy()
    overlay_annotated = overlay.copy()

    if i in detections:
        for (x1, y1, x2, y2, conf) in detections[i]:
            cv2.rectangle(rgb_annotated, (x1, y1), (x2, y2), (255, 0, 0), 3)
            cv2.putText(rgb_annotated, f'{conf:.2f}',
                       (x1, max(y1-8, 10)),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            cv2.rectangle(overlay_annotated, (x1, y1), (x2, y2), (255, 0, 0), 2)

    # Plot
    axes[i, 0].imshow(rgb_annotated)
    axes[i, 1].imshow(d_color)
    axes[i, 2].imshow(overlay_annotated)

    # Titles and borders
    if i in detections:
        n_det = len(detections[i])
        avg_conf = sum(c for _,_,_,_,c in detections[i]) / n_det
        axes[i, 0].set_title(f"Frame {i} — BLEEDING ({n_det} detections)",
                             color='red', fontweight='bold', fontsize=9)
        axes[i, 1].set_title(f"Depth — avg conf={avg_conf:.2f}",
                             color='red', fontsize=9)
        axes[i, 2].set_title(f"Overlay — BLEEDING",
                             color='red', fontsize=9)
        for j in range(3):
            for spine in axes[i, j].spines.values():
                spine.set_edgecolor('red')
                spine.set_linewidth(3)
    else:
        axes[i, 0].set_title(f"Frame {i} — Normal", fontsize=9)
        axes[i, 1].set_title("Depth", fontsize=9)
        axes[i, 2].set_title("Overlay", fontsize=9)

    for j in range(3):
        axes[i, j].axis("off")

plt.suptitle(f'Clinical WCE Analysis — {len(detections)} Bleeding Frames out of {N} Total',
             fontsize=14, fontweight='bold', color='red')

out_path = "results/clinical_figure.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"✔ Saved {out_path}")
