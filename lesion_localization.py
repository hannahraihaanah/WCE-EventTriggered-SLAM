import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

# ------------------------------------------------
# Load data
# ------------------------------------------------
depths = np.load('results/predictions.npy')
rgb_dir = 'event_frames'
rgb_files = sorted([f for f in os.listdir(rgb_dir) if f.endswith('.png')])

# Load poses
poses = []
with open('results/09.txt', 'r') as f:
    for line in f:
        values = list(map(float, line.strip().split()))
        if len(values) == 12:
            pose = np.array(values).reshape(3, 4)
            poses.append(pose)

# Load ALL detections
all_detections = []
with open('event_frames/all_detections.txt', 'r') as f:
    for line in f:
        vals = line.strip().split()
        if len(vals) >= 6:
            all_detections.append({
                'frame_idx': int(vals[0]),
                'bbox': (int(vals[1]), int(vals[2]), int(vals[3]), int(vals[4])),
                'conf': float(vals[5])
            })

print(f"Total detections loaded: {len(all_detections)}")

# Camera intrinsics
fx, fy = 287.0, 287.0
cx, cy = 160.0, 120.0

# ------------------------------------------------
# Localize ALL detections in 3D
# ------------------------------------------------
all_3d_positions = []

for d in all_detections:
    idx = d['frame_idx']
    if idx >= len(depths) or idx >= len(poses):
        continue

    x1, y1, x2, y2 = d['bbox']
    pixel_x = (x1 + x2) // 2
    pixel_y = (y1 + y2) // 2

    depth_map = depths[idx]
    frame = cv2.imread(os.path.join(rgb_dir, rgb_files[idx]))
    d_resized = cv2.resize(depth_map, (frame.shape[1], frame.shape[0]))
    lesion_depth = float(d_resized[pixel_y, pixel_x])

    # Back project to 3D
    x_cam = (pixel_x - cx) * lesion_depth / fx
    y_cam = (pixel_y - cy) * lesion_depth / fy
    z_cam = lesion_depth

    # Transform to world coordinates
    pose = poses[idx]
    R = pose[:3, :3]
    t = pose[:3, 3]
    pt_world = R @ np.array([x_cam, y_cam, z_cam]) + t

    all_3d_positions.append({
        'frame_idx': idx,
        'conf': d['conf'],
        'pixel': (pixel_x, pixel_y),
        'world': pt_world
    })

    print(f"Frame {idx} | conf={d['conf']:.2f} | 3D: X={pt_world[0]:.4f}, Y={pt_world[1]:.4f}, Z={pt_world[2]:.4f}")

# ------------------------------------------------
# Pick highest confidence as PRIMARY lesion
# ------------------------------------------------
best = max(all_3d_positions, key=lambda x: x['conf'])
print(f"\n✅ PRIMARY Lesion (highest conf={best['conf']:.2f}):")
print(f"Frame {best['frame_idx']} → X={best['world'][0]:.4f}, Y={best['world'][1]:.4f}, Z={best['world'][2]:.4f}")

# Save primary lesion
np.save('results/lesion_3d.npy', best['world'])

# Save all 3D positions
all_worlds = np.array([p['world'] for p in all_3d_positions])
np.save('results/all_lesions_3d.npy', all_worlds)

print(f"\nSaved results/lesion_3d.npy (primary)")
print(f"Saved results/all_lesions_3d.npy ({len(all_worlds)} positions)")

# ------------------------------------------------
# Visualize
# ------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Load best frame
best_frame = cv2.imread(os.path.join(rgb_dir, rgb_files[best['frame_idx']]))
best_frame = cv2.cvtColor(best_frame, cv2.COLOR_BGR2RGB)

# Draw all boxes on best frame
for d in all_detections:
    if d['frame_idx'] == best['frame_idx']:
        x1, y1, x2, y2 = d['bbox']
        cv2.rectangle(best_frame, (x1,y1), (x2,y2), (255,0,0), 3)
        cv2.putText(best_frame, f"{d['conf']:.2f}",
                   (x1, y1-8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2)

axes[0].imshow(best_frame)
axes[0].set_title(f'Best Detection Frame {best["frame_idx"]}\nconf={best["conf"]:.2f}',
                  color='red', fontweight='bold')
axes[0].axis('off')

# Depth map with all lesion markers
depth_map = depths[best['frame_idx']]
d_norm = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())
axes[1].imshow(d_norm, cmap='plasma')

for p in all_3d_positions:
    if p['frame_idx'] == best['frame_idx']:
        px, py = p['pixel']
        px_s = int(px * depth_map.shape[1] / best_frame.shape[1])
        py_s = int(py * depth_map.shape[0] / best_frame.shape[0])
        axes[1].plot(px_s, py_s, 'r*', markersize=15)

axes[1].set_title('Depth Map — Lesion Positions Marked', color='red')
axes[1].axis('off')

plt.suptitle(f'Lesion 3D Localization — {len(all_3d_positions)} Detections\nPrimary: X={best["world"][0]:.4f}, Y={best["world"][1]:.4f}, Z={best["world"][2]:.4f}',
             fontsize=12, fontweight='bold', color='red')
plt.tight_layout()
plt.savefig('results/lesion_localization.png', dpi=150, bbox_inches='tight')
print("Saved results/lesion_localization.png!")
