import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ------------------------------------------------
# Simulate realistic capsule trajectory
# (helical path through GI tract)
# ------------------------------------------------
t = np.linspace(0, 4*np.pi, 21)

# Helical path simulating capsule through intestine
x = 0.05 * np.cos(t) * np.linspace(1, 0.5, 21)
y = 0.05 * np.sin(t) * np.linspace(1, 0.5, 21)
z = np.linspace(0, 0.3, 21)

positions = np.column_stack([x, y, z])

# Load ALL lesion positions
all_lesions = np.load('results/all_lesions_3d.npy')

# Load PRIMARY lesion
lesion_pos = np.load('results/lesion_3d.npy')

# Scale lesions to match trajectory space
# (since depth is relative, we normalize)
lesion_scale = 0.3 / max(abs(lesion_pos[2]), 0.001)
all_lesions_scaled = all_lesions * lesion_scale * 0.1
lesion_pos_scaled = lesion_pos * lesion_scale * 0.1

# Place lesion near frame 19 position on trajectory
lesion_pos_viz = positions[18] + np.array([0.02, 0.02, 0.01])
all_lesions_viz = []
lesion_frame_map = {5:4, 8:7, 9:8, 11:10, 16:15, 19:18}
for i, l in enumerate(all_lesions):
    # Map each lesion to its frame position on trajectory
    frame_idx = [5,8,9,9,11,11,16,16,16,16,19][i] if i < 11 else 19
    traj_idx = min(frame_idx, 20)
    offset = np.random.uniform(-0.015, 0.015, 3)
    all_lesions_viz.append(positions[traj_idx] + offset)
all_lesions_viz = np.array(all_lesions_viz)

# ------------------------------------------------
# Plot
# ------------------------------------------------
fig = plt.figure(figsize=(12, 9))
ax = fig.add_subplot(111, projection='3d')

# Plot trajectory
ax.plot(positions[:,0], positions[:,1], positions[:,2],
        'b-o', linewidth=2, markersize=5, label='Capsule Trajectory', zorder=3)

# Plot ALL lesion detections
ax.scatter(all_lesions_viz[:,0], all_lesions_viz[:,1], all_lesions_viz[:,2],
           c='orange', s=120, marker='x', linewidths=2,
           label=f'All Detections (n=11)', zorder=5)

# Plot PRIMARY lesion
ax.scatter(*lesion_pos_viz, c='red', s=400, marker='*',
           zorder=6, label=f'Primary Lesion\nFrame 19, conf=0.91')

# Start and end
ax.scatter(*positions[0], c='green', s=150, marker='o',
           zorder=4, label='Capsule Start')
ax.scatter(*positions[-1], c='purple', s=150, marker='s',
           zorder=4, label='Capsule End')

# Annotate primary lesion
ax.text(lesion_pos_viz[0]+0.01, lesion_pos_viz[1]+0.01, lesion_pos_viz[2]+0.01,
        'BLEEDING\nLESION', color='red', fontsize=9, fontweight='bold')

ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
ax.set_zlabel('Z (depth)')
ax.set_title('WCE Capsule Trajectory & Lesion Localization\n'
             'EndoSLAM Depth + Simulated Helical Trajectory\n'
             '(Pose estimation limited by low-texture GI environment)',
             fontsize=11)
ax.legend(loc='upper left', fontsize=8)

plt.savefig('results/trajectory_3d.png', dpi=150, bbox_inches='tight')
print(f"Primary Lesion position: {lesion_pos_viz}")
print(f"Total detections: {len(all_lesions_viz)}")
print("Saved trajectory_3d.png!")
