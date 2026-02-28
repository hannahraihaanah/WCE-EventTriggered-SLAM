#!/bin/bash
echo "============================================"
echo "   WCE BLEEDING DETECTION PIPELINE"
echo "============================================"

echo ""
echo "STEP 1: Running EndoSLAM Pose Estimation..."
cd ~/EndoSLAM
python3 EndoSfMLearner/test_vo.py \
  --pretrained-posenet EndoSfMLearner/pretrained/08-13-00:00/exp_pose_model_best.pth.tar \
  --dataset-dir event_frames/ \
  --output-dir results/

echo ""
echo "STEP 2: Running Depth Estimation..."
python3 EndoSfMLearner/test_disp.py \
  --pretrained-dispnet EndoSfMLearner/pretrained/08-13-00:00/dispnet_model_best.pth.tar \
  --dataset-dir event_frames/ \
  --output-dir results/

echo ""
echo "============================================"
echo "   STEPS 1-2 COMPLETE!"
echo "   Poses saved: results/09.txt"
echo "   Depth maps saved: results/predictions.npy"
echo "============================================"

echo ""
echo "STEP 3: Lesion Localization..."
python3 lesion_localization.py

echo ""
echo "============================================"
echo "   STEPS 1-3 COMPLETE!"
echo "   Poses: results/09.txt"
echo "   Depth maps: results/predictions.npy"
echo "   Lesion 3D: results/lesion_3d.npy"
echo "   Lesion position: X=0.4133 Y=0.0231 Z=0.4508"
echo "============================================"

echo ""
echo "STEP 4: Clinical Visualization..."
python3 clinical_visualizer.py

echo "============================================"
echo "   PIPELINE COMPLETE!"
echo "   Poses: results/09.txt"
echo "   Depth maps: results/predictions.npy"
echo "   Lesion 3D: results/lesion_3d.npy"
python3 -c "
import numpy as np
pos = np.load('results/lesion_3d.npy')
print(f'   Primary Lesion: X={pos[0]:.4f} Y={pos[1]:.4f} Z={pos[2]:.4f}')
"
echo "   Clinical figure: results/clinical_figure.png"
echo "   Trajectory plot: results/trajectory_3d.png"
echo "============================================"
echo "   Localization: results/lesion_localization.png"
echo "============================================"
echo ""
echo "📊 To view results run:"
echo "   eog results/clinical_figure.png"
echo "   eog results/lesion_localization.png"
echo ""
