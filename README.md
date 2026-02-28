# Event-Triggered Bleeding Detection and 3D Lesion Localization for Wireless Capsule Endoscopy

## Overview
An event-triggered pipeline that combines YOLOv8 bleeding detection with EndoSLAM depth estimation for 3D lesion localization in Wireless Capsule Endoscopy (WCE).

## Pipeline
```
event_trigger.py → run_pipeline.sh → results
```

## Key Results
- 91% confidence bleeding detection
- 11 bleeding detections across 21 event frames
- 6 unique bleeding frames identified
- 3D lesion localization: X=-0.0955, Y=0.2831, Z=0.3359

## Usage
```bash
# Step 1: Detect bleeding and capture event frames
cd ~/Auto-WCEBleedGen
python3 event_trigger.py

# Step 2: Run full pipeline
cd ~/EndoSLAM
bash run_pipeline.sh
```

## Project Structure
```
├── event_trigger.py        # YOLO event detection
├── clinical_visualizer.py  # RGB + Depth visualization  
├── lesion_localization.py  # 3D lesion coordinate computation
├── run_pipeline.sh         # Master pipeline script
├── EndoSfMLearner/         # Depth + Pose estimation network
└── event_frames/           # Captured frames + detection files
```

## Requirements
- Python 3.10
- PyTorch (CPU)
- OpenCV 4.5
- scikit-image
- ultralytics (YOLOv8)

## References
- EndoSLAM: https://github.com/CapsuleEndoscope/EndoSLAM
- Auto-WCEBleedGen: https://github.com/pavan98765/Auto-WCEBleedGen

## Hackathon
24-Hour Healthcare Hackathon 2024
