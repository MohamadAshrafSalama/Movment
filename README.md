# Human Pose Detection with TensorFlow MoveNet

A lightweight pipeline for detecting human poses in videos using TensorFlow's MoveNet model. The project includes tools to set up a Python environment, run pose detection, and save annotated videos and CSV keypoint data.

## Getting Started

1. **Set up the environment** (conda or pip):
   ```bash
   ./setup_environment.sh
   # then activate the printed environment command
   ```
2. **Run pose detection** on a video file:
   ```bash
   python pose_detection_professional.py --input path/to/video.mp4
   ```
3. Results are saved in the `output/` folder (CSV file and two annotated videos).

## Directory Overview

- `pose_detection_professional.py` – main script for processing videos
- `demo.py` – simple example runner
- `setup_environment.sh` – helper to create a virtual environment
- `server/` – optional server for ECG + pose integration

## Example

```
python pose_detection_professional.py --input my_video.mp4 --output results
```

A CSV file named `my_video_keypoints.csv` and annotated videos will appear in the `results` directory.

## Requirements

- Python 3.8+
- TensorFlow 2.19+
- OpenCV, pandas and other packages listed in `requirements.txt`

## License

This project is released under the [MIT License](LICENSE).
