#!/usr/bin/env python3
"""
Professional Human Pose Detection with TensorFlow MoveNet

This module provides a complete pipeline for human pose detection in videos,
with automatic model management, comprehensive error handling, and multiple output formats.

Author: AI Assistant
Version: 2.0.0
License: MIT
"""

import os
import sys
import time
import logging
import argparse
from pathlib import Path
from typing import Tuple, Dict, List, Optional, Union
from datetime import datetime

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
import cv2
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Suppress TensorFlow warnings
tf.get_logger().setLevel('ERROR')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class Config:
    """Configuration class for pose detection parameters."""
    
    # Model configuration
    MODEL_NAME = "movenet_lightning"
    MODEL_URL = "https://tfhub.dev/google/movenet/singlepose/lightning/4"
    INPUT_SIZE = 192
    CONFIDENCE_THRESHOLD = 0.3
    
    # File paths
    MODEL_CACHE_DIR = Path("models")
    OUTPUT_DIR = Path("output")
    
    # Video processing
    PROGRESS_UPDATE_INTERVAL = 60  # frames
    
    # Keypoint definitions
    KEYPOINT_DICT = {
        'nose': 0, 'left_eye': 1, 'right_eye': 2, 'left_ear': 3, 'right_ear': 4,
        'left_shoulder': 5, 'right_shoulder': 6, 'left_elbow': 7, 'right_elbow': 8,
        'left_wrist': 9, 'right_wrist': 10, 'left_hip': 11, 'right_hip': 12,
        'left_knee': 13, 'right_knee': 14, 'left_ankle': 15, 'right_ankle': 16
    }
    
    KEYPOINT_CONNECTIONS = [
        (0, 1), (0, 2), (1, 3), (2, 4),  # Face
        (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # Arms
        (5, 11), (6, 12), (11, 12),  # Torso
        (11, 13), (13, 15), (12, 14), (14, 16)  # Legs
    ]

class ModelManager:
    """Manages TensorFlow Hub model loading and caching."""
    
    def __init__(self, config: Config):
        self.config = config
        self.model = None
        
    def _download_model(self) -> None:
        """Download model from TensorFlow Hub."""
        logger.info("📥 Downloading MoveNet model from TensorFlow Hub...")
        
        # Create cache directory
        self.config.MODEL_CACHE_DIR.mkdir(exist_ok=True)
        
        try:
            # Download model
            self.model = hub.load(self.config.MODEL_URL)
            logger.info(f"✅ Model downloaded and cached in TensorFlow Hub cache")
            
        except Exception as e:
            logger.error(f"❌ Failed to download model: {e}")
            raise
    
    def _load_cached_model(self) -> None:
        """Load model from TensorFlow Hub cache."""
        logger.info("📂 Loading cached model from TensorFlow Hub cache")
        
        try:
            self.model = hub.load(self.config.MODEL_URL)
            logger.info("✅ Model loaded from cache successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to load cached model: {e}")
            raise
    
    def load_model(self) -> None:
        """Load model from cache or download if not available."""
        try:
            # Try to load from TensorFlow Hub cache first
            self._load_cached_model()
        except Exception:
            logger.info("📥 Model not in cache, downloading...")
            self._download_model()
    
    def predict(self, input_image: tf.Tensor) -> np.ndarray:
        """Run pose detection on input image."""
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        # Get the model's signature function
        model_fn = self.model.signatures['serving_default']
        input_image = tf.cast(input_image, dtype=tf.int32)
        outputs = model_fn(input_image)
        return outputs['output_0'].numpy()

class PoseDetector:
    """Main pose detection class."""
    
    def __init__(self, config: Config):
        self.config = config
        self.model_manager = ModelManager(config)
        
    def setup(self) -> None:
        """Initialize the pose detector."""
        logger.info("🤖 Initializing Pose Detector...")
        self.model_manager.load_model()
        logger.info("✅ Pose Detector ready!")
    
    def detect_pose(self, image: np.ndarray) -> np.ndarray:
        """Process a single image and return keypoints."""
        # Convert to RGB if needed
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
        elif image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Get dimensions
        height, width, _ = image.shape
        
        # Preprocess image
        input_img = tf.convert_to_tensor(image)
        input_img = tf.image.resize_with_pad(input_img, self.config.INPUT_SIZE, self.config.INPUT_SIZE)
        input_img = tf.expand_dims(input_img, axis=0)
        
        # Run inference
        keypoints_with_scores = self.model_manager.predict(input_img)
        keypoints = keypoints_with_scores[0, 0, :, :]
        
        # Convert to pixel coordinates
        keypoints_px = keypoints.copy()
        keypoints_px[:, 0] = keypoints[:, 0] * height  # y coordinate
        keypoints_px[:, 1] = keypoints[:, 1] * width   # x coordinate
        
        return keypoints_px
    
    def draw_annotations(self, image: np.ndarray, keypoints: np.ndarray, 
                        background_color: str = 'black') -> np.ndarray:
        """Draw keypoints and connections on image."""
        # Create background
        if background_color == 'black':
            output_image = np.zeros_like(image)
            keypoint_color = (0, 255, 255)  # Cyan
            connection_color = (255, 255, 0)  # Yellow
            text_color = (255, 255, 255)  # White
        else:  # white background
            output_image = np.ones_like(image) * 255
            keypoint_color = (255, 0, 0)  # Blue
            connection_color = (0, 0, 255)  # Red
            text_color = (0, 0, 0)  # Black
        
        # Draw connections
        for start_idx, end_idx in self.config.KEYPOINT_CONNECTIONS:
            start_conf = keypoints[start_idx, 2]
            end_conf = keypoints[end_idx, 2]
            
            if start_conf > self.config.CONFIDENCE_THRESHOLD and end_conf > self.config.CONFIDENCE_THRESHOLD:
                start_x, start_y = int(keypoints[start_idx, 1]), int(keypoints[start_idx, 0])
                end_x, end_y = int(keypoints[end_idx, 1]), int(keypoints[end_idx, 0])
                cv2.line(output_image, (start_x, start_y), (end_x, end_y), connection_color, 3)
        
        # Draw keypoints
        for i, keypoint in enumerate(keypoints):
            confidence = keypoint[2]
            if confidence > self.config.CONFIDENCE_THRESHOLD:
                x, y = int(keypoint[1]), int(keypoint[0])
                cv2.circle(output_image, (x, y), 8, keypoint_color, -1)
                # Add keypoint number
                cv2.putText(output_image, str(i), (x+10, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, text_color, 1)
        
        return output_image

class VideoProcessor:
    """Handles video processing operations."""
    
    def __init__(self, config: Config, pose_detector: PoseDetector):
        self.config = config
        self.pose_detector = pose_detector
    
    def process_video(self, input_path: Union[str, Path], 
                     output_dir: Optional[Union[str, Path]] = None) -> Dict[str, Union[str, pd.DataFrame]]:
        """Process video and generate all outputs."""
        input_path = Path(input_path)
        output_dir = Path(output_dir) if output_dir else self.config.OUTPUT_DIR
        
        if not input_path.exists():
            raise FileNotFoundError(f"Input video not found: {input_path}")
        
        # Create output directory
        output_dir.mkdir(exist_ok=True)
        
        # Define output paths
        csv_path = output_dir / f"{input_path.stem}_keypoints.csv"
        black_video_path = output_dir / f"{input_path.stem}_annotations_black.mp4"
        white_video_path = output_dir / f"{input_path.stem}_annotations_white.mp4"
        
        logger.info(f"🎬 Processing video: {input_path}")
        
        # Open video
        cap = cv2.VideoCapture(str(input_path))
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video file: {input_path}")
        
        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        logger.info(f"📺 Video properties: {width}x{height}, {fps:.1f} FPS, {total_frames} frames")
        
        # Create video writers
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        black_writer = cv2.VideoWriter(str(black_video_path), fourcc, fps, (width, height))
        white_writer = cv2.VideoWriter(str(white_video_path), fourcc, fps, (width, height))
        
        if not black_writer.isOpened() or not white_writer.isOpened():
            raise RuntimeError("Could not initialize video writers")
        
        # Process frames
        all_data = []
        start_time = time.time()
        
        with tqdm(total=total_frames, desc="🔄 Processing frames", unit="frame") as pbar:
            frame_count = 0
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Calculate timestamps
                video_timestamp = frame_count / fps
                system_timestamp = datetime.now().isoformat()
                current_time = datetime.now().strftime('%H:%M:%S.%f')[:-3]
                
                # Detect pose
                keypoints = self.pose_detector.detect_pose(frame)
                
                # Create annotation frames
                black_frame = self.pose_detector.draw_annotations(frame, keypoints, 'black')
                white_frame = self.pose_detector.draw_annotations(frame, keypoints, 'white')
                
                # Add timestamps to frames
                self._add_timestamp_overlay(black_frame, video_timestamp, current_time, frame_count, 'white')
                self._add_timestamp_overlay(white_frame, video_timestamp, current_time, frame_count, 'black')
                
                # Write frames
                black_writer.write(black_frame)
                white_writer.write(white_frame)
                
                # Collect data
                row_data = self._create_data_row(keypoints, video_timestamp, system_timestamp, frame_count)
                all_data.append(row_data)
                
                frame_count += 1
                pbar.update(1)
        
        # Cleanup
        cap.release()
        black_writer.release()
        white_writer.release()
        
        # Save CSV
        logger.info("💾 Saving CSV data...")
        df = pd.DataFrame(all_data)
        df.to_csv(csv_path, index=False)
        
        # Calculate metrics
        processing_time = time.time() - start_time
        processing_fps = frame_count / processing_time
        
        # File sizes
        csv_size = csv_path.stat().st_size / (1024 * 1024)
        black_size = black_video_path.stat().st_size / (1024 * 1024)
        white_size = white_video_path.stat().st_size / (1024 * 1024)
        
        # Results summary
        results = {
            'csv_file': str(csv_path),
            'black_video': str(black_video_path),
            'white_video': str(white_video_path),
            'dataframe': df,
            'processing_time': processing_time,
            'processing_fps': processing_fps,
            'file_sizes': {
                'csv_mb': csv_size,
                'black_video_mb': black_size,
                'white_video_mb': white_size
            }
        }
        
        self._print_summary(results, frame_count)
        return results
    
    def _add_timestamp_overlay(self, frame: np.ndarray, video_time: float, 
                              current_time: str, frame_num: int, text_color: str) -> None:
        """Add timestamp overlay to frame."""
        color = (255, 255, 255) if text_color == 'white' else (0, 0, 0)
        
        cv2.putText(frame, f"Video: {video_time:.2f}s", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(frame, f"System: {current_time}", (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(frame, f"Frame: {frame_num}", (10, 90), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    
    def _create_data_row(self, keypoints: np.ndarray, video_timestamp: float, 
                        system_timestamp: str, frame_number: int) -> Dict:
        """Create data row for CSV output."""
        row_data = {
            'video_timestamp': video_timestamp,
            'system_timestamp': system_timestamp,
            'frame_number': frame_number
        }
        
        for name, idx in self.config.KEYPOINT_DICT.items():
            row_data[f'{name}_y'] = keypoints[idx, 0]
            row_data[f'{name}_x'] = keypoints[idx, 1]
            row_data[f'{name}_confidence'] = keypoints[idx, 2]
        
        return row_data
    
    def _print_summary(self, results: Dict, frame_count: int) -> None:
        """Print processing summary."""
        logger.info("\n🎉 PROCESSING COMPLETE!")
        logger.info("=" * 60)
        logger.info(f"📊 CSV file: {results['csv_file']}")
        logger.info(f"   └── {frame_count} rows, {results['file_sizes']['csv_mb']:.1f} MB")
        logger.info(f"📹 Black background video: {results['black_video']}")
        logger.info(f"   └── {results['file_sizes']['black_video_mb']:.1f} MB")
        logger.info(f"📹 White background video: {results['white_video']}")
        logger.info(f"   └── {results['file_sizes']['white_video_mb']:.1f} MB")
        logger.info(f"⏱️  Processing time: {results['processing_time']:.1f} seconds")
        logger.info(f"🚀 Average speed: {results['processing_fps']:.1f} fps")
        logger.info("=" * 60)

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Professional Human Pose Detection")
    parser.add_argument("--input", "-i", type=str, required=True,
                       help="Input video file path")
    parser.add_argument("--output", "-o", type=str, default="output",
                       help="Output directory (default: output)")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Enable verbose logging")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        # Initialize components
        config = Config()
        pose_detector = PoseDetector(config)
        video_processor = VideoProcessor(config, pose_detector)
        
        # Setup
        pose_detector.setup()
        
        # Process video
        results = video_processor.process_video(args.input, args.output)
        
        # Show sample data
        logger.info("\n📋 Sample of collected data:")
        sample_cols = ['video_timestamp', 'system_timestamp', 'nose_confidence', 
                      'left_wrist_y', 'right_wrist_y']
        print(results['dataframe'][sample_cols].head())
        
        logger.info("✅ All tasks completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 