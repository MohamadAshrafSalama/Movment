#!/usr/bin/env python3
"""
Demo script showing how to use the pose detection system programmatically.
"""

from pathlib import Path
from pose_detection_professional import Config, PoseDetector, VideoProcessor

def main():
    """Demo the pose detection system."""
    print("🎬 Pose Detection Demo")
    print("=" * 30)
    
    # Check if demo video exists
    demo_video = "Secret Stretches to do Anywhere! .mp4"
    if not Path(demo_video).exists():
        print(f"❌ Demo video '{demo_video}' not found!")
        print("Please place a video file in the current directory or modify the demo_video variable.")
        return
    
    print(f"📹 Using demo video: {demo_video}")
    
    # Initialize components
    print("🔧 Initializing components...")
    config = Config()
    pose_detector = PoseDetector(config)
    video_processor = VideoProcessor(config, pose_detector)
    
    # Setup
    print("🤖 Setting up pose detector...")
    pose_detector.setup()
    
    # Process video
    print("🔄 Processing video...")
    results = video_processor.process_video(
        input_path=demo_video,
        output_dir="demo_output"
    )
    
    # Show results
    print("\n📊 Results Summary:")
    print(f"CSV file: {results['csv_file']}")
    print(f"Black video: {results['black_video']}")
    print(f"White video: {results['white_video']}")
    print(f"Processing time: {results['processing_time']:.1f} seconds")
    print(f"Processing speed: {results['processing_fps']:.1f} fps")
    
    # Show sample data
    df = results['dataframe']
    print("\n📋 Sample keypoint data:")
    print(df[['video_timestamp', 'nose_confidence', 'left_wrist_y', 'right_wrist_y']].head())
    
    print("\n✅ Demo completed successfully!")
    print("Check the 'demo_output' directory for results.")

if __name__ == "__main__":
    main() 