#!/usr/bin/env python3
"""
ECG-Pose Integration Server

This server integrates ECG device data with pose detection using timestamp synchronization.
It can handle:
1. Image + ECG data (runs pose detection and matches with ECG by timestamp)
2. ECG data only (stores for future matching)

Author: AI Assistant
Version: 1.0.0
License: MIT
"""

import os
import sys
import time
import logging
import argparse
import threading
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union
import json

import numpy as np
import pandas as pd
import cv2
from flask import Flask, request, jsonify, render_template_string

# Add parent directory to path to import pose detection modules
sys.path.append(str(Path(__file__).parent.parent))
from pose_detection_professional import Config, PoseDetector

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

class ECGDataManager:
    """Manages ECG data storage and retrieval."""
    
    def __init__(self, csv_file: str = "dummy_ecg_data.csv"):
        self.csv_file = Path(__file__).parent / csv_file
        self.ecg_data = []
        self.load_ecg_data()
    
    def load_ecg_data(self) -> None:
        """Load ECG data from CSV file."""
        try:
            df = pd.read_csv(self.csv_file)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            self.ecg_data = df.to_dict('records')
            logger.info(f"✅ Loaded {len(self.ecg_data)} ECG records from {self.csv_file}")
        except FileNotFoundError:
            logger.warning(f"⚠️ ECG file {self.csv_file} not found. Starting with empty data.")
            self.ecg_data = []
        except Exception as e:
            logger.error(f"❌ Error loading ECG data: {e}")
            self.ecg_data = []
    
    def add_ecg_record(self, record: Dict) -> None:
        """Add a new ECG record."""
        record['timestamp'] = pd.to_datetime(record['timestamp'])
        self.ecg_data.append(record)
        logger.info(f"📊 Added ECG record: {record['timestamp']}")
    
    def find_nearest_ecg(self, target_timestamp: datetime, tolerance_seconds: float = 1.0) -> Optional[Dict]:
        """Find ECG record with nearest timestamp to target."""
        if not self.ecg_data:
            return None
        
        target_dt = pd.to_datetime(target_timestamp)
        min_diff = None
        nearest_record = None
        
        for record in self.ecg_data:
            diff = abs((record['timestamp'] - target_dt).total_seconds())
            if diff <= tolerance_seconds and (min_diff is None or diff < min_diff):
                min_diff = diff
                nearest_record = record
        
        if nearest_record:
            logger.info(f"🔍 Found ECG match: {min_diff:.3f}s difference")
        else:
            logger.warning(f"⚠️ No ECG data within {tolerance_seconds}s of {target_timestamp}")
        
        return nearest_record
    
    def get_ecg_in_range(self, start_time: datetime, end_time: datetime) -> List[Dict]:
        """Get all ECG records within a time range."""
        start_dt = pd.to_datetime(start_time)
        end_dt = pd.to_datetime(end_time)
        
        return [
            record for record in self.ecg_data
            if start_dt <= record['timestamp'] <= end_dt
        ]

class PoseECGIntegrator:
    """Integrates pose detection with ECG data."""
    
    def __init__(self):
        self.config = Config()
        self.pose_detector = PoseDetector(self.config)
        self.ecg_manager = ECGDataManager()
        self.setup_pose_detector()
        
        # Storage for integrated data
        self.integrated_data = []
    
    def setup_pose_detector(self) -> None:
        """Initialize the pose detector."""
        logger.info("🤖 Initializing Pose Detector...")
        self.pose_detector.setup()
        logger.info("✅ Pose Detector ready!")
    
    def process_image(self, image_path: str, timestamp: Optional[str] = None) -> Dict:
        """Process image through pose detection and integrate with ECG data."""
        if timestamp is None:
            timestamp = datetime.now().isoformat()
        
        # Load and process image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Detect pose
        keypoints = self.pose_detector.detect_pose(image)
        
        # Create pose data
        pose_data = {
            'timestamp': timestamp,
            'image_path': image_path,
            'pose_detected': True
        }
        
        # Add keypoint data
        for name, idx in self.config.KEYPOINT_DICT.items():
            pose_data[f'{name}_y'] = keypoints[idx, 0]
            pose_data[f'{name}_x'] = keypoints[idx, 1]
            pose_data[f'{name}_confidence'] = keypoints[idx, 2]
        
        # Find matching ECG data
        target_time = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
        ecg_record = self.ecg_manager.find_nearest_ecg(target_time)
        
        # Integrate ECG data
        if ecg_record:
            pose_data.update({
                'ecg_timestamp': ecg_record['timestamp'].isoformat(),
                'heart_rate': ecg_record.get('heart_rate'),
                'ecg_lead1': ecg_record.get('ecg_lead1'),
                'ecg_lead2': ecg_record.get('ecg_lead2'),
                'ecg_lead3': ecg_record.get('ecg_lead3'),
                'blood_pressure_systolic': ecg_record.get('blood_pressure_systolic'),
                'blood_pressure_diastolic': ecg_record.get('blood_pressure_diastolic'),
                'oxygen_saturation': ecg_record.get('oxygen_saturation'),
                'ecg_matched': True
            })
        else:
            pose_data.update({
                'ecg_timestamp': None,
                'heart_rate': None,
                'ecg_lead1': None,
                'ecg_lead2': None,
                'ecg_lead3': None,
                'blood_pressure_systolic': None,
                'blood_pressure_diastolic': None,
                'oxygen_saturation': None,
                'ecg_matched': False
            })
        
        # Store integrated data
        self.integrated_data.append(pose_data)
        
        logger.info(f"🎯 Processed image with pose detection and ECG integration")
        return pose_data
    
    def process_ecg_only(self, ecg_data: Dict) -> Dict:
        """Process ECG data only (no pose detection)."""
        # Add timestamp if not provided
        if 'timestamp' not in ecg_data:
            ecg_data['timestamp'] = datetime.now().isoformat()
        
        # Store ECG data
        self.ecg_manager.add_ecg_record(ecg_data.copy())
        
        # Create response
        response = {
            'timestamp': ecg_data['timestamp'],
            'pose_detected': False,
            'ecg_matched': True,
            **ecg_data
        }
        
        # Store in integrated data
        self.integrated_data.append(response)
        
        logger.info(f"📊 Processed ECG data only")
        return response
    
    def export_data(self, output_file: str = "integrated_data.csv") -> str:
        """Export all integrated data to CSV."""
        if not self.integrated_data:
            logger.warning("⚠️ No data to export")
            return ""
        
        df = pd.DataFrame(self.integrated_data)
        output_path = Path(__file__).parent / output_file
        df.to_csv(output_path, index=False)
        
        logger.info(f"💾 Exported {len(self.integrated_data)} records to {output_path}")
        return str(output_path)

# Flask web server
app = Flask(__name__)
integrator = PoseECGIntegrator()

# HTML template for web interface
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>ECG-Pose Integration Server</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; }
        .container { max-width: 800px; margin: 0 auto; }
        .form-group { margin-bottom: 20px; }
        label { display: block; margin-bottom: 5px; font-weight: bold; }
        input, textarea { width: 100%; padding: 8px; margin-bottom: 10px; }
        button { background: #007bff; color: white; padding: 10px 20px; border: none; cursor: pointer; }
        button:hover { background: #0056b3; }
        .result { margin-top: 20px; padding: 15px; background: #f8f9fa; border: 1px solid #dee2e6; }
        .status { margin: 20px 0; padding: 10px; border-radius: 5px; }
        .success { background: #d4edda; border: 1px solid #c3e6cb; color: #155724; }
        .info { background: #d1ecf1; border: 1px solid #bee5eb; color: #0c5460; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🏥 ECG-Pose Integration Server</h1>
        
        <div class="status info">
            <strong>Server Status:</strong> Running ✅<br>
            <strong>Pose Detector:</strong> Ready 🤖<br>
            <strong>ECG Data:</strong> {{ ecg_count }} records loaded 📊
        </div>
        
        <h2>📤 Submit Data</h2>
        
        <h3>Option 1: Image + ECG Data</h3>
        <form action="/process_image" method="post" enctype="multipart/form-data">
            <div class="form-group">
                <label>Upload Image:</label>
                <input type="file" name="image" accept="image/*" required>
            </div>
            <div class="form-group">
                <label>Timestamp (optional, ISO format):</label>
                <input type="text" name="timestamp" placeholder="2025-06-29T14:22:00.000000">
            </div>
            <button type="submit">🎯 Process Image & Match ECG</button>
        </form>
        
        <h3>Option 2: ECG Data Only</h3>
        <form action="/process_ecg" method="post">
            <div class="form-group">
                <label>Heart Rate:</label>
                <input type="number" name="heart_rate" placeholder="72">
            </div>
            <div class="form-group">
                <label>ECG Lead 1:</label>
                <input type="number" step="0.01" name="ecg_lead1" placeholder="0.12">
            </div>
            <div class="form-group">
                <label>ECG Lead 2:</label>
                <input type="number" step="0.01" name="ecg_lead2" placeholder="-0.05">
            </div>
            <div class="form-group">
                <label>ECG Lead 3:</label>
                <input type="number" step="0.01" name="ecg_lead3" placeholder="0.08">
            </div>
            <div class="form-group">
                <label>Blood Pressure (Systolic):</label>
                <input type="number" name="blood_pressure_systolic" placeholder="120">
            </div>
            <div class="form-group">
                <label>Blood Pressure (Diastolic):</label>
                <input type="number" name="blood_pressure_diastolic" placeholder="80">
            </div>
            <div class="form-group">
                <label>Oxygen Saturation:</label>
                <input type="number" name="oxygen_saturation" placeholder="98">
            </div>
            <button type="submit">📊 Submit ECG Data</button>
        </form>
        
        <h3>Data Management</h3>
        <a href="/export"><button>💾 Export All Data to CSV</button></a>
        <a href="/status"><button>📊 View Server Status</button></a>
    </div>
</body>
</html>
"""

@app.route('/')
def index():
    """Main web interface."""
    ecg_count = len(integrator.ecg_manager.ecg_data)
    return render_template_string(HTML_TEMPLATE, ecg_count=ecg_count)

@app.route('/process_image', methods=['POST'])
def process_image():
    """Process uploaded image with pose detection and ECG matching."""
    try:
        # Get uploaded file
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No image file selected'}), 400
        
        # Save uploaded file
        upload_dir = Path(__file__).parent / 'uploads'
        upload_dir.mkdir(exist_ok=True)
        
        timestamp = request.form.get('timestamp') or datetime.now().isoformat()
        filename = f"image_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        filepath = upload_dir / filename
        file.save(str(filepath))
        
        # Process image
        result = integrator.process_image(str(filepath), timestamp)
        
        return jsonify({
            'status': 'success',
            'message': 'Image processed successfully',
            'data': result
        })
        
    except Exception as e:
        logger.error(f"Error processing image: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/process_ecg', methods=['POST'])
def process_ecg():
    """Process ECG data only."""
    try:
        # Get form data
        ecg_data = {}
        for field in ['heart_rate', 'ecg_lead1', 'ecg_lead2', 'ecg_lead3', 
                     'blood_pressure_systolic', 'blood_pressure_diastolic', 'oxygen_saturation']:
            value = request.form.get(field)
            if value:
                ecg_data[field] = float(value) if '.' in value else int(value)
        
        if not ecg_data:
            return jsonify({'error': 'No ECG data provided'}), 400
        
        # Process ECG data
        result = integrator.process_ecg_only(ecg_data)
        
        return jsonify({
            'status': 'success',
            'message': 'ECG data processed successfully',
            'data': result
        })
        
    except Exception as e:
        logger.error(f"Error processing ECG data: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/export')
def export_data():
    """Export all integrated data to CSV."""
    try:
        output_file = integrator.export_data()
        return jsonify({
            'status': 'success',
            'message': f'Data exported successfully',
            'file': output_file,
            'records': len(integrator.integrated_data)
        })
    except Exception as e:
        logger.error(f"Error exporting data: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/status')
def server_status():
    """Get server status information."""
    return jsonify({
        'status': 'running',
        'pose_detector_ready': integrator.pose_detector.model_manager.model is not None,
        'ecg_records_loaded': len(integrator.ecg_manager.ecg_data),
        'integrated_records': len(integrator.integrated_data),
        'server_time': datetime.now().isoformat()
    })

def main():
    """Main entry point for the server."""
    parser = argparse.ArgumentParser(description="ECG-Pose Integration Server")
    parser.add_argument("--host", default="127.0.0.1", help="Host address (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=5000, help="Port number (default: 5000)")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    
    args = parser.parse_args()
    
    logger.info("🏥 Starting ECG-Pose Integration Server...")
    logger.info(f"🌐 Server will run at http://{args.host}:{args.port}")
    logger.info("📊 Loaded ECG data and initialized pose detector")
    logger.info("🎯 Ready to process images and ECG data!")
    
    # Run Flask server
    app.run(host=args.host, port=args.port, debug=args.debug)

if __name__ == "__main__":
    main() 