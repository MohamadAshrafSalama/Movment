#!/usr/bin/env python3
"""
Test script to verify the pose detection installation is working correctly.
"""

import sys
import subprocess
from pathlib import Path

def test_dependencies():
    """Test that all dependencies are installed."""
    print("🔍 Testing dependencies...")
    
    try:
        import tensorflow as tf
        print(f"✅ TensorFlow: {tf.__version__}")
    except ImportError:
        print("❌ TensorFlow not found")
        return False
    
    try:
        import tensorflow_hub as hub
        print(f"✅ TensorFlow Hub: {hub.__version__}")
    except ImportError:
        print("❌ TensorFlow Hub not found")
        return False
    
    try:
        import cv2
        print(f"✅ OpenCV: {cv2.__version__}")
    except ImportError:
        print("❌ OpenCV not found")
        return False
    
    try:
        import pandas as pd
        print(f"✅ Pandas: {pd.__version__}")
    except ImportError:
        print("❌ Pandas not found")
        return False
    
    try:
        import numpy as np
        print(f"✅ NumPy: {np.__version__}")
    except ImportError:
        print("❌ NumPy not found")
        return False
    
    return True

def test_script_help():
    """Test that the main script shows help correctly."""
    print("\n🔍 Testing script help...")
    
    try:
        result = subprocess.run([
            sys.executable, "pose_detection_professional.py", "--help"
        ], capture_output=True, text=True, timeout=10)
        
        if result.returncode == 0 and "Professional Human Pose Detection" in result.stdout:
            print("✅ Script help works correctly")
            return True
        else:
            print("❌ Script help failed")
            print(f"Return code: {result.returncode}")
            print(f"stdout: {result.stdout}")
            print(f"stderr: {result.stderr}")
            return False
    except subprocess.TimeoutExpired:
        print("❌ Script help timed out")
        return False
    except Exception as e:
        print(f"❌ Script help error: {e}")
        return False

def test_file_structure():
    """Test that all required files exist."""
    print("\n🔍 Testing file structure...")
    
    required_files = [
        "pose_detection_professional.py",
        "requirements.txt",
        "environment.yml",
        "setup.py",
        "setup_environment.sh",
        "README.md",
        "LICENSE"
    ]
    
    all_exist = True
    for file in required_files:
        if Path(file).exists():
            print(f"✅ {file}")
        else:
            print(f"❌ {file} missing")
            all_exist = False
    
    return all_exist

def main():
    """Run all tests."""
    print("🚀 Running installation tests...\n")
    
    tests = [
        ("Dependencies", test_dependencies),
        ("File Structure", test_file_structure),
        ("Script Help", test_script_help),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{'='*50}")
        print(f"Testing: {test_name}")
        print('='*50)
        
        if test_func():
            passed += 1
            print(f"✅ {test_name}: PASSED")
        else:
            print(f"❌ {test_name}: FAILED")
    
    print(f"\n{'='*60}")
    print(f"🎯 TEST SUMMARY: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Installation is working correctly.")
        print("\n📋 Next steps:")
        print("1. Run: python pose_detection_professional.py --input your_video.mp4")
        print("2. Check the output/ directory for results")
        return 0
    else:
        print("❌ Some tests failed. Please check the installation.")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 