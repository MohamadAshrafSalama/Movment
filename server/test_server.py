#!/usr/bin/env python3
"""
Test script for ECG-Pose Integration Server

This script demonstrates how to interact with the server programmatically.
"""

import requests
import json
import time
from datetime import datetime

def test_server_status():
    """Test server status endpoint."""
    print("🔍 Testing server status...")
    try:
        response = requests.get('http://127.0.0.1:5000/status')
        if response.status_code == 200:
            data = response.json()
            print("✅ Server Status:")
            print(f"   Status: {data['status']}")
            print(f"   Pose Detector Ready: {data['pose_detector_ready']}")
            print(f"   ECG Records: {data['ecg_records_loaded']}")
            print(f"   Integrated Records: {data['integrated_records']}")
            return True
        else:
            print(f"❌ Status check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Error testing server: {e}")
        return False

def test_ecg_submission():
    """Test ECG data submission."""
    print("\n📊 Testing ECG data submission...")
    
    ecg_data = {
        'heart_rate': 85,
        'ecg_lead1': 0.25,
        'ecg_lead2': 0.10,
        'ecg_lead3': 0.15,
        'blood_pressure_systolic': 125,
        'blood_pressure_diastolic': 85,
        'oxygen_saturation': 97
    }
    
    try:
        response = requests.post('http://127.0.0.1:5000/process_ecg', data=ecg_data)
        if response.status_code == 200:
            data = response.json()
            print("✅ ECG data submitted successfully")
            print(f"   Timestamp: {data['data']['timestamp']}")
            print(f"   Heart Rate: {data['data']['heart_rate']}")
            print(f"   ECG Matched: {data['data']['ecg_matched']}")
            return True
        else:
            print(f"❌ ECG submission failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Error submitting ECG data: {e}")
        return False

def test_export_data():
    """Test data export."""
    print("\n💾 Testing data export...")
    
    try:
        response = requests.get('http://127.0.0.1:5000/export')
        if response.status_code == 200:
            data = response.json()
            print("✅ Data exported successfully")
            print(f"   File: {data['file']}")
            print(f"   Records: {data['records']}")
            return True
        else:
            print(f"❌ Export failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Error exporting data: {e}")
        return False

def main():
    """Run all tests."""
    print("🧪 ECG-Pose Server Test Suite")
    print("=" * 40)
    print("⚠️  Make sure the server is running first!")
    print("   Start with: python ecg_pose_server.py")
    print()
    
    # Wait a moment for user to start server
    input("Press Enter when server is running...")
    
    tests = [
        ("Server Status", test_server_status),
        ("ECG Submission", test_ecg_submission),
        ("Data Export", test_export_data),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        if test_func():
            passed += 1
        time.sleep(1)  # Small delay between tests
    
    print(f"\n{'='*40}")
    print(f"🎯 Test Results: {passed}/{total} passed")
    
    if passed == total:
        print("🎉 All tests passed! Server is working correctly.")
        print("\n📋 Next steps:")
        print("1. Open http://127.0.0.1:5000 in your browser")
        print("2. Upload an image to test pose detection + ECG matching")
        print("3. Submit ECG data through the web form")
    else:
        print("❌ Some tests failed. Check server logs.")

if __name__ == "__main__":
    main() 