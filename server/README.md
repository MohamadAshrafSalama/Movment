# 🏥 ECG-Pose Integration Server

A web server that integrates ECG device data with human pose detection using timestamp synchronization.

## 🌟 Features

- **🔗 Timestamp Synchronization**: Matches ECG data with pose detection using timestamps
- **🎯 Dual Input Support**: 
  - Images + ECG matching (runs pose detection and finds nearest ECG data)
  - ECG data only (stores for future matching)
- **🌐 Web Interface**: Easy-to-use web UI for data submission
- **📊 Data Management**: Export integrated data to CSV
- **🔍 Nearest Neighbor Matching**: Finds closest ECG data within tolerance (1 second default)

## 📁 Server Structure

```
server/
├── ecg_pose_server.py      # Main server application
├── dummy_ecg_data.csv      # Sample ECG data (30 records)
├── requirements.txt        # Server dependencies
├── README.md              # This documentation
├── uploads/               # Auto-created: uploaded images
└── integrated_data.csv    # Auto-created: exported data
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
# Navigate to server directory
cd server

# Install requirements
pip install -r requirements.txt
```

### 2. Start Server

```bash
# Run server
python ecg_pose_server.py

# Or with custom settings
python ecg_pose_server.py --host 0.0.0.0 --port 8080 --debug
```

### 3. Access Web Interface

Open your browser and go to: `http://127.0.0.1:5000`

## 📊 ECG Data Format

The server expects ECG data with these fields:

```csv
timestamp,heart_rate,ecg_lead1,ecg_lead2,ecg_lead3,blood_pressure_systolic,blood_pressure_diastolic,oxygen_saturation
2025-06-29T14:22:00.000000,72,0.12,-0.05,0.08,120,80,98
```

### Fields:
- **timestamp**: ISO format timestamp (e.g., `2025-06-29T14:22:00.000000`)
- **heart_rate**: Beats per minute
- **ecg_lead1/2/3**: ECG signal values
- **blood_pressure_systolic/diastolic**: Blood pressure readings
- **oxygen_saturation**: SpO2 percentage

## 🎯 Usage Scenarios

### Scenario 1: Image + ECG Matching

1. **Upload Image**: Submit an image through the web interface
2. **Optional Timestamp**: Provide timestamp or use current time
3. **Pose Detection**: Server runs MoveNet pose detection
4. **ECG Matching**: Finds nearest ECG data within 1 second
5. **Integration**: Combines pose keypoints with ECG data in single record

### Scenario 2: ECG Data Only

1. **Submit ECG**: Send ECG data through web form or API
2. **Storage**: Data stored for future image matching
3. **Availability**: Available for matching with future images

## 📡 API Endpoints

### `POST /process_image`
Process uploaded image with pose detection and ECG matching.

**Form Data:**
- `image`: Image file
- `timestamp`: Optional ISO timestamp

**Response:**
```json
{
  "status": "success",
  "message": "Image processed successfully",
  "data": {
    "timestamp": "2025-06-29T14:22:00.000000",
    "pose_detected": true,
    "ecg_matched": true,
    "nose_x": 167.05,
    "nose_y": 144.76,
    "nose_confidence": 0.579,
    "heart_rate": 72,
    "ecg_lead1": 0.12,
    "blood_pressure_systolic": 120
  }
}
```

### `POST /process_ecg`
Process ECG data only.

**Form Data:**
- `heart_rate`: Number
- `ecg_lead1/2/3`: Float values
- `blood_pressure_systolic/diastolic`: Numbers
- `oxygen_saturation`: Number

### `GET /export`
Export all integrated data to CSV.

### `GET /status`
Get server status and statistics.

## 🔧 Configuration

### Timestamp Tolerance
Modify the tolerance for ECG matching:

```python
# In ECGDataManager.find_nearest_ecg()
tolerance_seconds = 1.0  # Default: 1 second
```

### ECG Data Source
Change the ECG data file:

```python
# In ECGDataManager.__init__()
csv_file = "your_ecg_data.csv"
```

## 📈 Data Output

### Integrated Data Columns

**Pose Data (17 keypoints × 3 values each):**
- `{keypoint}_x`: X coordinate
- `{keypoint}_y`: Y coordinate  
- `{keypoint}_confidence`: Detection confidence

**ECG Data:**
- `heart_rate`: BPM
- `ecg_lead1/2/3`: Signal values
- `blood_pressure_systolic/diastolic`: Pressure readings
- `oxygen_saturation`: SpO2 percentage

**Metadata:**
- `timestamp`: Image/data timestamp
- `ecg_timestamp`: Matched ECG timestamp
- `pose_detected`: Boolean
- `ecg_matched`: Boolean
- `image_path`: Path to uploaded image

## 🧪 Testing

### Test with Dummy Data

The server includes 30 dummy ECG records spanning 3 seconds:
- **Time Range**: `2025-06-29T14:22:00.000000` to `2025-06-29T14:22:02.900000`
- **Interval**: 100ms between records
- **Values**: Realistic heart rate, ECG leads, blood pressure, SpO2

### Sample Test:

1. Start server: `python ecg_pose_server.py`
2. Upload any image with timestamp: `2025-06-29T14:22:01.500000`
3. Server will find matching ECG data and integrate

## 🔍 Monitoring

### Server Logs
```
2025-06-29 14:22:00 - INFO - 🏥 Starting ECG-Pose Integration Server...
2025-06-29 14:22:00 - INFO - ✅ Loaded 30 ECG records from dummy_ecg_data.csv
2025-06-29 14:22:05 - INFO - 🤖 Initializing Pose Detector...
2025-06-29 14:22:10 - INFO - ✅ Pose Detector ready!
2025-06-29 14:22:15 - INFO - 🎯 Processed image with pose detection and ECG integration
2025-06-29 14:22:15 - INFO - 🔍 Found ECG match: 0.150s difference
```

### Web Interface Status
- **Server Status**: Running ✅
- **Pose Detector**: Ready 🤖  
- **ECG Data**: X records loaded 📊

## 🚨 Error Handling

- **Image Upload Failures**: Returns 400 with error message
- **ECG Data Issues**: Graceful handling of missing/invalid data
- **Timestamp Parsing**: Handles various ISO format variations
- **Model Loading**: Automatic retry on TensorFlow Hub failures

## 🔒 Security Notes

- **File Upload**: Images saved to `uploads/` directory
- **Input Validation**: Basic validation on form inputs
- **Local Server**: Runs on localhost by default
- **No Authentication**: Add authentication for production use

---

**Ready to integrate ECG data with pose detection! 🏥🤖** 