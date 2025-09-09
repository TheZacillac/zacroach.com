# Commercial Detection System for Raspberry Pi

An AI-powered system that automatically mutes your TV during commercials by analyzing video and audio content in real-time.

## Features

- **Real-time Commercial Detection**: Uses computer vision and audio analysis to detect commercials
- **Multiple TV Control Methods**: Supports HDMI-CEC, IR blaster, Smart TV API, and audio-only control
- **Configurable Detection**: Adjustable thresholds and parameters for different TV setups
- **Raspberry Pi Optimized**: Designed to run efficiently on Raspberry Pi hardware
- **Comprehensive Logging**: Detailed logging for monitoring and debugging

## How It Works

The system continuously monitors your TV's video and audio output using:

1. **Camera Input**: Captures video from your TV screen
2. **Microphone Input**: Captures audio from your TV speakers
3. **AI Analysis**: Analyzes visual and audio features to detect commercial patterns
4. **TV Control**: Automatically mutes/unmutes your TV when commercials are detected

### Detection Methods

- **Visual Analysis**: Detects sudden changes in brightness, contrast, color patterns, and motion
- **Audio Analysis**: Monitors volume changes, spectral features, and audio characteristics
- **Duration Analysis**: Uses commercial length patterns to improve accuracy
- **Confidence Scoring**: Combines multiple signals for reliable detection

## Hardware Requirements

### Minimum Requirements
- Raspberry Pi 4 (4GB RAM recommended)
- USB camera or Pi Camera module
- USB microphone or audio input
- SD card (32GB+ recommended)

### Optional Hardware
- HDMI-CEC compatible TV
- IR blaster for remote control
- Smart TV with API access

## Installation

### 1. System Setup

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install required system packages
sudo apt install -y python3-pip python3-venv git
sudo apt install -y libopencv-dev python3-opencv
sudo apt install -y portaudio19-dev python3-pyaudio
sudo apt install -y cec-utils  # For HDMI-CEC support
sudo apt install -y lirc       # For IR blaster support
```

### 2. Clone and Setup Project

```bash
# Clone the repository
git clone <your-repo-url>
cd commercial-detector

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install Python dependencies
pip install -r requirements.txt
```

### 3. Hardware Configuration

#### Camera Setup
```bash
# Test camera
lsusb  # Check if camera is detected
v4l2-ctl --list-devices  # List video devices

# Test camera with OpenCV
python3 -c "import cv2; cap = cv2.VideoCapture(0); print('Camera OK' if cap.isOpened() else 'Camera FAIL')"
```

#### Audio Setup
```bash
# Test audio devices
arecord -l  # List audio devices
python3 -c "import sounddevice as sd; print(sd.query_devices())"
```

#### HDMI-CEC Setup (Optional)
```bash
# Enable HDMI-CEC
sudo apt install cec-utils
echo 'hdmi_cec_init=1' | sudo tee -a /boot/config.txt

# Test CEC
echo 'scan' | cec-client -s -d 1
```

## Configuration

Edit `config.ini` to customize the system:

```ini
[camera]
index = 0              # Camera device index
width = 640            # Video width
height = 480           # Video height
fps = 30               # Frames per second

[audio]
sample_rate = 44100    # Audio sample rate
channels = 2           # Audio channels
buffer_size = 1024     # Audio buffer size

[detection]
visual_threshold = 0.3     # Visual change sensitivity
audio_threshold = 0.7      # Audio change sensitivity
confidence_threshold = 0.8 # Detection confidence required
commercial_duration_min = 15.0  # Minimum commercial length (seconds)
commercial_duration_max = 180.0 # Maximum commercial length (seconds)
transition_buffer = 2.0    # Buffer time to prevent rapid switching

[tv_control]
method = audio_only     # Control method: hdmi_cec, ir_blaster, smart_tv_api, audio_only
cec_device = 0         # HDMI-CEC device number
ir_device = /dev/ttyUSB0 # IR blaster device
tv_ip = 192.168.1.100  # Smart TV IP address
tv_port = 8080         # Smart TV API port
audio_device = default # Audio device for audio-only control
```

## Usage

### Basic Usage

```bash
# Activate virtual environment
source venv/bin/activate

# Run the commercial detector
python3 commercial_detector.py
```

### Advanced Usage

```bash
# Run with custom config
python3 commercial_detector.py --config custom_config.ini

# Run in background
nohup python3 commercial_detector.py > output.log 2>&1 &

# Run as system service (see Service Setup section)
sudo systemctl start commercial-detector
```

### Testing Components

```bash
# Test TV controller
python3 tv_controller.py

# Test camera only
python3 -c "
import cv2
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if ret:
        cv2.imshow('Test', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
cap.release()
cv2.destroyAllWindows()
"
```

## TV Control Methods

### 1. Audio-Only Control (Default)
- Mutes/unmutes system audio
- Works with any TV setup
- Requires audio output through Raspberry Pi

### 2. HDMI-CEC Control
- Controls TV directly via HDMI
- Requires CEC-compatible TV
- Most reliable method

### 3. IR Blaster Control
- Sends IR commands to TV
- Requires IR blaster hardware
- Works with any IR-controlled TV

### 4. Smart TV API Control
- Uses TV's network API
- Requires smart TV with API access
- Most flexible but TV-specific

## Service Setup (Optional)

Create a systemd service for automatic startup:

```bash
# Create service file
sudo nano /etc/systemd/system/commercial-detector.service
```

Add the following content:

```ini
[Unit]
Description=Commercial Detection System
After=network.target

[Service]
Type=simple
User=pi
WorkingDirectory=/home/pi/commercial-detector
Environment=PATH=/home/pi/commercial-detector/venv/bin
ExecStart=/home/pi/commercial-detector/venv/bin/python commercial_detector.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Enable and start the service:

```bash
sudo systemctl daemon-reload
sudo systemctl enable commercial-detector
sudo systemctl start commercial-detector

# Check status
sudo systemctl status commercial-detector

# View logs
sudo journalctl -u commercial-detector -f
```

## Troubleshooting

### Common Issues

#### Camera Not Detected
```bash
# Check USB devices
lsusb

# Check video devices
ls /dev/video*

# Test with different camera index
python3 -c "import cv2; print([i for i in range(5) if cv2.VideoCapture(i).isOpened()])"
```

#### Audio Issues
```bash
# Check audio devices
arecord -l
aplay -l

# Test audio recording
arecord -f cd -d 5 test.wav
aplay test.wav
```

#### Permission Issues
```bash
# Add user to audio and video groups
sudo usermod -a -G audio,video pi

# Reboot or logout/login
```

#### Performance Issues
- Reduce camera resolution in config
- Lower FPS setting
- Close unnecessary applications
- Use faster SD card (Class 10+)

### Logs and Debugging

```bash
# View application logs
tail -f commercial_detector.log

# Check system logs
sudo journalctl -u commercial-detector -f

# Test individual components
python3 -c "from tv_controller import TVController; tc = TVController(); print(tc.get_status())"
```

## Customization

### Adding New Detection Features

1. Modify `extract_visual_features()` or `extract_audio_features()` methods
2. Update `detect_commercial()` logic
3. Adjust thresholds in config file

### Training Custom Models

1. Collect training data (commercial vs. program content)
2. Implement machine learning model in `detect_commercial()`
3. Use scikit-learn or TensorFlow for model training

### Adding New TV Control Methods

1. Extend `TVControlMethod` enum
2. Add initialization method in `TVController`
3. Implement mute/unmute methods

## Performance Optimization

### For Raspberry Pi 4
- Use 4GB+ RAM
- Use fast SD card (Class 10+)
- Enable GPU memory split: `gpu_mem=128` in `/boot/config.txt`
- Overclock if needed: `arm_freq=1800` in `/boot/config.txt`

### For Lower-End Hardware
- Reduce camera resolution to 320x240
- Lower FPS to 15
- Use audio-only detection
- Disable visual features

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For issues and questions:
1. Check the troubleshooting section
2. Review the logs
3. Open an issue on GitHub
4. Check the wiki for additional documentation

## Roadmap

- [ ] Machine learning model training
- [ ] Web interface for configuration
- [ ] Mobile app for remote control
- [ ] Support for multiple TVs
- [ ] Cloud-based commercial database
- [ ] Integration with streaming services