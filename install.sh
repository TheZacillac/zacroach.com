#!/bin/bash

# Commercial Detection System Installation Script for Raspberry Pi
# This script automates the installation process

set -e  # Exit on any error

echo "=========================================="
echo "Commercial Detection System Installer"
echo "=========================================="

# Check if running on Raspberry Pi
if ! grep -q "Raspberry Pi" /proc/cpuinfo; then
    echo "Warning: This script is designed for Raspberry Pi"
    read -p "Continue anyway? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Update system packages
echo "Updating system packages..."
sudo apt update && sudo apt upgrade -y

# Install required system packages
echo "Installing system dependencies..."
sudo apt install -y \
    python3-pip \
    python3-venv \
    python3-dev \
    git \
    libopencv-dev \
    python3-opencv \
    portaudio19-dev \
    python3-pyaudio \
    libasound2-dev \
    cec-utils \
    lirc \
    v4l-utils \
    alsa-utils

# Create virtual environment
echo "Creating Python virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install Python dependencies
echo "Installing Python dependencies..."
pip install -r requirements.txt

# Set up audio permissions
echo "Setting up audio permissions..."
sudo usermod -a -G audio,video $USER

# Create systemd service file
echo "Creating systemd service..."
sudo tee /etc/systemd/system/commercial-detector.service > /dev/null <<EOF
[Unit]
Description=Commercial Detection System
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=$(pwd)
Environment=PATH=$(pwd)/venv/bin
ExecStart=$(pwd)/venv/bin/python commercial_detector.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

# Enable service (but don't start yet)
sudo systemctl daemon-reload
sudo systemctl enable commercial-detector

# Create desktop shortcut
echo "Creating desktop shortcut..."
cat > ~/Desktop/commercial-detector.desktop <<EOF
[Desktop Entry]
Version=1.0
Type=Application
Name=Commercial Detector
Comment=Start Commercial Detection System
Exec=$(pwd)/venv/bin/python $(pwd)/commercial_detector.py
Icon=applications-multimedia
Terminal=true
Categories=AudioVideo;
EOF

chmod +x ~/Desktop/commercial-detector.desktop

# Test camera
echo "Testing camera..."
if python3 -c "import cv2; cap = cv2.VideoCapture(0); print('Camera OK' if cap.isOpened() else 'Camera FAIL'); cap.release()" 2>/dev/null; then
    echo "✓ Camera test passed"
else
    echo "✗ Camera test failed - check camera connection"
fi

# Test audio
echo "Testing audio..."
if python3 -c "import sounddevice as sd; print('Audio OK')" 2>/dev/null; then
    echo "✓ Audio test passed"
else
    echo "✗ Audio test failed - check audio setup"
fi

# Test TV controller
echo "Testing TV controller..."
if python3 -c "from tv_controller import TVController; tc = TVController(); print('TV Controller OK')" 2>/dev/null; then
    echo "✓ TV controller test passed"
else
    echo "✗ TV controller test failed"
fi

echo ""
echo "=========================================="
echo "Installation Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Reboot your Raspberry Pi to apply audio/video group changes"
echo "2. Edit config.ini to match your hardware setup"
echo "3. Test the system: python3 commercial_detector.py"
echo "4. Start the service: sudo systemctl start commercial-detector"
echo ""
echo "Configuration file: $(pwd)/config.ini"
echo "Log file: $(pwd)/commercial_detector.log"
echo "Service: sudo systemctl status commercial-detector"
echo ""
echo "For help, see README.md or run: python3 commercial_detector.py --help"
echo ""