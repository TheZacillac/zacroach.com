#!/usr/bin/env python3
"""
Commercial Detection System for Raspberry Pi
Automatically mutes TV during commercials using AI detection
"""

import cv2
import numpy as np
import sounddevice as sd
import librosa
import threading
import time
import logging
from typing import Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import configparser
import os
from tv_controller import TVController

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('commercial_detector.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class DetectionState(Enum):
    """States for commercial detection"""
    PROGRAM = "program"
    COMMERCIAL = "commercial"
    TRANSITION = "transition"

@dataclass
class DetectionConfig:
    """Configuration for commercial detection"""
    # Camera settings
    camera_index: int = 0
    frame_width: int = 640
    frame_height: int = 480
    fps: int = 30
    
    # Audio settings
    sample_rate: int = 44100
    channels: int = 2
    audio_buffer_size: int = 1024
    
    # Detection thresholds
    visual_change_threshold: float = 0.3
    audio_volume_threshold: float = 0.7
    commercial_duration_min: float = 15.0  # seconds
    commercial_duration_max: float = 180.0  # seconds
    
    # Detection confidence
    confidence_threshold: float = 0.8
    transition_buffer: float = 2.0  # seconds

class CommercialDetector:
    """Main class for detecting commercials and controlling TV"""
    
    def __init__(self, config: DetectionConfig):
        self.config = config
        self.state = DetectionState.PROGRAM
        self.detection_confidence = 0.0
        self.last_state_change = time.time()
        self.is_running = False
        
        # Initialize components
        self.camera = None
        self.audio_stream = None
        self.frame_buffer = []
        self.audio_buffer = []
        self.tv_controller = TVController()
        
        # Detection history
        self.detection_history = []
        self.visual_features = []
        self.audio_features = []
        
        logger.info("Commercial Detector initialized")
    
    def initialize_camera(self) -> bool:
        """Initialize camera capture"""
        try:
            self.camera = cv2.VideoCapture(self.config.camera_index)
            if not self.camera.isOpened():
                logger.error(f"Failed to open camera {self.config.camera_index}")
                return False
            
            # Set camera properties
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.frame_width)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.frame_height)
            self.camera.set(cv2.CAP_PROP_FPS, self.config.fps)
            
            logger.info(f"Camera initialized: {self.config.frame_width}x{self.config.frame_height} @ {self.config.fps}fps")
            return True
            
        except Exception as e:
            logger.error(f"Camera initialization failed: {e}")
            return False
    
    def initialize_audio(self) -> bool:
        """Initialize audio capture"""
        try:
            self.audio_stream = sd.InputStream(
                samplerate=self.config.sample_rate,
                channels=self.config.channels,
                blocksize=self.config.audio_buffer_size,
                callback=self._audio_callback
            )
            
            logger.info(f"Audio stream initialized: {self.config.sample_rate}Hz, {self.config.channels} channels")
            return True
            
        except Exception as e:
            logger.error(f"Audio initialization failed: {e}")
            return False
    
    def _audio_callback(self, indata, frames, time, status):
        """Audio callback for real-time processing"""
        if status:
            logger.warning(f"Audio callback status: {status}")
        
        # Convert to numpy array and store in buffer
        audio_data = indata.copy()
        self.audio_buffer.append(audio_data)
        
        # Keep buffer size manageable
        if len(self.audio_buffer) > 100:  # Keep last 100 audio chunks
            self.audio_buffer.pop(0)
    
    def capture_frame(self) -> Optional[np.ndarray]:
        """Capture a single frame from camera"""
        if self.camera is None:
            return None
        
        ret, frame = self.camera.read()
        if not ret:
            logger.warning("Failed to capture frame")
            return None
        
        return frame
    
    def extract_visual_features(self, frame: np.ndarray) -> np.ndarray:
        """Extract visual features for commercial detection"""
        # Convert to grayscale for analysis
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Calculate various visual features
        features = []
        
        # 1. Brightness and contrast
        brightness = np.mean(gray)
        contrast = np.std(gray)
        features.extend([brightness, contrast])
        
        # 2. Edge density (commercials often have more dynamic visuals)
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / (frame.shape[0] * frame.shape[1])
        features.append(edge_density)
        
        # 3. Color histogram features
        hist_b = cv2.calcHist([frame], [0], None, [16], [0, 256])
        hist_g = cv2.calcHist([frame], [1], None, [16], [0, 256])
        hist_r = cv2.calcHist([frame], [2], None, [16], [0, 256])
        
        # Normalize histograms
        hist_b = hist_b.flatten() / np.sum(hist_b)
        hist_g = hist_g.flatten() / np.sum(hist_g)
        hist_r = hist_r.flatten() / np.sum(hist_r)
        
        features.extend(hist_b.tolist())
        features.extend(hist_g.tolist())
        features.extend(hist_r.tolist())
        
        # 4. Motion features (if we have previous frame)
        if len(self.frame_buffer) > 0:
            prev_gray = cv2.cvtColor(self.frame_buffer[-1], cv2.COLOR_BGR2GRAY)
            diff = cv2.absdiff(gray, prev_gray)
            motion = np.mean(diff)
            features.append(motion)
        else:
            features.append(0.0)
        
        return np.array(features)
    
    def extract_audio_features(self, audio_data: np.ndarray) -> np.ndarray:
        """Extract audio features for commercial detection"""
        if len(audio_data) == 0:
            return np.zeros(10)  # Return zero features if no audio
        
        # Convert to mono if stereo
        if len(audio_data.shape) > 1:
            audio_mono = np.mean(audio_data, axis=1)
        else:
            audio_mono = audio_data.flatten()
        
        features = []
        
        # 1. Volume/RMS
        rms = np.sqrt(np.mean(audio_mono**2))
        features.append(rms)
        
        # 2. Spectral features
        if len(audio_mono) > 0:
            # Zero crossing rate
            zcr = np.mean(librosa.feature.zero_crossing_rate(audio_mono, hop_length=512))
            features.append(zcr)
            
            # Spectral centroid
            spectral_centroids = librosa.feature.spectral_centroid(y=audio_mono, sr=self.config.sample_rate)
            features.append(np.mean(spectral_centroids))
            
            # Spectral rolloff
            spectral_rolloff = librosa.feature.spectral_rolloff(y=audio_mono, sr=self.config.sample_rate)
            features.append(np.mean(spectral_rolloff))
            
            # MFCC features (first 5 coefficients)
            mfccs = librosa.feature.mfcc(y=audio_mono, sr=self.config.sample_rate, n_mfcc=5)
            features.extend(np.mean(mfccs, axis=1).tolist())
        else:
            features.extend([0.0] * 8)  # Fill with zeros if no audio
        
        return np.array(features)
    
    def detect_commercial(self, visual_features: np.ndarray, audio_features: np.ndarray) -> Tuple[bool, float]:
        """Detect if current content is a commercial"""
        # Simple rule-based detection (can be enhanced with ML models)
        is_commercial = False
        confidence = 0.0
        
        # Visual indicators
        if len(self.visual_features) > 0:
            # Check for sudden visual changes (common in commercials)
            visual_change = np.linalg.norm(visual_features - self.visual_features[-1])
            if visual_change > self.config.visual_change_threshold:
                confidence += 0.3
        
        # Audio indicators
        if len(self.audio_features) > 0:
            # Check for volume changes (commercials often louder)
            volume_change = abs(audio_features[0] - self.audio_features[-1][0])
            if volume_change > self.config.audio_volume_threshold:
                confidence += 0.4
        
        # Duration-based detection
        current_time = time.time()
        if self.state == DetectionState.COMMERCIAL:
            duration = current_time - self.last_state_change
            if duration > self.config.commercial_duration_min:
                confidence += 0.2
            if duration > self.config.commercial_duration_max:
                confidence -= 0.3  # Likely not a commercial if too long
        
        # State transition logic
        if confidence > self.config.confidence_threshold:
            is_commercial = True
        
        return is_commercial, confidence
    
    def update_state(self, is_commercial: bool, confidence: float):
        """Update detection state based on current analysis"""
        current_time = time.time()
        
        # State transition with buffer to avoid rapid switching
        if is_commercial and self.state != DetectionState.COMMERCIAL:
            if current_time - self.last_state_change > self.config.transition_buffer:
                self.state = DetectionState.COMMERCIAL
                self.last_state_change = current_time
                logger.info(f"Commercial detected (confidence: {confidence:.2f})")
                self.mute_tv()
        
        elif not is_commercial and self.state != DetectionState.PROGRAM:
            if current_time - self.last_state_change > self.config.transition_buffer:
                self.state = DetectionState.PROGRAM
                self.last_state_change = current_time
                logger.info("Program content detected")
                self.unmute_tv()
    
    def mute_tv(self):
        """Mute the TV"""
        logger.info("MUTING TV")
        self.tv_controller.mute_tv()
    
    def unmute_tv(self):
        """Unmute the TV"""
        logger.info("UNMUTING TV")
        self.tv_controller.unmute_tv()
    
    def run_detection_loop(self):
        """Main detection loop"""
        logger.info("Starting commercial detection loop")
        self.is_running = True
        
        while self.is_running:
            try:
                # Capture frame
                frame = self.capture_frame()
                if frame is None:
                    time.sleep(0.1)
                    continue
                
                # Extract visual features
                visual_features = self.extract_visual_features(frame)
                self.visual_features.append(visual_features)
                
                # Keep feature history manageable
                if len(self.visual_features) > 50:
                    self.visual_features.pop(0)
                
                # Extract audio features
                audio_features = np.zeros(10)  # Default
                if len(self.audio_buffer) > 0:
                    # Combine recent audio data
                    recent_audio = np.concatenate(self.audio_buffer[-5:])
                    audio_features = self.extract_audio_features(recent_audio)
                
                self.audio_features.append(audio_features)
                if len(self.audio_features) > 50:
                    self.audio_features.pop(0)
                
                # Detect commercial
                is_commercial, confidence = self.detect_commercial(visual_features, audio_features)
                
                # Update state
                self.update_state(is_commercial, confidence)
                
                # Store frame for motion detection
                self.frame_buffer.append(frame.copy())
                if len(self.frame_buffer) > 2:
                    self.frame_buffer.pop(0)
                
                # Control loop timing
                time.sleep(1.0 / self.config.fps)
                
            except Exception as e:
                logger.error(f"Error in detection loop: {e}")
                time.sleep(1.0)
    
    def start(self):
        """Start the commercial detection system"""
        logger.info("Starting Commercial Detection System")
        
        # Initialize camera
        if not self.initialize_camera():
            logger.error("Failed to initialize camera")
            return False
        
        # Initialize audio
        if not self.initialize_audio():
            logger.error("Failed to initialize audio")
            return False
        
        # Start audio stream
        self.audio_stream.start()
        
        # Start detection loop in separate thread
        detection_thread = threading.Thread(target=self.run_detection_loop)
        detection_thread.daemon = True
        detection_thread.start()
        
        logger.info("Commercial Detection System started successfully")
        return True
    
    def stop(self):
        """Stop the commercial detection system"""
        logger.info("Stopping Commercial Detection System")
        self.is_running = False
        
        if self.audio_stream:
            self.audio_stream.stop()
            self.audio_stream.close()
        
        if self.camera:
            self.camera.release()
        
        # Ensure TV is unmuted when stopping
        if self.tv_controller.is_muted:
            self.tv_controller.unmute_tv()
        
        cv2.destroyAllWindows()
        logger.info("Commercial Detection System stopped")

def load_config(config_file: str = "config.ini") -> DetectionConfig:
    """Load configuration from file"""
    config = DetectionConfig()
    
    if os.path.exists(config_file):
        parser = configparser.ConfigParser()
        parser.read(config_file)
        
        # Load camera settings
        if 'camera' in parser:
            config.camera_index = parser.getint('camera', 'index', fallback=config.camera_index)
            config.frame_width = parser.getint('camera', 'width', fallback=config.frame_width)
            config.frame_height = parser.getint('camera', 'height', fallback=config.frame_height)
            config.fps = parser.getint('camera', 'fps', fallback=config.fps)
        
        # Load audio settings
        if 'audio' in parser:
            config.sample_rate = parser.getint('audio', 'sample_rate', fallback=config.sample_rate)
            config.channels = parser.getint('audio', 'channels', fallback=config.channels)
            config.audio_buffer_size = parser.getint('audio', 'buffer_size', fallback=config.audio_buffer_size)
        
        # Load detection settings
        if 'detection' in parser:
            config.visual_change_threshold = parser.getfloat('detection', 'visual_threshold', fallback=config.visual_change_threshold)
            config.audio_volume_threshold = parser.getfloat('detection', 'audio_threshold', fallback=config.audio_volume_threshold)
            config.confidence_threshold = parser.getfloat('detection', 'confidence_threshold', fallback=config.confidence_threshold)
    
    return config

def main():
    """Main function"""
    try:
        # Load configuration
        config = load_config()
        
        # Create detector
        detector = CommercialDetector(config)
        
        # Start detection
        if detector.start():
            logger.info("Press Ctrl+C to stop")
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                logger.info("Shutdown requested")
        else:
            logger.error("Failed to start detection system")
            return 1
        
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        return 1
    finally:
        if 'detector' in locals():
            detector.stop()
    
    return 0

if __name__ == "__main__":
    exit(main())