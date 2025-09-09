#!/usr/bin/env python3
"""
System Test Script for Commercial Detection System
Tests all components to ensure proper functionality
"""

import sys
import time
import logging
from typing import Dict, List, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

class SystemTester:
    """Test suite for commercial detection system"""
    
    def __init__(self):
        self.test_results = {}
        self.passed_tests = 0
        self.total_tests = 0
    
    def run_test(self, test_name: str, test_func) -> bool:
        """Run a single test and record results"""
        self.total_tests += 1
        logger.info(f"Running test: {test_name}")
        
        try:
            result = test_func()
            if result:
                logger.info(f"✓ {test_name} - PASSED")
                self.passed_tests += 1
                self.test_results[test_name] = "PASSED"
            else:
                logger.error(f"✗ {test_name} - FAILED")
                self.test_results[test_name] = "FAILED"
            return result
        except Exception as e:
            logger.error(f"✗ {test_name} - ERROR: {e}")
            self.test_results[test_name] = f"ERROR: {e}"
            return False
    
    def test_imports(self) -> bool:
        """Test that all required modules can be imported"""
        try:
            import cv2
            import numpy as np
            import sounddevice as sd
            import librosa
            import configparser
            from tv_controller import TVController
            from commercial_detector import CommercialDetector, DetectionConfig
            return True
        except ImportError as e:
            logger.error(f"Import error: {e}")
            return False
    
    def test_camera(self) -> bool:
        """Test camera functionality"""
        try:
            import cv2
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                return False
            
            ret, frame = cap.read()
            cap.release()
            return ret and frame is not None
        except Exception:
            return False
    
    def test_audio(self) -> bool:
        """Test audio functionality"""
        try:
            import sounddevice as sd
            devices = sd.query_devices()
            return len(devices) > 0
        except Exception:
            return False
    
    def test_tv_controller(self) -> bool:
        """Test TV controller initialization"""
        try:
            from tv_controller import TVController
            controller = TVController()
            status = controller.get_status()
            return status is not None
        except Exception:
            return False
    
    def test_commercial_detector(self) -> bool:
        """Test commercial detector initialization"""
        try:
            from commercial_detector import CommercialDetector, DetectionConfig
            config = DetectionConfig()
            detector = CommercialDetector(config)
            return detector is not None
        except Exception:
            return False
    
    def test_config_loading(self) -> bool:
        """Test configuration file loading"""
        try:
            from commercial_detector import load_config
            config = load_config()
            return config is not None
        except Exception:
            return False
    
    def test_visual_features(self) -> bool:
        """Test visual feature extraction"""
        try:
            import cv2
            import numpy as np
            from commercial_detector import CommercialDetector, DetectionConfig
            
            config = DetectionConfig()
            detector = CommercialDetector(config)
            
            # Create a test frame
            test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            features = detector.extract_visual_features(test_frame)
            
            return len(features) > 0
        except Exception:
            return False
    
    def test_audio_features(self) -> bool:
        """Test audio feature extraction"""
        try:
            import numpy as np
            from commercial_detector import CommercialDetector, DetectionConfig
            
            config = DetectionConfig()
            detector = CommercialDetector(config)
            
            # Create test audio data
            test_audio = np.random.randn(1024).astype(np.float32)
            features = detector.extract_audio_features(test_audio)
            
            return len(features) > 0
        except Exception:
            return False
    
    def test_detection_logic(self) -> bool:
        """Test commercial detection logic"""
        try:
            import numpy as np
            from commercial_detector import CommercialDetector, DetectionConfig
            
            config = DetectionConfig()
            detector = CommercialDetector(config)
            
            # Add some test features
            detector.visual_features = [np.random.randn(50) for _ in range(5)]
            detector.audio_features = [np.random.randn(10) for _ in range(5)]
            
            # Test detection
            visual_features = np.random.randn(50)
            audio_features = np.random.randn(10)
            is_commercial, confidence = detector.detect_commercial(visual_features, audio_features)
            
            return isinstance(is_commercial, bool) and isinstance(confidence, float)
        except Exception:
            return False
    
    def test_hdmi_cec_availability(self) -> bool:
        """Test if HDMI-CEC tools are available"""
        try:
            import subprocess
            result = subprocess.run(['which', 'cec-client'], capture_output=True, text=True)
            return result.returncode == 0
        except Exception:
            return False
    
    def test_ir_blaster_availability(self) -> bool:
        """Test if IR blaster tools are available"""
        try:
            import subprocess
            result = subprocess.run(['which', 'irsend'], capture_output=True, text=True)
            return result.returncode == 0
        except Exception:
            return False
    
    def test_audio_control_availability(self) -> bool:
        """Test if audio control tools are available"""
        try:
            import subprocess
            pactl_result = subprocess.run(['which', 'pactl'], capture_output=True, text=True)
            amixer_result = subprocess.run(['which', 'amixer'], capture_output=True, text=True)
            return pactl_result.returncode == 0 or amixer_result.returncode == 0
        except Exception:
            return False
    
    def run_all_tests(self) -> Dict[str, str]:
        """Run all tests and return results"""
        logger.info("Starting system tests...")
        logger.info("=" * 50)
        
        # Core functionality tests
        self.run_test("Import Dependencies", self.test_imports)
        self.run_test("Camera Access", self.test_camera)
        self.run_test("Audio Access", self.test_audio)
        self.run_test("TV Controller", self.test_tv_controller)
        self.run_test("Commercial Detector", self.test_commercial_detector)
        self.run_test("Config Loading", self.test_config_loading)
        
        # Feature extraction tests
        self.run_test("Visual Feature Extraction", self.test_visual_features)
        self.run_test("Audio Feature Extraction", self.test_audio_features)
        self.run_test("Detection Logic", self.test_detection_logic)
        
        # Optional component tests
        self.run_test("HDMI-CEC Tools", self.test_hdmi_cec_availability)
        self.run_test("IR Blaster Tools", self.test_ir_blaster_availability)
        self.run_test("Audio Control Tools", self.test_audio_control_availability)
        
        return self.test_results
    
    def print_summary(self):
        """Print test summary"""
        logger.info("=" * 50)
        logger.info("TEST SUMMARY")
        logger.info("=" * 50)
        
        for test_name, result in self.test_results.items():
            status_icon = "✓" if result == "PASSED" else "✗"
            logger.info(f"{status_icon} {test_name}: {result}")
        
        logger.info("=" * 50)
        logger.info(f"Tests passed: {self.passed_tests}/{self.total_tests}")
        
        if self.passed_tests == self.total_tests:
            logger.info("🎉 All tests passed! System is ready to use.")
            return True
        else:
            failed_tests = self.total_tests - self.passed_tests
            logger.warning(f"⚠️  {failed_tests} test(s) failed. Check the issues above.")
            return False

def main():
    """Main test function"""
    tester = SystemTester()
    results = tester.run_all_tests()
    success = tester.print_summary()
    
    if success:
        logger.info("\nNext steps:")
        logger.info("1. Edit config.ini to match your hardware")
        logger.info("2. Run: python3 commercial_detector.py")
        logger.info("3. Check logs: tail -f commercial_detector.log")
        return 0
    else:
        logger.error("\nPlease fix the failed tests before running the system.")
        return 1

if __name__ == "__main__":
    sys.exit(main())