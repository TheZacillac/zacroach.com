#!/usr/bin/env python3
"""
TV Control Module for Commercial Detection System
Supports multiple methods for controlling TV muting/unmuting
"""

import logging
import subprocess
import requests
import time
from typing import Optional
from enum import Enum
import configparser
import os

logger = logging.getLogger(__name__)

class TVControlMethod(Enum):
    """Available TV control methods"""
    HDMI_CEC = "hdmi_cec"
    IR_BLASTER = "ir_blaster"
    SMART_TV_API = "smart_tv_api"
    AUDIO_ONLY = "audio_only"

class TVController:
    """Controller for TV muting/unmuting operations"""
    
    def __init__(self, config_file: str = "config.ini"):
        self.config = self._load_config(config_file)
        self.method = TVControlMethod(self.config.get('tv_control', 'method', fallback='audio_only'))
        self.is_muted = False
        
        # Initialize based on method
        if self.method == TVControlMethod.HDMI_CEC:
            self._init_hdmi_cec()
        elif self.method == TVControlMethod.IR_BLASTER:
            self._init_ir_blaster()
        elif self.method == TVControlMethod.SMART_TV_API:
            self._init_smart_tv_api()
        elif self.method == TVControlMethod.AUDIO_ONLY:
            self._init_audio_control()
        
        logger.info(f"TV Controller initialized with method: {self.method.value}")
    
    def _load_config(self, config_file: str) -> configparser.ConfigParser:
        """Load configuration from file"""
        config = configparser.ConfigParser()
        if os.path.exists(config_file):
            config.read(config_file)
        return config
    
    def _init_hdmi_cec(self):
        """Initialize HDMI-CEC control"""
        try:
            # Check if cec-client is available
            result = subprocess.run(['which', 'cec-client'], capture_output=True, text=True)
            if result.returncode != 0:
                logger.warning("cec-client not found. Install with: sudo apt-get install cec-utils")
                return False
            
            self.cec_device = self.config.getint('tv_control', 'cec_device', fallback=0)
            logger.info(f"HDMI-CEC initialized for device {self.cec_device}")
            return True
            
        except Exception as e:
            logger.error(f"HDMI-CEC initialization failed: {e}")
            return False
    
    def _init_ir_blaster(self):
        """Initialize IR blaster control"""
        try:
            self.ir_device = self.config.get('tv_control', 'ir_device', fallback='/dev/ttyUSB0')
            
            # Check if device exists
            if not os.path.exists(self.ir_device):
                logger.warning(f"IR device {self.ir_device} not found")
                return False
            
            logger.info(f"IR blaster initialized for device {self.ir_device}")
            return True
            
        except Exception as e:
            logger.error(f"IR blaster initialization failed: {e}")
            return False
    
    def _init_smart_tv_api(self):
        """Initialize smart TV API control"""
        try:
            self.tv_ip = self.config.get('tv_control', 'tv_ip', fallback='192.168.1.100')
            self.tv_port = self.config.getint('tv_control', 'tv_port', fallback=8080)
            self.tv_url = f"http://{self.tv_ip}:{self.tv_port}"
            
            # Test connection
            response = requests.get(f"{self.tv_url}/status", timeout=5)
            if response.status_code == 200:
                logger.info(f"Smart TV API initialized for {self.tv_url}")
                return True
            else:
                logger.warning(f"Smart TV API connection failed: {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"Smart TV API initialization failed: {e}")
            return False
    
    def _init_audio_control(self):
        """Initialize audio-only control"""
        try:
            self.audio_device = self.config.get('tv_control', 'audio_device', fallback='default')
            
            # Check if pulseaudio is available
            result = subprocess.run(['which', 'pactl'], capture_output=True, text=True)
            if result.returncode == 0:
                logger.info("Audio control initialized with PulseAudio")
                return True
            
            # Check if alsamixer is available
            result = subprocess.run(['which', 'amixer'], capture_output=True, text=True)
            if result.returncode == 0:
                logger.info("Audio control initialized with ALSA")
                return True
            
            logger.warning("No audio control system found (PulseAudio or ALSA)")
            return False
            
        except Exception as e:
            logger.error(f"Audio control initialization failed: {e}")
            return False
    
    def mute_tv(self) -> bool:
        """Mute the TV"""
        if self.is_muted:
            logger.debug("TV already muted")
            return True
        
        try:
            success = False
            
            if self.method == TVControlMethod.HDMI_CEC:
                success = self._mute_hdmi_cec()
            elif self.method == TVControlMethod.IR_BLASTER:
                success = self._mute_ir_blaster()
            elif self.method == TVControlMethod.SMART_TV_API:
                success = self._mute_smart_tv_api()
            elif self.method == TVControlMethod.AUDIO_ONLY:
                success = self._mute_audio()
            
            if success:
                self.is_muted = True
                logger.info("TV muted successfully")
            else:
                logger.error("Failed to mute TV")
            
            return success
            
        except Exception as e:
            logger.error(f"Error muting TV: {e}")
            return False
    
    def unmute_tv(self) -> bool:
        """Unmute the TV"""
        if not self.is_muted:
            logger.debug("TV already unmuted")
            return True
        
        try:
            success = False
            
            if self.method == TVControlMethod.HDMI_CEC:
                success = self._unmute_hdmi_cec()
            elif self.method == TVControlMethod.IR_BLASTER:
                success = self._unmute_ir_blaster()
            elif self.method == TVControlMethod.SMART_TV_API:
                success = self._unmute_smart_tv_api()
            elif self.method == TVControlMethod.AUDIO_ONLY:
                success = self._unmute_audio()
            
            if success:
                self.is_muted = False
                logger.info("TV unmuted successfully")
            else:
                logger.error("Failed to unmute TV")
            
            return success
            
        except Exception as e:
            logger.error(f"Error unmuting TV: {e}")
            return False
    
    def _mute_hdmi_cec(self) -> bool:
        """Mute TV using HDMI-CEC"""
        try:
            # Send mute command via cec-client
            cmd = f"echo 'tx 1F:36' | cec-client -s -d {self.cec_device}"
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            
            if result.returncode == 0:
                logger.debug("HDMI-CEC mute command sent")
                return True
            else:
                logger.error(f"HDMI-CEC mute failed: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"HDMI-CEC mute error: {e}")
            return False
    
    def _unmute_hdmi_cec(self) -> bool:
        """Unmute TV using HDMI-CEC"""
        try:
            # Send unmute command via cec-client
            cmd = f"echo 'tx 1F:36' | cec-client -s -d {self.cec_device}"
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            
            if result.returncode == 0:
                logger.debug("HDMI-CEC unmute command sent")
                return True
            else:
                logger.error(f"HDMI-CEC unmute failed: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"HDMI-CEC unmute error: {e}")
            return False
    
    def _mute_ir_blaster(self) -> bool:
        """Mute TV using IR blaster"""
        try:
            # This is a placeholder - you would need to implement
            # IR command sending based on your specific IR blaster hardware
            # and TV remote codes
            
            # Example using lirc (Linux Infrared Remote Control)
            # subprocess.run(['irsend', 'SEND_ONCE', 'tv', 'KEY_MUTE'], check=True)
            
            logger.warning("IR blaster mute not implemented - requires hardware-specific code")
            return False
            
        except Exception as e:
            logger.error(f"IR blaster mute error: {e}")
            return False
    
    def _unmute_ir_blaster(self) -> bool:
        """Unmute TV using IR blaster"""
        try:
            # This is a placeholder - you would need to implement
            # IR command sending based on your specific IR blaster hardware
            # and TV remote codes
            
            logger.warning("IR blaster unmute not implemented - requires hardware-specific code")
            return False
            
        except Exception as e:
            logger.error(f"IR blaster unmute error: {e}")
            return False
    
    def _mute_smart_tv_api(self) -> bool:
        """Mute TV using smart TV API"""
        try:
            # Example API call - adjust based on your TV's API
            response = requests.post(
                f"{self.tv_url}/audio/mute",
                json={"mute": True},
                timeout=5
            )
            
            if response.status_code == 200:
                logger.debug("Smart TV API mute command sent")
                return True
            else:
                logger.error(f"Smart TV API mute failed: {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"Smart TV API mute error: {e}")
            return False
    
    def _unmute_smart_tv_api(self) -> bool:
        """Unmute TV using smart TV API"""
        try:
            # Example API call - adjust based on your TV's API
            response = requests.post(
                f"{self.tv_url}/audio/mute",
                json={"mute": False},
                timeout=5
            )
            
            if response.status_code == 200:
                logger.debug("Smart TV API unmute command sent")
                return True
            else:
                logger.error(f"Smart TV API unmute failed: {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"Smart TV API unmute error: {e}")
            return False
    
    def _mute_audio(self) -> bool:
        """Mute system audio"""
        try:
            # Try PulseAudio first
            result = subprocess.run(['which', 'pactl'], capture_output=True, text=True)
            if result.returncode == 0:
                # Mute all sinks
                subprocess.run(['pactl', 'set-sink-mute', '@DEFAULT_SINK@', '1'], check=True)
                logger.debug("Audio muted via PulseAudio")
                return True
            
            # Try ALSA
            result = subprocess.run(['which', 'amixer'], capture_output=True, text=True)
            if result.returncode == 0:
                subprocess.run(['amixer', 'set', 'Master', 'mute'], check=True)
                logger.debug("Audio muted via ALSA")
                return True
            
            logger.error("No audio control system available")
            return False
            
        except Exception as e:
            logger.error(f"Audio mute error: {e}")
            return False
    
    def _unmute_audio(self) -> bool:
        """Unmute system audio"""
        try:
            # Try PulseAudio first
            result = subprocess.run(['which', 'pactl'], capture_output=True, text=True)
            if result.returncode == 0:
                # Unmute all sinks
                subprocess.run(['pactl', 'set-sink-mute', '@DEFAULT_SINK@', '0'], check=True)
                logger.debug("Audio unmuted via PulseAudio")
                return True
            
            # Try ALSA
            result = subprocess.run(['which', 'amixer'], capture_output=True, text=True)
            if result.returncode == 0:
                subprocess.run(['amixer', 'set', 'Master', 'unmute'], check=True)
                logger.debug("Audio unmuted via ALSA")
                return True
            
            logger.error("No audio control system available")
            return False
            
        except Exception as e:
            logger.error(f"Audio unmute error: {e}")
            return False
    
    def get_status(self) -> dict:
        """Get current TV control status"""
        return {
            "method": self.method.value,
            "is_muted": self.is_muted,
            "available": True  # Could be enhanced to check actual availability
        }

# Example usage and testing
if __name__ == "__main__":
    import logging
    
    logging.basicConfig(level=logging.INFO)
    
    # Test TV controller
    controller = TVController()
    
    print("Testing TV Controller...")
    print(f"Method: {controller.method.value}")
    print(f"Status: {controller.get_status()}")
    
    # Test mute/unmute
    print("\nTesting mute...")
    controller.mute_tv()
    time.sleep(2)
    
    print("Testing unmute...")
    controller.unmute_tv()
    
    print("Test completed")