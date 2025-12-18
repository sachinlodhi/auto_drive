#!/usr/bin/env python3
"""
SENSOR-FUSION AUTONOMOUS DRIVING SYSTEM
========================================

Multi-modal sensor-based autonomous navigation using:
  - Semantic Segmentation → Lane following, road detection
  - LiDAR → 3D obstacle distance measurement
  - Radar → Moving object velocity tracking
  - YOLO → Real-time object classification
  - RGB Camera → Visual perception for YOLO

Advanced autonomous driving with sensor fusion and computer vision.
"""

import os
import sys

# Add CARLA agents to path
try:
    # /home/sachin/Desktop/auto/auto_vlm/main.py -> /home/sachin/Desktop/auto
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Add CARLA_Latest/PythonAPI/carla
    carla_agents_path = os.path.join(project_root, 'CARLA_Latest', 'PythonAPI', 'carla')
    if os.path.exists(carla_agents_path):
        sys.path.append(carla_agents_path)
        print(f"✅ Added CARLA agents path: {carla_agents_path}")
    else:
        print(f"⚠️  Could not find CARLA agents at: {carla_agents_path}")
        
except IndexError:
    pass

from autonomous_driver import SensorFusionDriver

if __name__ == "__main__":
    os.environ["QT_QPA_PLATFORM"] = "xcb"
    
    print("="*60)
    print("AUTONOMOUS DRIVING SYSTEM - SENSOR FUSION")
    print("Multi-Modal Perception & Real-Time Control")
    print("="*60)

    driver = SensorFusionDriver()
    driver.run()
