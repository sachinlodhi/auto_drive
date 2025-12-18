import cv2
import numpy as np
import math
import time
from .config import *

try:
    from ultralytics import YOLO
    HAVE_YOLO = True
except ImportError:
    HAVE_YOLO = False
    print("YOLO not installed (pip install ultralytics)")

class PerceptionManager:
    """
    Processes raw sensor data into meaningful information.
    """
    def __init__(self):
        # YOLO
        self.yolo = None
        if HAVE_YOLO:
            try:
                self.yolo = YOLO('yolov8x.pt') # Upgraded to Extra Large model
                print("YOLOv8 Extra Large (x) loaded")
            except Exception as e:
                print(f"YOLO failed: {e}")
        
        # Data storage
        self.semantic_data = {
            'road_center_offset': 0.0,
            'road_visible': False,
            'road_width': 0,
            'pedestrian_danger': False,
            'vehicle_ahead': False,
            'traffic_light_visible': False,
            'left_lane_clear': True,
            'right_lane_clear': True,
        }

        self.lidar_data = {
            'front_distance': 100.0,
            'left_distance': 100.0,
            'right_distance': 100.0,
            'obstacle_detected': False,
        }

        self.radar_data = {
            'approaching_objects': [],
            'fastest_approach_speed': 0.0,
            'closest_moving': 100.0,
        }

        self.yolo_data = {
            'detections': [],
            'person_close': False,
            'person_very_close': False,  # Emergency stop level
            'vehicle_close': False,
            'vehicle_very_close': False,
            'stop_sign': False,
            'traffic_light_status': None,
        }
        
        self.last_radar_detections = []
        
        # Timestamps for stale data checking
        self.last_lidar_time = 0
        self.last_radar_time = 0

    def check_stale_data(self, current_time):
        """Clear sensor data if it's too old (>0.5s)"""
        # Radar
        if current_time - self.last_radar_time > 0.5:
            self.radar_data['approaching_objects'] = []
            self.radar_data['fastest_approach_speed'] = 0.0
            self.radar_data['closest_moving'] = 100.0
            self.last_radar_detections = []
            
        # LiDAR (Optional, but good for safety)
        if current_time - self.last_lidar_time > 0.5:
            self.lidar_data['front_distance'] = 100.0
            self.lidar_data['obstacle_detected'] = False

    def process_yolo(self, rgb_image, frame_count):
        """Run YOLO object detection"""
        if self.yolo is None:
            return []

        # Mask out the hood (bottom 120 pixels) to prevent self-detection
        height, width = rgb_image.shape[:2]
        roi = rgb_image.copy()
        roi[height-120:, :] = 0 

        results = self.yolo(roi, verbose=False)
        detections = []
        
        # Reset flags
        self.yolo_data['person_close'] = False
        self.yolo_data['person_very_close'] = False
        self.yolo_data['vehicle_close'] = False
        self.yolo_data['vehicle_very_close'] = False
        self.yolo_data['stop_sign'] = False
        self.yolo_data['traffic_light_status'] = None

        for r in results:
            boxes = r.boxes
            for box in boxes:
                # Bounding Box
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                
                # Class
                cls = int(box.cls[0])
                name = self.yolo.names[cls]
                conf = float(box.conf[0])

                if conf < 0.4:
                    continue

                detections.append({
                    'box': [x1, y1, x2, y2],
                    'class': name,
                    'conf': conf
                })

                # Logic
                area = (x2 - x1) * (y2 - y1)
                center_x = (x1 + x2) / 2
                
                # STRICT path check: 200 < x < 600 (Center 400px of 800px image)
                # Widened from 350-450 to catch crossing pedestrians earlier
                in_path = 200 < center_x < 600

                if name == 'person':
                    # DISTANCE-AWARE path zone for pedestrians:
                    # - Far pedestrians (small): narrow zone (only dead center)
                    # - Close pedestrians (large): wider zone
                    # This prevents braking for pedestrians clearly on the side of road
                    if area > 5000:
                        # Close - wide path zone
                        person_in_path = 200 < center_x < 600
                    elif area > 2500:
                        # Medium - medium zone
                        person_in_path = 280 < center_x < 520
                    else:
                        # Far - narrow zone (only if dead center)
                        person_in_path = 320 < center_x < 480

                    if person_in_path:
                        # Two-level detection:
                        # - person_very_close: EMERGENCY STOP (area > 5000 = ~5m away)
                        # - person_close: CAUTION/SLOW (area > 2000 = ~10m away)
                        if area > 5000:
                            print(f"[!] YOLO Person VERY CLOSE! Area={area:.0f}")
                            self.yolo_data['person_very_close'] = True
                            self.yolo_data['person_close'] = True
                        elif area > 2000:
                            if frame_count % 10 == 0:
                                print(f"[P] YOLO Person detected. Area={area:.0f} center={center_x:.0f}")
                            self.yolo_data['person_close'] = True
                
                if name in ['car', 'truck', 'bus', 'motorcycle']:
                    # BALANCED CHECK: Not too strict (miss obstacles) or too loose (stop for parked cars)
                    #
                    # Path zone: center 400px of 800px image (x: 200-600)
                    #
                    # Check 1: CENTER of vehicle is IN the path zone
                    center_in_path = 200 < center_x < 600

                    # Check 2: CENTER is NEAR path and vehicle has MAJOR overlap (>50%)
                    center_near_path = 150 < center_x < 650
                    box_width = x2 - x1
                    overlap_start = max(x1, 200)
                    overlap_end = min(x2, 600)
                    overlap_width = max(0, overlap_end - overlap_start)
                    major_overlap = (overlap_width / max(box_width, 1)) > 0.5

                    # Check 3: HUGE vehicle (dangerous regardless) - fills most of screen
                    is_huge = area > 120000

                    # Dangerous if:
                    # - Center is IN path, OR
                    # - Center is NEAR path AND major overlap, OR
                    # - Vehicle is HUGE
                    is_dangerous = center_in_path or (center_near_path and major_overlap) or is_huge

                    # Thresholds
                    very_close_threshold = 25000 if name == 'truck' else 35000
                    close_threshold = 8000 if name == 'truck' else 12000

                    if area > very_close_threshold and is_dangerous:
                        self.yolo_data['vehicle_very_close'] = True
                        self.yolo_data['vehicle_close'] = True
                        if frame_count % 10 == 0:
                            print(f"[!] YOLO {name.upper()} VERY CLOSE! Area={area:.0f} center={center_x:.0f}")
                    elif area > close_threshold and is_dangerous:
                        self.yolo_data['vehicle_close'] = True
                        if frame_count % 10 == 0:
                            print(f"[V] YOLO {name} close. Area={area:.0f}")
                       # Check for traffic lights
                # Lowered threshold from 1000 to 200 to detect distant lights
                if name == 'traffic light' and area > 200:
                    # Crop the traffic light
                    roi = rgb_image[y1:y2, x1:x2]
                    if roi.size == 0: continue
                    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_RGB2HSV)
                    
                    # Red color range
                    lower_red1 = np.array([0, 100, 100])
                    upper_red1 = np.array([10, 255, 255])
                    lower_red2 = np.array([160, 100, 100])
                    upper_red2 = np.array([179, 255, 255])
                    mask_red = cv2.inRange(hsv_roi, lower_red1, upper_red1) + cv2.inRange(hsv_roi, lower_red2, upper_red2)
                    
                    # Green color range
                    lower_green = np.array([40, 100, 100])
                    upper_green = np.array([80, 255, 255])
                    mask_green = cv2.inRange(hsv_roi, lower_green, upper_green)

                    # Yellow color range
                    lower_yellow = np.array([20, 100, 100])
                    upper_yellow = np.array([30, 255, 255])
                    mask_yellow = cv2.inRange(hsv_roi, lower_yellow, upper_yellow)
                    
                    red_pixels = np.sum(mask_red > 0)
                    green_pixels = np.sum(mask_green > 0)
                    yellow_pixels = np.sum(mask_yellow > 0)
                    
                    total_pixels = roi.size / 3
                    if total_pixels > 0:
                        detected_color = None
                        if red_pixels > green_pixels and red_pixels > yellow_pixels and red_pixels > (total_pixels * 0.1):
                            detected_color = 'red'
                        elif green_pixels > red_pixels and green_pixels > yellow_pixels and green_pixels > (total_pixels * 0.1):
                            detected_color = 'green'
                        elif yellow_pixels > red_pixels and yellow_pixels > green_pixels and yellow_pixels > (total_pixels * 0.1):
                            detected_color = 'yellow'
                        
                        # PRIORITIZE RED > YELLOW > GREEN
                        current_status = self.yolo_data['traffic_light_status']
                        
                        if detected_color == 'red':
                            self.yolo_data['traffic_light_status'] = 'red'
                        elif detected_color == 'yellow' and current_status != 'red':
                            self.yolo_data['traffic_light_status'] = 'yellow'
                        elif detected_color == 'green' and current_status is None:
                            self.yolo_data['traffic_light_status'] = 'green'
                    
                    if self.yolo_data['traffic_light_status'] and frame_count % 30 == 0:
                        print(f"🚦 YOLO Traffic Light: {self.yolo_data['traffic_light_status']} (Area={area:.0f})")

                if name == 'stop sign' and area > 1500:
                    self.yolo_data['stop_sign'] = True

        self.yolo_data['detections'] = detections
        return detections

    def process_semantic(self, semantic_img, frame_count=0):
        """
        Process semantic segmentation image to find:
        1. Road center (for steering)
        2. Dangers (pedestrians, vehicles)
        """
        # Convert to numpy array
        array = np.frombuffer(semantic_img.raw_data, dtype=np.uint8)
        array = array.reshape((SEMANTIC_HEIGHT, SEMANTIC_WIDTH, 4))
        # Semantic tag is in the Red channel (Index 2 in BGRA)
        labels = array[:, :, 2]

        # --- 1. Lane Following Logic ---
        # Look at the bottom half of the image (closer to car = more reliable)
        roi = labels[150:300, :]

        # CARLA Semantic Labels - include multiple road-like surfaces
        # Road=7, RoadLine=6, Ground=14 (some maps use Ground for road)
        road_mask = (labels == 7) | (labels == 6) | (labels == 14)

        # Find center of road in ROI
        y_indices, x_indices = np.where(road_mask[150:300, :])

        # DEBUG: Print every 30 frames
        if frame_count % 30 == 0:
            unique, counts = np.unique(roi, return_counts=True)
            tag_dict = dict(zip(unique, counts))
            road_pixels = tag_dict.get(7, 0) + tag_dict.get(6, 0)
            print(f"Semantic Tags: {tag_dict} | Road pixels in ROI: {len(x_indices)} (need >100)")

        if len(x_indices) > 100:
            road_center_x = np.mean(x_indices)
            image_center_x = SEMANTIC_WIDTH / 2
            # Offset: -1 (left) to +1 (right)
            offset = (road_center_x - image_center_x) / (SEMANTIC_WIDTH / 2)
            
            self.semantic_data['road_center_offset'] = offset
            self.semantic_data['road_visible'] = True
            self.semantic_data['road_center_offset'] = offset
            self.semantic_data['road_visible'] = True
            self.semantic_data['road_width'] = np.max(x_indices) - np.min(x_indices)
            self.semantic_data['road_center_x'] = road_center_x # Store for visualization
        else:
            self.semantic_data['road_visible'] = False
            self.semantic_data['road_center_offset'] = 0.0

        # --- 2. Pedestrian/Vehicle Detection ---
        # Pedestrian=4, Vehicle=10
        # Only check the BOTTOM HALF of the image (close to vehicle, more relevant)
        bottom_half = labels[150:300, :]
        pedestrian_mask = (bottom_half == 4)
        vehicle_mask = (bottom_half == 10)

        ped_pixels = np.sum(pedestrian_mask)
        veh_pixels = np.sum(vehicle_mask)

        # FIX: Increased threshold to 5000 and only print every 30 frames to reduce spam
        # FIX: Use 'pedestrian_danger' to match what control.py expects
        if ped_pixels > 5000:
            if frame_count % 30 == 0:
                print(f"[!] SEMANTIC PEDESTRIAN DETECTED! Pixels: {ped_pixels}")
            self.semantic_data['pedestrian_danger'] = True
        else:
            self.semantic_data['pedestrian_danger'] = False

        # Vehicle detection - increased threshold and only in bottom half
        self.semantic_data['vehicle_ahead'] = veh_pixels > 5000

        return labels # Return for visualization

    def process_lidar(self, lidar_data, steering=0.0, frame_count=0):
        """
        Process LiDAR point cloud to find obstacles
        """
        self.last_lidar_time = time.time()
        
        # Convert to numpy
        points = np.frombuffer(lidar_data.raw_data, dtype=np.float32)
        points = np.reshape(points, (int(points.shape[0] / 4), 4))
        
        # Filter: Keep only points in front (x > 0)
        # x = forward, y = left/right, z = up/down

        # Ground removal:
        # Sensor is at z=1.2 relative to ground. So ground is at z=-1.2.
        # FIX: More aggressive ground removal - keep z > -0.5 (was -0.8)
        # Also filter out very high points (overhead signs, bridges)
        non_ground = points[(points[:, 2] > -0.5) & (points[:, 2] < 2.0)]

        # CURVED DETECTION ZONE
        # Shift the center of the detection box based on steering angle
        visual_offset = self.semantic_data.get('road_center_offset', 0.0)

        # Weighted average: 40% Steering, 60% Visual (Visual sees the curve first)
        combined_steer = (steering * 0.4) + (visual_offset * 0.6)

        y_shift = non_ground[:, 0] * combined_steer * 0.5

        # SAFETY FIX: Use WIDER corridor at close range, narrower at far range
        # Close range needs to catch vehicles (trucks are 2.5m wide)
        # At 2m: width = 1.2m (wide - catch everything close)
        # At 15m: width = 0.8m (medium)
        # At 40m: width = 1.0m (wide again for lane changes)
        distance = non_ground[:, 0]
        corridor_width = np.where(
            distance < 15.0,
            1.2 - (distance / 15.0) * 0.4,  # 1.2m at 2m, 0.8m at 15m
            0.8 + ((distance - 15.0) / 25.0) * 0.2  # 0.8m at 15m, 1.0m at 40m
        )

        # FIX: Reduced min distance from 1.5m to 1.0m for earlier detection
        front_mask = (non_ground[:, 0] > 1.0) & \
                     (non_ground[:, 0] < 40) & \
                     (np.abs(non_ground[:, 1] - y_shift) < corridor_width)
        front_points = non_ground[front_mask]

        # Wall detection - ONLY for objects clearly on the side, never for center obstacles
        is_wall = False
        if len(front_points) > 25:
            mean_y = np.mean(front_points[:, 1])
            std_y = np.std(front_points[:, 1])
            min_dist = np.min(front_points[:, 0])

            # Wall must be CLEARLY on the side (|mean_y| > 0.6) - increased from 0.4
            # AND very uniform (std < 0.25) AND far (> 8m) AND many points
            is_clearly_on_side = abs(mean_y) > 0.6
            is_very_uniform = std_y < 0.25
            is_far = min_dist > 8.0
            has_many_points = len(front_points) > 60

            if is_clearly_on_side and is_very_uniform and is_far and has_many_points:
                is_wall = True

            # SAFETY: NEVER classify as wall if close or centered
            if min_dist < 5.0 or abs(mean_y) < 0.5:
                is_wall = False

        if len(front_points) > 20 and not is_wall:
            self.lidar_data['front_distance'] = np.min(front_points[:, 0])
            self.lidar_data['obstacle_detected'] = True
        else:
            self.lidar_data['front_distance'] = 100.0
            self.lidar_data['obstacle_detected'] = False

        return points # Return for visualization

    def process_radar(self, radar_data, ego_velocity_vec):
        """
        Process Radar data for moving objects
        """
        self.last_radar_time = time.time()
        
        # Calculate scalar ego speed (m/s)
        ego_speed = math.sqrt(ego_velocity_vec.x**2 + ego_velocity_vec.y**2 + ego_velocity_vec.z**2)

        points = np.frombuffer(radar_data.raw_data, dtype=np.float32)
        points = np.reshape(points, (int(points.shape[0] / 4), 4))
        # [velocity, azimuth, altitude, depth]
        
        detections = []
        approaching = []
        fastest_speed = 0.0
        closest_moving = 100.0

        for p in points:
            velocity = p[0] # m/s
            azimuth = math.degrees(p[1])
            altitude = math.degrees(p[2])
            depth = p[3]
            
            # 1. Filter Ground/Overhead (Altitude)
            # Stricter altitude check (5 degrees)
            # Move this check BEFORE appending to detections to clean up visualization
            if abs(altitude) > 5.0: 
                continue

            det = {
                'velocity': velocity,
                'azimuth': azimuth,
                'altitude': altitude,
                'distance': depth
            }
            detections.append(det)

            # Track approaching objects (negative velocity = coming towards us)
            
            # 2. Filter Static Objects
            is_static = abs(velocity + ego_speed) < 3.0 
            
            # 3. Only detect objects approaching FASTER than 2 m/s
            is_approaching = velocity < -2.0
            
            # Stricter azimuth for distant objects
            is_in_front = abs(azimuth) < 8 
            is_crossing_close = abs(azimuth) < 20 and depth < 15.0
            
            # Only consider approaching if it's NOT static
            # CRITICAL FIX: Removed "or depth < 10.0"
            # We trust LiDAR for static obstacles (walls, stopped cars).
            # Radar should ONLY track objects that are moving relative to the world.
            if not is_static and is_approaching and (is_in_front or is_crossing_close):
                detections.append({
                    'distance': depth,
                    'velocity': velocity,
                    'azimuth': azimuth
                })
                approaching.append(det)
                
                if abs(velocity) > fastest_speed:
                    fastest_speed = abs(velocity)
                if depth < closest_moving:
                    closest_moving = depth

        self.radar_data['closest_moving'] = closest_moving
        self.last_radar_detections = detections
        
        return detections
