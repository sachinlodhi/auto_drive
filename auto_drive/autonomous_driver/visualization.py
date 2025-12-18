import cv2
import numpy as np
import math
from .config import *

class Visualizer:
    """
    Handles all visualization and HUD drawing.
    """
    def __init__(self):
        pass

    def draw_main_window(self, rgb_image, control, mode, yolo_detections, show_overlay=True):
        """Main camera view with all info"""
        if rgb_image is None:
            return np.zeros((600, 800, 3), dtype=np.uint8)

        result = rgb_image.copy()

        # Info panel (optional)
        if show_overlay:
            cv2.rectangle(result, (10, 10), (350, 150), (0, 0, 0), -1)
            cv2.rectangle(result, (10, 10), (350, 150), (255, 255, 0), 2)

            cv2.putText(result, "AUTONOMOUS DRIVING", (20, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.putText(result, "Sensor Fusion System", (20, 65),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            # Mode
            mode_color = (0, 255, 0) if "CLEAR" in mode else (0, 165, 255) if "CAUTION" in mode else (0, 0, 255)
            cv2.putText(result, mode, (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, mode_color, 2)

            # Controls
            cv2.putText(result, f"Steer: {control.steer:+.2f} Throttle: {control.throttle:.2f} Brake: {control.brake:.2f}",
                       (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Draw YOLO detections
        for det in yolo_detections:
            x1, y1, x2, y2 = det['box']
            label = f"{det['class']}"
            color = (0, 255, 0)
            if det['class'] == 'person': color = (0, 0, 255)
            elif det['class'] in ['car', 'truck', 'bus']: color = (0, 255, 0)
            elif det['class'] == 'stop sign': color = (0, 0, 255)
            elif det['class'] == 'traffic light': color = (0, 165, 255)

            cv2.rectangle(result, (x1, y1), (x2, y2), color, 2)
            cv2.putText(result, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        return result

    def draw_semantic_window(self, semantic_img, semantic_data):
        """Show what the car 'sees' semantically"""
        if semantic_img is None:
            return np.zeros((300, 400, 3), dtype=np.uint8)

        # Colorize
        color_img = np.zeros((SEMANTIC_HEIGHT, SEMANTIC_WIDTH, 3), dtype=np.uint8)
        
        # We need to map tags to colors. Fast way?
        # Loop is slow in Python, but image is small (400x300).
        # Let's use numpy indexing for speed if possible, or just loop for now.
        # Actually, let's just show the raw tags as grayscale for speed, 
        # or a simplified color map.
        
        # Better: Use a fixed palette
        # Create an RGB image from the tags
        # Create an RGB image from the tags
        # Handle unknown tags by assigning a random color based on tag ID
        unique_tags = np.unique(semantic_img)
        for tag in unique_tags:
            if tag in SEMANTIC_COLORS:
                color = SEMANTIC_COLORS[tag][1]
            else:
                # Generate random color for unknown tags
                np.random.seed(int(tag))
                color = tuple(np.random.randint(50, 255, 3).tolist())
            
            mask = (semantic_img == tag)
            color_img[mask] = color

        # Overlay info
        cv2.putText(color_img, "SEMANTIC - Lane Following", (10, 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        status = "ROAD VISIBLE" if semantic_data['road_visible'] else "NO ROAD"
        color = (0, 255, 0) if semantic_data['road_visible'] else (0, 0, 255)
        cv2.putText(color_img, status, (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        cv2.putText(color_img, f"Offset: {semantic_data['road_center_offset']:+.2f}", (10, 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Draw center line
        center_x = int(SEMANTIC_WIDTH / 2)
        cv2.line(color_img, (center_x, 0), (center_x, 300), (100, 100, 100), 1)
        
        # Draw target line
        if semantic_data['road_visible']:
            target_x = int(center_x + semantic_data['road_center_offset'] * (SEMANTIC_WIDTH/2))
            cv2.line(color_img, (target_x, 0), (target_x, 300), (0, 255, 255), 2)

        # Draw Danger Zone (Center column, bottom half)
        # Draw Center Line (Yellow)
        cv2.line(color_img, (200, 150), (200, 300), (0, 255, 255), 2)
        
        # Draw Calculated Road Center (Blue)
        if 'road_center_x' in semantic_data:
             cx = int(semantic_data['road_center_x'])
             cv2.line(color_img, (cx, 150), (cx, 300), (255, 0, 0), 2)

        # Draw Vehicle Box (Red)

        return color_img

    def draw_lidar_window(self, lidar_points, lidar_data):
        """Top-down LiDAR view"""
        img = np.zeros((300, 400, 3), dtype=np.uint8)
        
        # Scale: 1 pixel = 0.2 meters (Total range 80m width)
        scale = 5.0 
        center_x = 200
        center_y = 250 # Car is at bottom center

        if lidar_points is not None:
            # Filter for display (only nearby points)
            mask = (lidar_points[:, 0] > -10) & (lidar_points[:, 0] < 50) & \
                   (lidar_points[:, 1] > -40) & (lidar_points[:, 1] < 40)
            points = lidar_points[mask]

            for p in points:
                x = p[0] # Forward
                y = p[1] # Left/Right
                z = p[2] # Up/Down

                # Map to screen
                # Screen X = Center X + Y * Scale
                # Screen Y = Center Y - X * Scale
                px = int(center_x + y * scale)
                py = int(center_y - x * scale)

                if 0 <= px < 400 and 0 <= py < 300:
                    # Color based on height (z)
                    # Ground is gray, obstacles are green/yellow/red
                    if z < -1.5:
                        color = (50, 50, 50) # Ground
                    else:
                        # Height map
                        if np.isnan(z):
                            val = 0
                        else:
                            val = int((z + 1.5) * 100)
                            val = max(0, min(255, val))
                        color = (0, val, 255-val) # Red to Green gradient

                    cv2.circle(img, (px, py), 1, color, -1)

        # Draw Car
        cv2.rectangle(img, (center_x-10, center_y-20), (center_x+10, center_y+20), (0, 0, 255), -1)

        # Draw detection zones
        # Front zone (Visualizing the +/- 0.75m detection width)
        # Scale is 5.0 px/m. 0.75m * 5 = 3.75 pixels.
        width_px = int(0.75 * scale)
        length_px = int(40.0 * scale) # 40m range
        cv2.rectangle(img, (center_x-width_px, center_y-length_px), (center_x+width_px, center_y), (0, 50, 0), 1)

        cv2.putText(img, "LiDAR - Obstacle Detection", (10, 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        dist = lidar_data['front_distance']
        color = (0, 255, 0) if dist > 15 else (0, 0, 255)
        cv2.putText(img, f"Front: {dist:.1f}m", (10, 280),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        return img

    def draw_radar_window(self, radar_detections, radar_data):
        """Radar view (Polar plot style)"""
        img = np.zeros((300, 400, 3), dtype=np.uint8)
        
        center_x = 200
        center_y = 280
        max_range = RADAR_RANGE
        scale = 250 / max_range

        # Draw arcs
        for r in [10, 20, 30, 40, 50]:
            radius = int(r * scale)
            cv2.ellipse(img, (center_x, center_y), (radius, radius), 0, 180, 360, (50, 50, 50), 1)

        # Draw detections
        for det in radar_detections:
            # Azimuth is in degrees
            azimuth = det['azimuth']
            dist = det['distance']
            
            # Convert to Cartesian
            # x = dist * sin(azimuth)
            # y = dist * cos(azimuth)
            rad = math.radians(azimuth)
            
            px = int(center_x + dist * scale * math.sin(rad))
            py = int(center_y - dist * scale * math.cos(rad))

            if 0 <= px < 400 and 0 <= py < 300:
                if det['velocity'] < -2:
                    color = (0, 0, 255)  # Approaching = red
                elif det['velocity'] > 2:
                    color = (0, 255, 0)  # Moving away = green
                else:
                    color = (255, 255, 0)  # Stationary = yellow
                cv2.circle(img, (px, py), 6, color, -1)

        cv2.putText(img, "RADAR - Moving Objects", (10, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        cv2.putText(img, f"Objects: {len(radar_detections)}", (280, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(img, "Red=Approaching  Green=Away", (10, 290),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1)

        return img

    def draw_telemetry_window(self, gnss_data, imu_data, vehicle_velocity, lane_invasion_count):
        """Show GPS, IMU, Speed, and other telemetry"""
        img = np.zeros((300, 400, 3), dtype=np.uint8)

        cv2.putText(img, "TELEMETRY", (10, 28),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        y = 55

        # GPS
        cv2.putText(img, "GPS:", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
        y += 22
        cv2.putText(img, f"  Lat: {gnss_data['latitude']:.6f}", (10, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
        y += 18
        cv2.putText(img, f"  Lon: {gnss_data['longitude']:.6f}", (10, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
        y += 22

        # IMU
        cv2.putText(img, "IMU:", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
        y += 20
        cv2.putText(img, f"  Compass: {imu_data['compass']:.1f} deg", (10, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
        y += 22

        # Vehicle speed
        speed_ms = math.sqrt(vehicle_velocity.x**2 + vehicle_velocity.y**2 + vehicle_velocity.z**2)
        speed_kmh = speed_ms * 3.6

        cv2.putText(img, "VEHICLE:", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
        y += 22
        speed_color = (0, 255, 0) if speed_kmh < 50 else (0, 165, 255) if speed_kmh < 80 else (0, 0, 255)
        cv2.putText(img, f"  Speed: {speed_kmh:.1f} km/h", (10, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.55, speed_color, 2)
        y += 22

        # Lane invasions
        inv_color = (0, 255, 0) if lane_invasion_count == 0 else (0, 0, 255)
        cv2.putText(img, f"  Lane Invasions: {lane_invasion_count}", (10, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, inv_color, 1)

        # Draw compass - RIGHT SIDE
        compass_center = (320, 85)
        compass_radius = 55
        cv2.circle(img, compass_center, compass_radius, (100, 100, 100), 2)
        cv2.putText(img, "N", (313, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Compass needle
        heading_rad = math.radians(imu_data['compass'])
        needle_x = int(compass_center[0] + compass_radius * 0.8 * math.sin(heading_rad))
        needle_y = int(compass_center[1] - compass_radius * 0.8 * math.cos(heading_rad))
        cv2.arrowedLine(img, compass_center, (needle_x, needle_y), (0, 0, 255), 3)

        return img

    def create_map_display(self, route, current_wp_index, vehicle_loc, vehicle_yaw):
        """Create a 2D map visualization"""
        display = np.zeros((300, 400, 3), dtype=np.uint8)
        
        # Draw title
        cv2.putText(display, "GLOBAL MAP", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        if not route:
            cv2.putText(display, "No Route", (150, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
            return display
            
        # Simple visualization: Draw the route relative to ego
        # Center is (200, 250) representing the car
        center_x, center_y = 200, 250
        scale = 5.0 # pixels per meter
        
        my_yaw_rad = math.radians(vehicle_yaw)
        
        # Draw Ego Car
        cv2.arrowedLine(display, (center_x, center_y+10), (center_x, center_y-10), (0, 255, 0), 3)
        
        # Draw Waypoints
        for i, (wp, _) in enumerate(route):
            if i < current_wp_index:
                continue
            if i > current_wp_index + 50: # Only show next 50 points
                break
                
            # Relative coordinates
            dx = wp.transform.location.x - vehicle_loc.x
            dy = wp.transform.location.y - vehicle_loc.y
            
            # Rotate to ego frame
            x_rel = dx * math.cos(my_yaw_rad) + dy * math.sin(my_yaw_rad)
            y_rel = -dx * math.sin(my_yaw_rad) + dy * math.cos(my_yaw_rad)
            
            # Map to screen (Up is negative Y)
            screen_x = int(center_x + y_rel * scale)
            screen_y = int(center_y - x_rel * scale)
            
            if 0 <= screen_x < 400 and 0 <= screen_y < 300:
                color = (0, 255, 255) # Yellow path
                if i == len(route) - 1:
                    color = (0, 0, 255) # Red destination
                    cv2.circle(display, (screen_x, screen_y), 8, color, -1)
                else:
                    cv2.circle(display, (screen_x, screen_y), 2, color, -1)
                    
        return display
