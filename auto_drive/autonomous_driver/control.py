import carla
import math
import numpy as np
import time
import random

class Controller:
    """
    Handles vehicle control logic (Steering, Throttle, Brake)
    based on sensor data and path planning.
    """
    def __init__(self, vehicle):
        self.vehicle = vehicle

        # Configurable max speed (can be changed from dashboard)
        self.max_speed = 45.0

        # State
        self.current_steering = 0.0
        self.current_throttle = 0.0
        self.current_brake = 0.0
        self.reason = "UNKNOWN"
        
        # Smoothing
        self.steering_history = []
        
        # Stuck recovery
        self.last_position = None
        self.stuck_counter = 0
        self.reverse_mode = False
        self.reverse_start_time = 0
        self.last_collision_time = 0
        
        # Speed PID
        self.speed_error_sum = 0.0
        self.last_speed_error = 0.0

        # Overtaking Logic
        self.blocked_timer = 0.0
        self.is_overtaking = False

        # Lateral Bias (to break stuck loops)
        self.lateral_bias = 0.0
        self.bias_timer = 0.0

        # Logging attributes for screenshot system
        self._last_p_term = 0.0
        self._last_i_term = 0.0
        self._last_d_term = 0.0
        self._last_path_steering = 0.0
        self._last_semantic_steering = 0.0
        self._last_target_speed = 0.0

    def calculate_steering(self, vehicle_transform, target_wp, semantic_data, lidar_data, imu_data, frame_count):
        """
        Calculate steering angle by fusing Path Planning + Semantic Lane Detection
        """
        if not target_wp:
            return 0.0

        # 1. PATH FOLLOWING (Global Route)
        # Calculate angle to target waypoint
        vehicle_loc = vehicle_transform.location
        vehicle_yaw = vehicle_transform.rotation.yaw
        
        target_loc = target_wp.transform.location
        
        # Vector to target
        dx = target_loc.x - vehicle_loc.x
        dy = target_loc.y - vehicle_loc.y
        
        target_yaw = math.degrees(math.atan2(dy, dx))
        diff = target_yaw - vehicle_yaw
        
        # Normalize to [-180, 180]
        while diff > 180: diff -= 360
        while diff < -180: diff += 360
        
        # More aggressive P-controller for sharp turns
        # Reduced gain from 40.0 to 55.0 to prevent cutting corners (hitting inside curb)
        path_steering = diff / 55.0
        
        # 2. SEMANTIC LANE KEEPING
        # Calculate steering
        # Semantic steering: 0.0 is center, -1.0 is left, 1.0 is right
        # REMOVED BIAS: It caused right-side curb collisions. Trust the center.
        semantic_steering = semantic_data['road_center_offset'] * 1.5
        semantic_steering = np.clip(semantic_steering, -1.0, 1.0)

        # Store for logging
        self._last_path_steering = path_steering
        self._last_semantic_steering = semantic_steering

        # OBSTACLE AVOIDANCE BIAS
        # If obstacle is close (< 15m), trust Path more (to steer around)
        # and ignore Semantic (which keeps us in lane behind obstacle)
        front_dist = lidar_data.get('front_distance', 100.0)
        
        # 3. FUSION: Combine them!
        
        # OBSTACLE AVOIDANCE BIAS
        # If obstacle is close (< 15m), trust Path more (to steer around)
        # and ignore Semantic (which keeps us in lane behind obstacle)
        # (Note: front_dist is already calculated above due to the instruction's placement)
        
        if front_dist < 15.0:
            # Obstacle ahead! Bias towards path to avoid it.
            final_steering = path_steering * 0.9 + semantic_steering * 0.1
        elif semantic_data['road_visible']:
            # 4. DECISION LOGIC
            # CONFLICT RESOLUTION: If Path and Semantic disagree on direction, TRUST PATH!
            # (e.g. Path says Left Turn, Semantic says Right because of wide intersection)
            if np.sign(path_steering) != np.sign(semantic_steering) and abs(path_steering) > 0.05:
                 final_steering = path_steering # Trust Map 100%
            # If the map says "SHARP TURN" (> 0.30 rad), trust the map 90%
            elif abs(path_steering) > 0.30:
                final_steering = path_steering * 0.9 + semantic_steering * 0.1
            else:
                # Normal driving: Trust Map 50%, Vision 50%
                final_steering = semantic_steering * 0.5 + path_steering * 0.5
        else:
            # No road visible? Trust the map!
            final_steering = path_steering

        # D: Add derivative (rate of change) for smoother control
        if self.steering_history:
            derivative = final_steering - self.steering_history[-1]
            final_steering += derivative * 0.3

        # OVERTAKE BIAS
        # If we are in Overtake Mode, force a left bias to go around
        if self.is_overtaking:
             final_steering -= 0.3 # Steer Left
             
        # LATERAL BIAS (Stuck Recovery)
        if self.bias_timer > 0:
            final_steering += self.lateral_bias
            self.bias_timer -= 1.0 / 20.0 # Assume 20 FPS
            if self.bias_timer <= 0:
                self.lateral_bias = 0.0
                print("Lateral Bias Reset")

        # USE IMU: If rotating too fast, reduce steering (stability)
        gyro_z = abs(imu_data['gyroscope']['z'])
        if gyro_z > 0.8:
            final_steering *= 0.8

        # Clamp
        final_steering = np.clip(final_steering, -0.8, 0.8)

        # Smoothing
        self.steering_history.append(final_steering)
        if len(self.steering_history) > 3:
            self.steering_history.pop(0)

        smoothed = np.mean(self.steering_history)
        
        if frame_count % 10 == 0:
            print(f"Steer: {smoothed:.2f} | Path: {path_steering:.2f} | Sem: {semantic_steering:.2f} | Off: {semantic_data.get('road_center_offset', 0.0):.2f}")
            
        return smoothed

    def calculate_throttle_brake(self, lidar_data, radar_data, yolo_data, semantic_data, frame_count):
        """
        Calculate throttle and brake based on LiDAR + Radar + YOLO
        """
        throttle = 0.0
        brake = 0.0
        reason = "UNKNOWN"

        front_dist = lidar_data['front_distance']
        approaching_speed = abs(radar_data['fastest_approach_speed'])

        # === EMERGENCY STOPS - CHECK YOLO FIRST! ===

        # YOLO vehicle VERY close - EMERGENCY BRAKE!
        if yolo_data.get('vehicle_very_close', False):
            return 0.0, 1.0, "YOLO: VEHICLE VERY CLOSE!"

        # Pedestrian VERY CLOSE - EMERGENCY STOP
        if yolo_data.get('person_very_close', False) or semantic_data['pedestrian_danger']:
            return 0.0, 1.0, "PEDESTRIAN CLOSE!"

        # Very close obstacle (LiDAR) - reduced threshold for later braking
        if front_dist < 3:
            return 0.0, 1.0, f"LIDAR OBSTACLE {front_dist:.1f}m!"

        # YOLO vehicle close - strong brake
        if yolo_data['vehicle_close']:
            return 0.1, 0.7, "YOLO: VEHICLE CLOSE!"

        # Pedestrian detected (not very close) - slow down, don't full stop
        if yolo_data['person_close']:
            return 0.2, 0.5, "PEDESTRIAN AHEAD"

        # Fast approaching object (Radar)
        if approaching_speed > 5 and radar_data['closest_moving'] < 15:
            return 0.0, 0.8, f"RADAR INCOMING {approaching_speed:.1f}m/s!"

        # === SPEED CONTROL ===

        obstacle_dist = lidar_data['front_distance']
        radar_count = len(radar_data['approaching_objects'])
        
        target_speed = self.max_speed  # Configurable from dashboard
        
        # SLOW DOWN ON TURNS (Human-like driving)
        if self.steering_history:
            last_steer = abs(self.steering_history[-1])
            if last_steer > 0.4:
                target_speed = min(target_speed, 20.0)
            elif last_steer > 0.2:
                target_speed = min(target_speed, 30.0)
            elif last_steer > 0.1:
                target_speed = min(target_speed, 35.0)

        reason = "CLEAR"

        # Only care about radar objects if they are somewhat close (< 30m)
        radar_danger = radar_count > 0 and radar_data['closest_moving'] < 20.0

        # STOP CONDITIONS (High Priority)
        # 1. Obstacle (Hysteresis: Stop < 8m, Resume > 10m)
        # We need state to handle hysteresis properly, but for now let's just use a gap
        # Actually, let's just use a simple check.
        
        # STOP CONDITIONS (High Priority)
        # Simulation mode: Can brake late since no real inertia
        stop_threshold = 4.0 
        
        # Check if we are clear of obstacle to reset overtaking
        if self.is_overtaking and obstacle_dist > 15.0:
            self.is_overtaking = False
            self.blocked_timer = 0.0

        if obstacle_dist < stop_threshold:
            target_speed = 0.0
            reason = "OBSTACLE"
            if frame_count % 10 == 0:
                print(f"[STOP] Obstacle ahead ({obstacle_dist:.1f}m)")
            
            # BLOCKED DETECTION
            self.blocked_timer += 1.0 / 20.0
            
            if self.blocked_timer > 3.0:
                self.is_overtaking = True
                print("[!] BLOCKED! Attempting to OVERTAKE...")

            # OVERRIDE if overtaking
            if self.is_overtaking:
                target_speed = 15.0
                reason = "OVERTAKING"
        
        elif yolo_data['traffic_light_status'] == 'red':
            target_speed = 0.0
            reason = "TRAFFIC LIGHT RED"
            self.blocked_timer = 0.0 # Reset if stopped for light
            target_speed = 0.0
            reason = "TRAFFIC LIGHT RED"
            
        # SLOW CONDITIONS (Medium Priority)
        # Linear Deceleration for LiDAR Obstacles
        # If obstacle is between 10m and 40m, scale speed linearly
        elif obstacle_dist < 40.0:
            # Map distance [10, 40] to speed [0, max_speed]
            # factor 0.0 at 10m, 1.0 at 40m
            factor = (obstacle_dist - 10.0) / 30.0
            factor = np.clip(factor, 0.1, 1.0) # Don't go below 10% speed unless stopped
            target_speed = self.max_speed * factor
            reason = f"BRAKING (LIDAR {obstacle_dist:.1f}m)"
            
        elif radar_danger:
            target_speed = 30.0
            reason = "CAUTION (RADAR)"
            
        elif yolo_data['traffic_light_status'] == 'yellow':
            target_speed = 20.0 # Slow down for yellow
            reason = "TRAFFIC LIGHT YELLOW"

        # Calculate throttle/brake based on target speed
        current_speed = self.vehicle.get_velocity()
        current_kmh = 3.6 * math.sqrt(current_speed.x**2 + current_speed.y**2)
        
        # PID Control for Throttle
        error = target_speed - current_kmh

        # P = 0.15 (Increased from 0.05), I = 0.001, D = 0.01
        p_term = 0.15 * error
        # Anti-windup
        self.speed_error_sum = np.clip(self.speed_error_sum, -200.0, 200.0)

        i_term = 0.001 * self.speed_error_sum
        d_term = 0.01 * (error - self.last_speed_error)
        self.last_speed_error = error

        # Store for logging
        self._last_p_term = p_term
        self._last_i_term = i_term
        self._last_d_term = d_term
        self._last_target_speed = target_speed

        throttle_pid = p_term + i_term + d_term
        
        if current_kmh < target_speed:
            throttle = np.clip(throttle_pid, 0.0, 1.0) # Allow full throttle
            brake = 0.0
        else:
            throttle = 0.0
            diff = current_kmh - target_speed
            
            # Emergency braking
            if obstacle_dist < 10.0:
                brake = 1.0
            elif diff > 10.0:
                brake = 0.5 # Strong brake for big overshoot
            elif diff > 2.0:
                brake = 0.1 # Light brake for small overshoot
            else:
                brake = 0.0 # Coast (0-2 km/h overshoot)

        # Reduce speed if vehicle detected by semantic - BUT only if LiDAR confirms obstacle
        # FIX: Don't trust semantic alone - it gives false positives from distant/parked cars
        if semantic_data['vehicle_ahead'] and obstacle_dist < 15.0:
            throttle *= 0.6
            brake = max(brake, 0.2)
            reason = "SEMANTIC: VEHICLE"

        # Debug print for speed tuning (moved AFTER semantic check to show actual values)
        if frame_count % 10 == 0:
            print(f"Speed: {current_kmh:.1f}/{target_speed:.1f} | Throt: {throttle:.2f} Brake: {brake:.2f} | Reason: {reason}")

        return throttle, brake, reason

    def check_stuck(self, throttle, brake, reason):
        """Check if we're stuck and need recovery"""
        # Only check if we are TRYING to move (throttle > 0) and NOT braking
        # IMPORTANT: Do NOT trigger stuck logic if we are stopped for a valid reason!
        valid_stops = ["TRAFFIC LIGHT", "OBSTACLE", "STOPPING", "PEDESTRIAN", "YOLO"]
        is_valid_stop = any(s in reason for s in valid_stops)

        if throttle < 0.1 or brake > 0.1 or is_valid_stop:
            self.stuck_counter = 0
            return False

        pos = self.vehicle.get_location()

        if self.last_position:
            dist = pos.distance(self.last_position)
            if dist < 0.1:  # Increased from 0.05 - allow some wobble
                self.stuck_counter += 1
            else:
                self.stuck_counter = max(0, self.stuck_counter - 1)

        self.last_position = pos
        return self.stuck_counter > 60  # Stuck for ~2 seconds

    def get_control(self, target_wp, semantic_data, lidar_data, radar_data, yolo_data, imu_data, frame_count):
        """
        MAIN CONTROL FUNCTION - Sensor fusion decision making!
        """
        
        # === COLLISION RECOVERY ===
        if time.time() - self.last_collision_time < 1.5:
            control = carla.VehicleControl()
            control.reverse = True
            control.throttle = 0.4
            control.steer = random.choice([-0.5, 0.5])
            return control, "COLLISION RECOVERY"

        # Get steering
        steering = self.calculate_steering(self.vehicle.get_transform(), target_wp, semantic_data, lidar_data, imu_data, frame_count)

        # Get throttle/brake
        throttle, brake, speed_reason = self.calculate_throttle_brake(lidar_data, radar_data, yolo_data, semantic_data, frame_count)

        # Store current values
        self.current_steering = steering
        self.current_throttle = throttle
        self.current_brake = brake
        # Create Control Object
        control = carla.VehicleControl()
        
        # === STUCK RECOVERY ===
        current_speed = self.vehicle.get_velocity()
        speed_kmh = 3.6 * math.sqrt(current_speed.x**2 + current_speed.y**2)
        
        # Detect if we are trying to move but not moving
        # REMOVED brake check: If throttle is high and speed is low, we are stuck!
        # Also check reason to ensure we aren't stopped for a valid reason (like a light)
        # FIX: Use speed_reason (current frame) instead of self.reason (previous frame)
        # FIX: Added "NUDGING", "BRAKING", "OVERTAKING" to valid stops
        valid_stops = ["TRAFFIC LIGHT", "OBSTACLE", "STOPPING", "PEDESTRIAN", "YOLO", "NUDGING", "BRAKING", "OVERTAKING", "CAUTION"]
        is_valid_stop = any(s in speed_reason for s in valid_stops)

        # FIX: Also check if there's actually an obstacle close - don't get stuck if path is clear
        front_clear = lidar_data.get('front_distance', 100.0) > 5.0

        if throttle > 0.5 and speed_kmh < 1.0 and not is_valid_stop and not front_clear:
            self.stuck_counter += 1
        else:
            self.stuck_counter = max(0, self.stuck_counter - 2)  # Faster decay

        # Trigger Reverse if stuck for > 2.5 seconds (50 frames) - increased threshold
        if self.stuck_counter > 50:
            self.reverse_mode = True
            self.reverse_start_time = time.time()
            self.stuck_counter = 0
            
            # Determine Stuck Side and set Bias
            left_d = lidar_data.get('left_distance', 100.0)
            right_d = lidar_data.get('right_distance', 100.0)
            
            if left_d < 2.0:
                self.lateral_bias = 0.4 # Steer Right
                print(f"[!] STUCK LEFT ({left_d:.1f}m)! Biasing RIGHT...")
            elif right_d < 2.0:
                self.lateral_bias = -0.4 # Steer Left
                print(f"[!] STUCK RIGHT ({right_d:.1f}m)! Biasing LEFT...")
            else:
                self.lateral_bias = random.choice([-0.3, 0.3]) # Random bias
                print("[!] STUCK CENTER! Random Bias...")
                
            self.bias_timer = 5.0 # Apply bias for 5 seconds
            print("[!] STUCK DETECTED! Reversing...")
            
        # Execute Reverse Maneuver
        if self.reverse_mode:
            # Safety Check: Stop reversing if obstacle behind!
            rear_clear = True
            if 'rear_distance' in lidar_data and lidar_data['rear_distance'] < 1.0:
                 rear_clear = False
                 print("[STOP] REAR OBSTACLE! Stopping Reverse.")

            # FIX: Also abort reverse if front is now clear (> 8m)
            front_now_clear = lidar_data.get('front_distance', 100.0) > 8.0

            if time.time() - self.reverse_start_time < 2.0 and rear_clear and not front_now_clear:
                control.throttle = 0.5  # Reduced from 0.6
                # Steer based on lateral bias (smarter than random)
                if self.lateral_bias != 0:
                    control.steer = -self.lateral_bias  # Opposite direction when reversing
                else:
                    direction = 1.0 if (int(time.time()) % 2 == 0) else -1.0
                    control.steer = direction * 0.6  # Reduced from 0.8
                control.brake = 0.0
                control.reverse = True
                self.reason = "STUCK RECOVERY"
                return control, "STUCK"
            else:
                self.reverse_mode = False
                # Reset PID to avoid jumping forward
                self.speed_error_sum = 0.0
                self.last_speed_error = 0.0
                if front_now_clear:
                    print("Front cleared - exiting reverse early")
        
        control.steer = steering
        control.throttle = throttle
        control.brake = brake
        control.hand_brake = False
        control.manual_gear_shift = False
        
        # NUDGE LOGIC:
        # If stopped for obstacle, but we are steering away (> 0.15), allow creeping
        if "OBSTACLE" in speed_reason and abs(steering) > 0.15:
            control.throttle = 0.35 # Increased from 0.25
            control.brake = 0.0
            speed_reason = "NUDGING"

        self.reason = speed_reason

        # Build mode string
        if semantic_data['road_visible']:
            road_info = f"Road: {semantic_data['road_center_offset']:+.2f}"
        else:
            road_info = "NO ROAD!"

        mode = f"{speed_reason} | {road_info}"

        return control, mode
