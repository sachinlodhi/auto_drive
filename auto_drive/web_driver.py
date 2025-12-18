#!/usr/bin/env python3
"""
Web-Enabled Autonomous Driver
Access the dashboard at http://localhost:8000
"""

import os
import sys
import time
import threading
import queue

# Add CARLA agents to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
carla_agents_path = os.path.join(project_root, 'CARLA_Latest', 'PythonAPI', 'carla')
if os.path.exists(carla_agents_path):
    sys.path.append(carla_agents_path)

import carla
import cv2
import numpy as np

from autonomous_driver.config import *
from autonomous_driver.sensors import SensorManager
from autonomous_driver.perception import PerceptionManager
from autonomous_driver.planning import PathPlanner
from autonomous_driver.control import Controller
from autonomous_driver.visualization import Visualizer
from autonomous_driver.web.server import WebDashboard


class WebEnabledDriver:
    """
    Autonomous Driver with Web Dashboard
    """

    def __init__(self, web_port=8000):
        # CARLA
        self.client = None
        self.world = None
        self.vehicle = None

        # Modules
        self.sensors = None
        self.perception = PerceptionManager()
        self.planner = None
        self.controller = None
        self.visualizer = Visualizer()

        # Web Dashboard
        self.dashboard = WebDashboard()
        self.web_port = web_port

        # Start with auto mode DISABLED - user must enable from dashboard
        self.dashboard.control_params['auto_mode'] = False

        # State
        self.frame_count = 0
        self.running = True

        # Visualization
        self.last_rgb = None
        self.last_semantic = None

    def setup(self):
        """Setup CARLA and sensors"""
        print("\n" + "=" * 60)
        print("WEB-ENABLED AUTONOMOUS DRIVER")
        print("=" * 60)

        print("\nConnecting to CARLA...")
        self.client = carla.Client('localhost', 2000)
        self.client.set_timeout(10.0)
        self.world = self.client.get_world()

        # Spawn vehicle
        bp_lib = self.world.get_blueprint_library()
        vehicle_bp = bp_lib.find('vehicle.tesla.model3')
        spawn_points = self.world.get_map().get_spawn_points()

        for _ in range(10):
            sp = np.random.choice(spawn_points)
            try:
                self.vehicle = self.world.try_spawn_actor(vehicle_bp, sp)
                if self.vehicle:
                    break
            except:
                continue

        if not self.vehicle:
            self.vehicle = self.world.spawn_actor(vehicle_bp, spawn_points[0])

        print(f"Vehicle spawned!")

        # Initialize modules
        self.sensors = SensorManager(self.world, self.vehicle)
        self.sensors.setup_all()

        self.planner = PathPlanner(self.world, self.vehicle)
        self.planner.setup()

        self.controller = Controller(self.vehicle)

        print("All systems ready!")

    def run(self):
        """Main loop with web dashboard"""
        self.setup()

        # Start web server in background
        self.dashboard.run_in_thread(port=self.web_port)
        time.sleep(1)  # Give server time to start

        print("\nControls:")
        print("  q - Quit")
        print("  Open browser: http://localhost:{}".format(self.web_port))
        print("=" * 60 + "\n")

        # Optional: OpenCV window (uncomment to enable)
        # show_cv_window = True
        # if show_cv_window:
        #     cv2.namedWindow("Driver View", cv2.WINDOW_NORMAL)
        #     cv2.resizeWindow("Driver View", 800, 600)

        try:
            while self.running:
                self.frame_count += 1

                # Check stale data
                self.perception.check_stale_data(time.time())

                # === SENSORS ===

                # Check if sensors are enabled via dashboard
                sensor_states = self.dashboard.sensor_states

                # RGB + YOLO
                yolo_detections = []
                rgb_data = None
                while not self.sensors.rgb_queue.empty():
                    rgb_data = self.sensors.rgb_queue.get()

                if rgb_data and sensor_states.get('rgb_camera', True):
                    array = np.frombuffer(rgb_data.raw_data, dtype=np.uint8)
                    self.last_rgb = array.reshape((IMAGE_HEIGHT, IMAGE_WIDTH, 4))[:, :, :3]

                    if self.frame_count % 3 == 0 and sensor_states.get('yolo', True):
                        yolo_detections = self.perception.process_yolo(self.last_rgb, self.frame_count)
                    else:
                        yolo_detections = self.perception.yolo_data['detections']

                # Semantic
                sem_data = None
                while not self.sensors.semantic_queue.empty():
                    sem_data = self.sensors.semantic_queue.get()

                if sem_data and sensor_states.get('semantic_camera', True):
                    self.last_semantic = self.perception.process_semantic(sem_data, self.frame_count)

                # LiDAR
                lidar_data = None
                while not self.sensors.lidar_queue.empty():
                    lidar_data = self.sensors.lidar_queue.get()

                lidar_points = None
                if lidar_data and sensor_states.get('lidar', True):
                    current_control = self.vehicle.get_control()
                    lidar_points = self.perception.process_lidar(lidar_data, current_control.steer, self.frame_count)

                # Radar
                radar_data = None
                while not self.sensors.radar_queue.empty():
                    radar_data = self.sensors.radar_queue.get()

                radar_detections = []
                if radar_data and sensor_states.get('radar', True):
                    velocity = self.vehicle.get_velocity()
                    radar_detections = self.perception.process_radar(radar_data, velocity)
                else:
                    radar_detections = self.perception.last_radar_detections

                # === PLANNING ===
                self.planner.check_destination_reached()
                target_wp = self.planner.get_target_waypoint()

                # === CONTROL ===
                control_params = self.dashboard.control_params
                auto_mode = control_params.get('auto_mode', False)

                # Check if manual mode (car waits for user input)
                if not auto_mode:
                    # Manual control from web dashboard (WASD keys)
                    manual_steer = self.dashboard.manual_steering
                    manual_throttle = self.dashboard.manual_throttle
                    manual_brake = self.dashboard.manual_brake

                    control = carla.VehicleControl()
                    control.steer = float(manual_steer)

                    # Handle throttle/reverse
                    if manual_throttle > 0:
                        control.throttle = float(manual_throttle)
                        control.brake = 0.0
                        mode = "MANUAL (Driving)"
                    elif manual_throttle < 0:
                        control.reverse = True
                        control.throttle = abs(float(manual_throttle))
                        control.brake = 0.0
                        mode = "MANUAL (Reverse)"
                    elif manual_brake > 0:
                        control.throttle = 0.0
                        control.brake = float(manual_brake)
                        mode = "MANUAL (Braking)"
                    else:
                        # No input - apply parking brake
                        control.throttle = 0.0
                        control.brake = 0.5
                        mode = "MANUAL (Waiting)"
                else:
                    # Autonomous control
                    # Apply max speed from dashboard
                    self.controller.max_speed = control_params.get('max_speed', 45.0)

                    # Sync collision time from sensors to controller
                    self.controller.last_collision_time = self.sensors.last_collision_time

                    control, mode = self.controller.get_control(
                        target_wp,
                        self.perception.semantic_data,
                        self.perception.lidar_data,
                        self.perception.radar_data,
                        self.perception.yolo_data,
                        self.sensors.imu_data,
                        self.frame_count
                    )

                    # Apply steering sensitivity
                    steer_sens = control_params.get('steering_sensitivity', 1.0)
                    control.steer *= steer_sens

                self.vehicle.apply_control(control)

                # === VISUALIZATION ===
                # Create placeholder if no RGB yet
                if self.last_rgb is None:
                    self.last_rgb = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH, 3), dtype=np.uint8)
                    cv2.putText(self.last_rgb, "Waiting for camera...", (200, 300),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

                if self.last_rgb is not None:
                    view_opts = self.dashboard.view_options
                    sensor_states = self.dashboard.sensor_states

                    # Helper to create disabled panel
                    def disabled_panel(size, name):
                        panel = np.zeros((size[1], size[0], 3), dtype=np.uint8)
                        panel[:] = (30, 30, 30)
                        cv2.putText(panel, name, (size[0]//2 - 50, size[1]//2 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 100), 2)
                        cv2.putText(panel, "DISABLED", (size[0]//2 - 45, size[1]//2 + 15),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)
                        return panel

                    # Main camera (with YOLO)
                    if sensor_states.get('rgb_camera', True):
                        show_yolo = sensor_states.get('yolo', True) and view_opts.get('show_yolo_boxes', True)
                        # With overlay for collage
                        camera_display = self.visualizer.draw_main_window(
                            self.last_rgb, control, mode,
                            yolo_detections if show_yolo else []
                        )
                        # Clean version for web (YOLO boxes only, no overlay)
                        camera_display_web = self.visualizer.draw_main_window(
                            self.last_rgb, control, mode,
                            yolo_detections if show_yolo else [],
                            show_overlay=False
                        )
                    else:
                        camera_display = disabled_panel((800, 600), "RGB CAMERA")
                        camera_display_web = camera_display

                    # Semantic
                    if sensor_states.get('semantic_camera', True):
                        semantic_display = self.visualizer.draw_semantic_window(
                            self.last_semantic, self.perception.semantic_data
                        )
                    else:
                        semantic_display = disabled_panel((400, 300), "SEMANTIC")

                    # LiDAR
                    if sensor_states.get('lidar', True):
                        lidar_display = self.visualizer.draw_lidar_window(
                            lidar_points, self.perception.lidar_data
                        )
                    else:
                        lidar_display = disabled_panel((400, 300), "LiDAR")

                    # Radar
                    if sensor_states.get('radar', True):
                        radar_display = self.visualizer.draw_radar_window(
                            radar_detections, self.perception.radar_data
                        )
                    else:
                        radar_display = disabled_panel((400, 300), "RADAR")

                    # Telemetry
                    telemetry_display = self.visualizer.draw_telemetry_window(
                        self.sensors.gnss_data,
                        self.sensors.imu_data,
                        self.vehicle.get_velocity(),
                        self.sensors.lane_invasion_count
                    )

                    # Map
                    map_display = self.visualizer.create_map_display(
                        self.planner.route,
                        self.planner.current_waypoint_index,
                        self.vehicle.get_location(),
                        self.vehicle.get_transform().rotation.yaw
                    )

                    # Build collage
                    right_col_top = np.vstack((semantic_display, lidar_display))
                    top_section = np.hstack((camera_display, right_col_top))
                    bottom_section = np.hstack((radar_display, telemetry_display, map_display))
                    collage = np.vstack((top_section, bottom_section))

                    # Update dashboard frames
                    # Send clean version with YOLO boxes but no overlay
                    self.dashboard.update_frame('rgb', camera_display_web)
                    self.dashboard.update_frame('semantic', semantic_display)
                    self.dashboard.update_frame('map', map_display)
                    self.dashboard.update_frame('collage', collage)

                    # Update telemetry (throttled to ~20 Hz to avoid WebSocket spam)
                    if self.frame_count % 3 == 0:
                        velocity = self.vehicle.get_velocity()
                        speed = 3.6 * np.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)

                        # Telemetry data
                        telemetry_data = {
                            'speed': float(round(speed, 1)),
                            'steering': float(round(control.steer, 2)),
                            'throttle': float(round(control.throttle, 2)),
                            'brake': float(round(control.brake, 2)),
                            'gps_lat': float(self.sensors.gnss_data.get('lat', 0)) if self.sensors.gnss_data else 0.0,
                            'gps_lon': float(self.sensors.gnss_data.get('lon', 0)) if self.sensors.gnss_data else 0.0,
                            'compass': float(round(self.sensors.imu_data.get('compass', 0), 1)) if self.sensors.imu_data else 0.0,
                            'lidar_front': float(round(self.perception.lidar_data['front_distance'], 1)),
                            'status': str(mode),
                        }

                        # Chart data for PID and steering fusion graphs
                        chart_data = {
                            'target_speed': float(self.controller.max_speed),
                            'path_steering': float(round(getattr(self.controller, '_last_path_steering', 0), 3)),
                            'semantic_steering': float(round(getattr(self.controller, '_last_semantic_steering', 0), 3)),
                        }

                        self.dashboard.update_telemetry(telemetry_data, chart_data)

                    # Send raw sensor data for Canvas rendering (every 2 frames for responsiveness)
                    if self.frame_count % 2 == 0:
                        # Prepare LiDAR points (subsample for performance)
                        lidar_points_list = []
                        if lidar_points is not None and len(lidar_points) > 0:
                            # Subsample to max 500 points
                            step = max(1, len(lidar_points) // 500)
                            for i in range(0, len(lidar_points), step):
                                p = lidar_points[i]
                                lidar_points_list.append({
                                    'x': float(round(p[0], 2)),
                                    'y': float(round(p[1], 2)),
                                    'z': float(round(p[2], 2))
                                })

                        # Prepare Radar detections (they are already dicts)
                        radar_list = []
                        if radar_detections:
                            for det in radar_detections[:20]:  # Max 20 detections
                                radar_list.append({
                                    'distance': float(round(det.get('distance', 0), 1)),
                                    'azimuth': float(round(det.get('azimuth', 0), 1)),
                                    'velocity': float(round(det.get('velocity', 0), 1))
                                })

                        self.dashboard.update_sensor_data(lidar_points_list, radar_list)

                # Optional: Show OpenCV window (uncomment to enable)
                # if show_cv_window:
                #     cv2.imshow("Driver View", collage)
                #     key = cv2.waitKey(16) & 0xFF
                #     if key == ord('q'):
                #         break

                # Small delay to prevent CPU spin
                time.sleep(0.016)  # ~60 FPS

        finally:
            self.cleanup()

    def cleanup(self):
        """Cleanup resources"""
        print("\nCleaning up...")
        self.running = False

        if self.sensors:
            self.sensors.destroy()

        if self.vehicle:
            self.vehicle.destroy()

        # cv2.destroyAllWindows()  # Uncomment if using OpenCV window
        print("Done!")


if __name__ == "__main__":
    os.environ["QT_QPA_PLATFORM"] = "xcb"

    driver = WebEnabledDriver(web_port=8080)  # Changed from 8000 to avoid conflicts
    driver.run()
