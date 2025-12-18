import carla
import cv2
import numpy as np
import time
import queue
import sys
import os

from .config import *
from .sensors import SensorManager
from .perception import PerceptionManager
from .planning import PathPlanner
from .control import Controller
from .visualization import Visualizer
from .screenshot import ScreenshotCapture

class SensorFusionDriver:
    """
    Autonomous Driving System using sensor fusion.
    Multi-modal perception for real-time navigation.
    """

    def __init__(self):
        # CARLA connection
        self.client = None
        self.world = None
        self.vehicle = None

        # Modules
        self.sensors = None
        self.perception = PerceptionManager()
        self.planner = None
        self.controller = None
        self.visualizer = Visualizer()
        self.screenshot = ScreenshotCapture()

        # Stats
        self.frame_count = 0
        self.start_time = time.time()
        self.fullscreen = False

        # Visualization Data
        self.last_rgb = None
        self.last_semantic = None

        # Town name for screenshots
        self.town_name = "unknown"

    def setup(self):
        """Setup CARLA and all sensors"""
        print("\n" + "="*60)
        print("AUTONOMOUS DRIVING SYSTEM - SENSOR FUSION")
        print("Multi-Modal Perception & Real-Time Control")
        print("="*60)

        print("\nConnecting to CARLA...")
        self.client = carla.Client('localhost', 2000)
        self.client.set_timeout(10.0)
        self.world = self.client.get_world()

        # Get town name for screenshots
        self.town_name = self.world.get_map().name.split('/')[-1]
        print(f"Map: {self.town_name}")

        # Spawn vehicle ON A ROAD
        bp_lib = self.world.get_blueprint_library()
        vehicle_bp = bp_lib.find('vehicle.tesla.model3')
        spawn_points = self.world.get_map().get_spawn_points()

        # Try a few spawn points
        spawn_point = None
        for _ in range(10):
            sp = np.random.choice(spawn_points)
            try:
                self.vehicle = self.world.try_spawn_actor(vehicle_bp, sp)
                if self.vehicle is not None:
                    spawn_point = sp
                    break
            except:
                continue

        if self.vehicle is None:
            spawn_point = spawn_points[0]
            self.vehicle = self.world.spawn_actor(vehicle_bp, spawn_point)

        print(f"Vehicle spawned at {spawn_point.location}")

        # Initialize Modules
        self.sensors = SensorManager(self.world, self.vehicle)
        self.sensors.setup_all()
        
        self.planner = PathPlanner(self.world, self.vehicle)
        self.planner.setup()
        
        self.controller = Controller(self.vehicle)

        print("\nAll systems ready!")

    def run(self):
        """Main control loop"""
        self.setup()

        print("\n" + "="*60)
        print("FULL SENSOR SUITE AUTONOMOUS DRIVER")
        print("="*60)
        print("\nDriving Controls:")
        print("  q - Quit")
        print("  f - Toggle fullscreen")
        print("\nScreenshot Controls (see above for details):")
        print("  1-9, 0, p - Capture figures")
        print("="*60 + "\n")

        cv2.namedWindow("SENSOR FUSION AUTONOMOUS DRIVER", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("SENSOR FUSION AUTONOMOUS DRIVER", 1200, 900)

        try:
            while True:
                self.frame_count += 1

                # === PROCESS SENSORS ===
                
                # Check for stale data (clears if > 0.5s old)
                self.perception.check_stale_data(time.time())
                
                # RGB + YOLO
                yolo_detections = []
                
                # DRAIN QUEUE to get latest frame
                rgb_data = None
                while not self.sensors.rgb_queue.empty():
                    rgb_data = self.sensors.rgb_queue.get()
                
                if rgb_data:
                    array = np.frombuffer(rgb_data.raw_data, dtype=np.uint8)
                    self.last_rgb = array.reshape((IMAGE_HEIGHT, IMAGE_WIDTH, 4))[:, :, :3]

                    if self.frame_count % 3 == 0:
                        yolo_detections = self.perception.process_yolo(self.last_rgb, self.frame_count)
                    else:
                        yolo_detections = self.perception.yolo_data['detections']

                # Semantic
                semantic_img = None
                
                # DRAIN QUEUE
                sem_data = None
                while not self.sensors.semantic_queue.empty():
                    sem_data = self.sensors.semantic_queue.get()
                    # Process Semantic Camera
                if sem_data:
                    self.last_semantic = self.perception.process_semantic(sem_data, self.frame_count)
                
                semantic_img = self.last_semantic

                # LiDAR
                lidar_points = None
                
                # DRAIN QUEUE
                lidar_data = None
                while not self.sensors.lidar_queue.empty():
                    lidar_data = self.sensors.lidar_queue.get()
                    
                if lidar_data:
                    # Pass steering angle to curve the detection zone
                    current_control = self.vehicle.get_control()
                    lidar_points = self.perception.process_lidar(lidar_data, current_control.steer, self.frame_count)

                # Radar
                radar_detections = []
                
                # DRAIN QUEUE
                radar_data = None
                while not self.sensors.radar_queue.empty():
                    radar_data = self.sensors.radar_queue.get()
                    
                if radar_data:
                    # Pass ego velocity for static object filtering
                    velocity = self.vehicle.get_velocity()
                    radar_detections = self.perception.process_radar(radar_data, velocity)
                else:
                    # Use last known detections for visualization (so it matches control logic)
                    radar_detections = self.perception.last_radar_detections

                # === PLANNING ===
                self.planner.check_destination_reached()
                target_wp = self.planner.get_target_waypoint()

                # === CONTROL ===
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

                self.vehicle.apply_control(control)

                # === VISUALIZATION ===
                if self.last_rgb is not None:
                    # 1. Main camera
                    camera_display = self.visualizer.draw_main_window(
                        self.last_rgb, control, mode, self.perception.yolo_data['detections']
                    )
                    # 2. Semantic
                    semantic_display = self.visualizer.draw_semantic_window(
                        semantic_img, self.perception.semantic_data
                    )
                    # 3. LiDAR
                    lidar_display = self.visualizer.draw_lidar_window(
                        lidar_points, self.perception.lidar_data
                    )
                    # 4. Radar
                    radar_display = self.visualizer.draw_radar_window(
                        radar_detections, self.perception.radar_data
                    )
                    # 5. Telemetry
                    telemetry_display = self.visualizer.draw_telemetry_window(
                        self.sensors.gnss_data,
                        self.sensors.imu_data,
                        self.vehicle.get_velocity(),
                        self.sensors.lane_invasion_count
                    )
                    # 6. Global Map
                    map_display = self.visualizer.create_map_display(
                        self.planner.route,
                        self.planner.current_waypoint_index,
                        self.vehicle.get_location(),
                        self.vehicle.get_transform().rotation.yaw
                    )

                    # Combine:
                    # Top Section: [Camera (800x600)] | [Semantic (400x300) / LiDAR (400x300)]
                    # Bottom Section: [Radar (400x300)] | [Telemetry (400x300)] | [Map (400x300)]

                    right_col_top = np.vstack((semantic_display, lidar_display)) # 400x600
                    top_section = np.hstack((camera_display, right_col_top))     # 1200x600

                    bottom_section = np.hstack((radar_display, telemetry_display, map_display)) # 1200x300

                    final_display = np.vstack((top_section, bottom_section)) # 1200x900

                    # === SCREENSHOT SYSTEM: Update sequences ===
                    traffic_light_status = self.perception.yolo_data.get('traffic_light_status', 'none')
                    is_reversing = control.reverse if hasattr(control, 'reverse') else False

                    self.screenshot.update_traffic_light_sequence(final_display, traffic_light_status)
                    self.screenshot.update_stuck_recovery_sequence(final_display, mode, is_reversing)

                    # Log data for PID and steering plots
                    if self.screenshot.pid_logging_enabled:
                        current_speed = self.vehicle.get_velocity()
                        current_kmh = 3.6 * np.sqrt(current_speed.x**2 + current_speed.y**2)
                        self.screenshot.log_pid_data(
                            timestamp=time.time(),
                            target_speed=self.controller.max_speed,
                            current_speed=current_kmh,
                            throttle=control.throttle,
                            brake=control.brake,
                            p_term=getattr(self.controller, '_last_p_term', 0),
                            i_term=getattr(self.controller, '_last_i_term', 0),
                            d_term=getattr(self.controller, '_last_d_term', 0),
                            error=getattr(self.controller, 'last_speed_error', 0)
                        )

                    if self.screenshot.steering_logging_enabled:
                        self.screenshot.log_steering_data(
                            timestamp=time.time(),
                            path_steering=getattr(self.controller, '_last_path_steering', 0),
                            semantic_steering=getattr(self.controller, '_last_semantic_steering', 0),
                            final_steering=control.steer,
                            road_offset=self.perception.semantic_data.get('road_center_offset', 0),
                            front_distance=self.perception.lidar_data.get('front_distance', 100)
                        )

                    # Store panels for screenshot access
                    self._camera_display = camera_display
                    self._lidar_display = lidar_display
                    self._final_display = final_display

                    cv2.imshow("SENSOR FUSION AUTONOMOUS DRIVER", final_display)

                key = cv2.waitKey(16) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('f'):
                    self.fullscreen = not self.fullscreen
                    if self.fullscreen:
                        cv2.setWindowProperty("SENSOR FUSION AUTONOMOUS DRIVER", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                    else:
                        cv2.setWindowProperty("SENSOR FUSION AUTONOMOUS DRIVER", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
                elif key != 255:
                    # Handle screenshot keys
                    self.screenshot.handle_key(
                        key,
                        final_display=getattr(self, '_final_display', None),
                        camera_display=getattr(self, '_camera_display', None),
                        lidar_display=getattr(self, '_lidar_display', None),
                        town_name=self.town_name
                    )

        finally:
            self.cleanup()

    def cleanup(self):
        """Cleanup"""
        print("\nCleaning up...")
        if self.sensors:
            self.sensors.destroy()
        if self.vehicle:
            self.vehicle.destroy()
        cv2.destroyAllWindows()
