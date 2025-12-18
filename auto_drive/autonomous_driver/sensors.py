import carla
import cv2
import numpy as np
import queue
import weakref
import math
import time
from .config import *

class SensorManager:
    """
    Manages all sensors for the autonomous vehicle.
    """
    def __init__(self, world, vehicle):
        self.world = world
        self.vehicle = vehicle
        
        # Sensors
        self.camera_rgb = None
        self.camera_semantic = None
        self.lidar = None
        self.radar = None
        self.collision_sensor = None
        self.gnss = None
        self.imu = None
        self.lane_invasion = None

        # Data Queues
        self.rgb_queue = queue.Queue()
        self.semantic_queue = queue.Queue()
        self.lidar_queue = queue.Queue()
        self.radar_queue = queue.Queue()
        
        # Data storage
        self.gnss_data = {'latitude': 0.0, 'longitude': 0.0, 'altitude': 0.0}
        self.imu_data = {
            'accelerometer': {'x': 0.0, 'y': 0.0, 'z': 0.0},
            'gyroscope': {'x': 0.0, 'y': 0.0, 'z': 0.0},
            'compass': 0.0
        }
        self.lane_invasion_count = 0
        self.last_collision_time = 0

    def setup_all(self):
        """Setup all sensors"""
        bp_lib = self.world.get_blueprint_library()
        self.setup_rgb_camera(bp_lib)
        self.setup_semantic_camera(bp_lib)
        self.setup_lidar(bp_lib)
        self.setup_radar(bp_lib)
        self.setup_collision_sensor(bp_lib)
        self.setup_gnss(bp_lib)
        self.setup_imu(bp_lib)
        self.setup_lane_invasion(bp_lib)

    def setup_rgb_camera(self, bp_lib):
        bp = bp_lib.find('sensor.camera.rgb')
        bp.set_attribute('image_size_x', str(IMAGE_WIDTH))
        bp.set_attribute('image_size_y', str(IMAGE_HEIGHT))
        bp.set_attribute('fov', str(FOV))
        
        # Adjust position: Higher and tilted down slightly
        spawn_point = carla.Transform(carla.Location(x=0.8, z=1.7), carla.Rotation(pitch=-5))
        self.camera_rgb = self.world.spawn_actor(bp, spawn_point, attach_to=self.vehicle)
        self.camera_rgb.listen(self.rgb_queue.put)
        print(f"RGB Camera ({IMAGE_WIDTH}x{IMAGE_HEIGHT}, FOV {FOV})")

    def setup_semantic_camera(self, bp_lib):
        bp = bp_lib.find('sensor.camera.semantic_segmentation')
        bp.set_attribute('image_size_x', str(SEMANTIC_WIDTH))
        bp.set_attribute('image_size_y', str(SEMANTIC_HEIGHT))
        bp.set_attribute('fov', str(SEMANTIC_FOV))
        
        # High angle for lane detection
        spawn_point = carla.Transform(carla.Location(x=0.0, z=2.0), carla.Rotation(pitch=-25))
        self.camera_semantic = self.world.spawn_actor(bp, spawn_point, attach_to=self.vehicle)
        self.camera_semantic.listen(self.semantic_queue.put)
        print(f"Semantic Camera ({SEMANTIC_WIDTH}x{SEMANTIC_HEIGHT}, pitch=-25) - LANE FOLLOWING")

    def setup_lidar(self, bp_lib):
        bp = bp_lib.find('sensor.lidar.ray_cast')
        bp.set_attribute('range', str(LIDAR_RANGE))
        bp.set_attribute('channels', str(LIDAR_CHANNELS))
        bp.set_attribute('rotation_frequency', str(LIDAR_ROTATION_FREQUENCY))
        bp.set_attribute('points_per_second', str(1300000)) # Upgraded for 64ch
        bp.set_attribute('upper_fov', str(10.0))
        bp.set_attribute('lower_fov', str(-30.0))
        
        # Move LiDAR to FRONT BUMPER/HOOD to avoid seeing the car roof/hood
        spawn_point = carla.Transform(carla.Location(x=2.0, z=1.2))
        self.lidar = self.world.spawn_actor(bp, spawn_point, attach_to=self.vehicle)
        self.lidar.listen(self.lidar_queue.put)
        print(f"LiDAR ({LIDAR_CHANNELS}ch, {LIDAR_RANGE}m range) - OBSTACLE DETECTION")

    def setup_radar(self, bp_lib):
        bp = bp_lib.find('sensor.other.radar')
        bp.set_attribute('horizontal_fov', str(35))
        bp.set_attribute('vertical_fov', str(20))
        bp.set_attribute('range', str(RADAR_RANGE))
        
        spawn_point = carla.Transform(carla.Location(x=2.0, z=1.0))
        self.radar = self.world.spawn_actor(bp, spawn_point, attach_to=self.vehicle)
        self.radar.listen(self.radar_queue.put)
        print(f"Radar ({RADAR_RANGE}m range) - MOVING OBJECT TRACKING")

    def setup_collision_sensor(self, bp_lib):
        bp = bp_lib.find('sensor.other.collision')
        self.collision_sensor = self.world.spawn_actor(bp, carla.Transform(), attach_to=self.vehicle)
        
        # We need a weak reference to self to avoid circular reference in callback
        weak_self = weakref.ref(self)
        self.collision_sensor.listen(lambda event: SensorManager._on_collision(weak_self, event))
        print("Collision sensor")

    def setup_gnss(self, bp_lib):
        bp = bp_lib.find('sensor.other.gnss')
        self.gnss = self.world.spawn_actor(bp, carla.Transform(carla.Location(x=0, z=0)), attach_to=self.vehicle)
        
        weak_self = weakref.ref(self)
        self.gnss.listen(lambda event: SensorManager._on_gnss(weak_self, event))
        print("GNSS (GPS) sensor")

    def setup_imu(self, bp_lib):
        bp = bp_lib.find('sensor.other.imu')
        self.imu = self.world.spawn_actor(bp, carla.Transform(), attach_to=self.vehicle)
        
        weak_self = weakref.ref(self)
        self.imu.listen(lambda event: SensorManager._on_imu(weak_self, event))
        print("IMU (accelerometer + gyroscope)")

    def setup_lane_invasion(self, bp_lib):
        bp = bp_lib.find('sensor.other.lane_invasion')
        self.lane_invasion = self.world.spawn_actor(bp, carla.Transform(), attach_to=self.vehicle)
        
        weak_self = weakref.ref(self)
        self.lane_invasion.listen(lambda event: SensorManager._on_lane_invasion(weak_self, event))
        print("Lane Invasion detector")

    @staticmethod
    def _on_collision(weak_self, event):
        self = weak_self()
        if not self: return
        print(f"[COLLISION] {event.other_actor.type_id}")
        # FIX: Use system time (time.time()) not CARLA timestamp - must match controller
        self.last_collision_time = time.time()

    @staticmethod
    def _on_gnss(weak_self, event):
        self = weak_self()
        if not self: return
        self.gnss_data['latitude'] = event.latitude
        self.gnss_data['longitude'] = event.longitude
        self.gnss_data['altitude'] = event.altitude

    @staticmethod
    def _on_imu(weak_self, sensor_data):
        self = weak_self()
        if not self: return
        self.imu_data['accelerometer'] = {
            'x': sensor_data.accelerometer.x,
            'y': sensor_data.accelerometer.y,
            'z': sensor_data.accelerometer.z
        }
        self.imu_data['gyroscope'] = {
            'x': sensor_data.gyroscope.x,
            'y': sensor_data.gyroscope.y,
            'z': sensor_data.gyroscope.z
        }
        self.imu_data['compass'] = math.degrees(sensor_data.compass)

    @staticmethod
    def _on_lane_invasion(weak_self, event):
        self = weak_self()
        if not self: return
        self.lane_invasion_count += 1
        # print(f"Lane Invasion! Total: {self.lane_invasion_count}")

    def destroy(self):
        """Cleanup sensors"""
        sensors = [self.camera_rgb, self.camera_semantic, self.lidar,
                   self.radar, self.collision_sensor, self.gnss, self.imu,
                   self.lane_invasion]
        for s in sensors:
            if s:
                s.destroy()
