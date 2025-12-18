"""
Configuration constants for the Autonomous Driver
"""

# Semantic segmentation colors (CARLA standard)
SEMANTIC_COLORS = {
    0: ('Unlabeled', (0, 0, 0)),
    1: ('Building', (70, 70, 70)),
    2: ('Fence', (100, 40, 40)),
    3: ('Other', (55, 90, 80)),
    4: ('Pedestrian', (220, 20, 60)),      # RED - DANGER!
    5: ('Pole', (153, 153, 153)),
    6: ('RoadLine', (157, 234, 50)),       # GREEN - Lane marking
    7: ('Road', (128, 64, 128)),           # PURPLE - Driveable
    8: ('SideWalk', (244, 35, 232)),       # PINK - Not driveable
    9: ('Vegetation', (107, 142, 35)),
    10: ('Vehicles', (0, 0, 142)),         # BLUE - Obstacle
    11: ('Wall', (102, 102, 156)),
    12: ('TrafficSign', (220, 220, 0)),    # YELLOW
    13: ('Sky', (70, 130, 180)),
    14: ('Ground', (81, 0, 81)),
    15: ('Bridge', (150, 100, 100)),
    16: ('RailTrack', (230, 150, 140)),
    17: ('GuardRail', (180, 165, 180)),
    18: ('TrafficLight', (250, 170, 30)),  # ORANGE
    19: ('Static', (110, 190, 160)),
    20: ('Dynamic', (170, 120, 50)),
    21: ('Water', (45, 60, 150)),
    22: ('Terrain', (145, 170, 100)),
}

# Sensor Settings
IMAGE_WIDTH = 800
IMAGE_HEIGHT = 600
FOV = 100

SEMANTIC_WIDTH = 400
SEMANTIC_HEIGHT = 300
SEMANTIC_FOV = 100

LIDAR_RANGE = 50.0
LIDAR_CHANNELS = 64 # Upgraded from 32
LIDAR_ROTATION_FREQUENCY = 20

RADAR_RANGE = 60.0
