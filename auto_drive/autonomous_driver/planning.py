import carla
import random
from agents.navigation.global_route_planner import GlobalRoutePlanner

class PathPlanner:
    """
    Manages global path planning and navigation.
    """
    def __init__(self, world, vehicle):
        self.world = world
        self.vehicle = vehicle
        self.grp = None
        self.route = []
        self.current_waypoint_index = 0
        self.destination = None

    def setup(self):
        """Initialize Global Route Planner"""
        print("🗺️  Initializing Global Route Planner...")
        amap = self.world.get_map()
        sampling_resolution = 2.0
        self.grp = GlobalRoutePlanner(amap, sampling_resolution)
        self.set_random_destination()

    def set_random_destination(self):
        """Pick a random spawn point far away"""
        amap = self.world.get_map()
        spawn_points = amap.get_spawn_points()
        
        if len(spawn_points) > 1:
            my_loc = self.vehicle.get_location()
            best_dist = 0
            best_point = None
            
            # Find furthest point
            for sp in spawn_points:
                dist = my_loc.distance(sp.location)
                if dist > best_dist:
                    best_dist = dist
                    best_point = sp
            
            if best_point:
                self.set_destination(best_point.location)

    def set_destination(self, location):
        """Calculate route to destination"""
        self.destination = location
        start_location = self.vehicle.get_location()
        
        self.route = self.grp.trace_route(start_location, location)
        self.current_waypoint_index = 0
        
        print(f"📍 New destination set! Distance: {start_location.distance(location):.1f}m")
        print(f"🛣️  Route calculated: {len(self.route)} waypoints")

    def get_target_waypoint(self):
        """Get the next target waypoint on the route"""
        if not self.route:
            return None

        my_loc = self.vehicle.get_location()
        
        # Find closest waypoint index (don't look backwards too much)
        closest_dist = float('inf')
        closest_index = self.current_waypoint_index
        
        # Search window
        search_start = max(0, self.current_waypoint_index - 5)
        search_end = min(len(self.route), self.current_waypoint_index + 20)
        
        for i in range(search_start, search_end):
            wp_loc = self.route[i][0].transform.location
            dist = my_loc.distance(wp_loc)
            if dist < closest_dist:
                closest_dist = dist
                closest_index = i
        
        # REROUTE CHECK: If we are too far from the route, recalculate (with cooldown)
        if closest_dist > 15.0:  # Increased threshold
            import time
            now = time.time()
            if not hasattr(self, '_last_reroute') or (now - self._last_reroute) > 5.0:
                print(f"[!] Off route ({closest_dist:.1f}m)! Recalculating...")
                self._last_reroute = now
                self.set_destination(self.destination)
                return self.route[0][0] if self.route else None

        self.current_waypoint_index = closest_index
        
        # Look ahead
        target_index = min(len(self.route) - 1, self.current_waypoint_index + 5)
        return self.route[target_index][0]

    def check_destination_reached(self):
        """Check if we reached the destination"""
        if self.destination and self.vehicle:
            dist = self.vehicle.get_location().distance(self.destination)
            if dist < 10.0:
                print("🏁 Destination reached! Setting new random destination...")
                self.set_random_destination()
