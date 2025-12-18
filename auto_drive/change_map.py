#!/usr/bin/env python3
import carla
import sys
import time

def main():
    if len(sys.argv) != 2:
        print("Usage: python3 change_map.py <MapName>")
        print("Example: python3 change_map.py Town04")
        return

    map_name = sys.argv[1]
    
    try:
        client = carla.Client('localhost', 2000)
        client.set_timeout(10.0)
        
        print(f"Connecting to CARLA to load map: {map_name}...")
        
        # Check if map exists (simple check)
        available_maps = client.get_available_maps()
        full_map_path = None
        
        # Try to find exact match or partial match
        for m in available_maps:
            if map_name in m:
                full_map_path = m
                break
        
        if not full_map_path:
            print(f"❌ Map '{map_name}' not found!")
            print("Run 'python3 list_maps.py' to see available maps.")
            return

        print(f"Loading {full_map_path}...")
        client.load_world(full_map_path)
        print("✅ Map loaded successfully!")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == '__main__':
    main()
