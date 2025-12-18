#!/usr/bin/env python3
import carla
import os
import sys

def main():
    try:
        client = carla.Client('localhost', 2000)
        client.set_timeout(10.0)
        
        print("Connecting to CARLA...")
        available_maps = client.get_available_maps()
        
        print("\n" + "="*40)
        print(f"AVAILABLE CARLA MAPS ({len(available_maps)})")
        print("="*40)
        
        # Sort for better readability
        available_maps.sort()
        
        for m in available_maps:
            # Extract just the map name (e.g., "Town01" from "/Game/Carla/Maps/Town01")
            map_name = m.split('/')[-1]
            print(f"  - {map_name} ({m})")
            
        print("\nTo load a map, run:")
        print("  python3 change_map.py <MapName>")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == '__main__':
    main()
