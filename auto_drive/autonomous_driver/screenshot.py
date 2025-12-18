"""
Screenshot Capture System for Autonomous Driving Documentation
Captures high-quality images for paper figures.

Figure Reference:
- Fig 2: 6-Panel HUD (Full display)
- Fig 3: PID Response (Generated plot)
- Fig 4: YOLO Detection (Camera panel with bounding boxes)
- Fig 5: LiDAR View (LiDAR panel)
- Fig 6: Different Maps (Full HUD on different towns)
- Fig 7: Traffic Light Response (Sequence capture)
- Fig 8: Steering Fusion (Comparison visualization)
- Fig 9: Stuck Recovery (Sequence capture)
- Fig 10: Full System (Full display or web dashboard)
"""

import cv2
import numpy as np
import os
import time
from datetime import datetime
from collections import deque

try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend for saving
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("[Screenshot] Warning: matplotlib not available. PID plots disabled.")


class ScreenshotCapture:
    """
    High-quality screenshot capture system for documentation figures.
    """

    def __init__(self, output_dir="screenshots"):
        """Initialize the screenshot capture system."""
        self.output_dir = output_dir
        self.ensure_directories()

        # Capture state
        self.capture_enabled = True
        self.last_capture_time = 0
        self.capture_cooldown = 0.5  # seconds between captures

        # Sequence capture state
        self.sequence_mode = None  # 'traffic_light', 'stuck_recovery', 'continuous'
        self.sequence_frames = []
        self.sequence_start_time = 0
        self.sequence_max_duration = 30  # seconds
        self.sequence_interval = 0.5  # seconds between sequence captures
        self.last_sequence_capture = 0

        # Traffic light sequence state
        self.traffic_light_colors_seen = set()
        self.traffic_light_frames = {}  # {'red': frame, 'yellow': frame, 'green': frame}

        # PID data logging
        self.pid_history = deque(maxlen=500)  # Last 500 frames (~25 seconds at 20 FPS)
        self.pid_logging_enabled = False

        # Steering fusion data logging
        self.steering_history = deque(maxlen=500)
        self.steering_logging_enabled = False

        # Panel references (set by driver)
        self.panels = {}

        print(f"\n[Screenshot] System initialized. Output: {os.path.abspath(self.output_dir)}")
        self.print_help()

    def ensure_directories(self):
        """Create output directories if they don't exist."""
        dirs = [
            self.output_dir,
            os.path.join(self.output_dir, "fig2_hud"),
            os.path.join(self.output_dir, "fig3_pid"),
            os.path.join(self.output_dir, "fig4_yolo"),
            os.path.join(self.output_dir, "fig5_lidar"),
            os.path.join(self.output_dir, "fig6_maps"),
            os.path.join(self.output_dir, "fig7_traffic_light"),
            os.path.join(self.output_dir, "fig8_steering_fusion"),
            os.path.join(self.output_dir, "fig9_stuck_recovery"),
            os.path.join(self.output_dir, "fig10_full_system"),
            os.path.join(self.output_dir, "sequences"),
        ]
        for d in dirs:
            os.makedirs(d, exist_ok=True)

    def print_help(self):
        """Print keyboard shortcuts."""
        print("\n" + "="*60)
        print("SCREENSHOT CAPTURE CONTROLS")
        print("="*60)
        print("  1 - Capture Fig 2: Full 6-Panel HUD")
        print("  2 - Capture Fig 3: Generate PID Response Plot")
        print("  3 - Capture Fig 4: YOLO Detection Panel")
        print("  4 - Capture Fig 5: LiDAR View Panel")
        print("  5 - Capture Fig 6: Map Screenshot (current town)")
        print("  6 - Start/Stop Fig 7: Traffic Light Sequence")
        print("  7 - Capture Fig 8: Steering Fusion Comparison")
        print("  8 - Start/Stop Fig 9: Stuck Recovery Sequence")
        print("  9 - Capture Fig 10: Full System Screenshot")
        print("  0 - Toggle PID/Steering data logging")
        print("  p - Generate all plots from logged data")
        print("="*60 + "\n")

    def get_timestamp(self):
        """Get formatted timestamp for filenames."""
        return datetime.now().strftime("%Y%m%d_%H%M%S")

    def can_capture(self):
        """Check if enough time has passed since last capture."""
        return time.time() - self.last_capture_time > self.capture_cooldown

    def save_image(self, image, filepath, scale=1.0):
        """Save image with optional upscaling for higher quality."""
        if image is None:
            print(f"[Screenshot] Error: No image to save")
            return False

        # Upscale if requested (for higher quality figures)
        if scale != 1.0:
            h, w = image.shape[:2]
            new_size = (int(w * scale), int(h * scale))
            image = cv2.resize(image, new_size, interpolation=cv2.INTER_LANCZOS4)

        # Save with maximum quality
        cv2.imwrite(filepath, image, [cv2.IMWRITE_PNG_COMPRESSION, 0])
        print(f"[Screenshot] Saved: {filepath}")
        self.last_capture_time = time.time()
        return True

    # =========================================================================
    # FIGURE CAPTURE METHODS
    # =========================================================================

    def capture_fig2_hud(self, final_display):
        """Fig 2: Full 6-Panel HUD screenshot."""
        if not self.can_capture():
            return False

        filepath = os.path.join(
            self.output_dir, "fig2_hud",
            f"fig2_hud_{self.get_timestamp()}.png"
        )
        return self.save_image(final_display, filepath, scale=2.0)

    def capture_fig4_yolo(self, camera_display):
        """Fig 4: YOLO Detection panel (camera with bounding boxes)."""
        if not self.can_capture():
            return False

        filepath = os.path.join(
            self.output_dir, "fig4_yolo",
            f"fig4_yolo_{self.get_timestamp()}.png"
        )
        return self.save_image(camera_display, filepath, scale=2.0)

    def capture_fig5_lidar(self, lidar_display):
        """Fig 5: LiDAR View panel."""
        if not self.can_capture():
            return False

        filepath = os.path.join(
            self.output_dir, "fig5_lidar",
            f"fig5_lidar_{self.get_timestamp()}.png"
        )
        return self.save_image(lidar_display, filepath, scale=3.0)

    def capture_fig6_map(self, final_display, town_name="unknown"):
        """Fig 6: Map screenshot for different towns."""
        if not self.can_capture():
            return False

        filepath = os.path.join(
            self.output_dir, "fig6_maps",
            f"fig6_map_{town_name}_{self.get_timestamp()}.png"
        )
        return self.save_image(final_display, filepath, scale=2.0)

    def capture_fig10_full(self, final_display):
        """Fig 10: Full system screenshot."""
        if not self.can_capture():
            return False

        filepath = os.path.join(
            self.output_dir, "fig10_full_system",
            f"fig10_full_{self.get_timestamp()}.png"
        )
        return self.save_image(final_display, filepath, scale=2.0)

    # =========================================================================
    # PID RESPONSE LOGGING AND PLOTTING (Fig 3)
    # =========================================================================

    def log_pid_data(self, timestamp, target_speed, current_speed, throttle, brake,
                     p_term, i_term, d_term, error):
        """Log PID controller data for later plotting."""
        if not self.pid_logging_enabled:
            return

        self.pid_history.append({
            'timestamp': timestamp,
            'target_speed': target_speed,
            'current_speed': current_speed,
            'throttle': throttle,
            'brake': brake,
            'p_term': p_term,
            'i_term': i_term,
            'd_term': d_term,
            'error': error
        })

    def capture_fig3_pid(self):
        """Fig 3: Generate PID response plot from logged data."""
        if not MATPLOTLIB_AVAILABLE:
            print("[Screenshot] Error: matplotlib required for PID plots")
            return False

        if len(self.pid_history) < 10:
            print("[Screenshot] Error: Not enough PID data. Enable logging with '0' key and drive for a while.")
            return False

        # Extract data
        data = list(self.pid_history)
        timestamps = [d['timestamp'] - data[0]['timestamp'] for d in data]
        target_speeds = [d['target_speed'] for d in data]
        current_speeds = [d['current_speed'] for d in data]
        throttles = [d['throttle'] for d in data]
        brakes = [d['brake'] for d in data]
        p_terms = [d['p_term'] for d in data]
        i_terms = [d['i_term'] for d in data]
        d_terms = [d['d_term'] for d in data]
        errors = [d['error'] for d in data]

        # Create figure with subplots
        fig, axes = plt.subplots(3, 1, figsize=(12, 10), dpi=150)
        fig.suptitle('PID Controller Response Analysis', fontsize=14, fontweight='bold')

        # Plot 1: Speed tracking
        ax1 = axes[0]
        ax1.plot(timestamps, target_speeds, 'r--', label='Target Speed', linewidth=2)
        ax1.plot(timestamps, current_speeds, 'b-', label='Current Speed', linewidth=1.5)
        ax1.fill_between(timestamps, current_speeds, target_speeds, alpha=0.3, color='gray')
        ax1.set_ylabel('Speed (km/h)')
        ax1.set_title('Speed Tracking Performance')
        ax1.legend(loc='upper right')
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim([timestamps[0], timestamps[-1]])

        # Plot 2: Control outputs
        ax2 = axes[1]
        ax2.plot(timestamps, throttles, 'g-', label='Throttle', linewidth=1.5)
        ax2.plot(timestamps, brakes, 'r-', label='Brake', linewidth=1.5)
        ax2.set_ylabel('Control Value (0-1)')
        ax2.set_title('Throttle and Brake Commands')
        ax2.legend(loc='upper right')
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim([timestamps[0], timestamps[-1]])
        ax2.set_ylim([-0.1, 1.1])

        # Plot 3: PID terms
        ax3 = axes[2]
        ax3.plot(timestamps, p_terms, 'b-', label='P term', linewidth=1)
        ax3.plot(timestamps, i_terms, 'g-', label='I term', linewidth=1)
        ax3.plot(timestamps, d_terms, 'r-', label='D term', linewidth=1)
        ax3.plot(timestamps, errors, 'k--', label='Error', linewidth=1, alpha=0.5)
        ax3.set_xlabel('Time (seconds)')
        ax3.set_ylabel('PID Terms')
        ax3.set_title('PID Controller Components')
        ax3.legend(loc='upper right')
        ax3.grid(True, alpha=0.3)
        ax3.set_xlim([timestamps[0], timestamps[-1]])

        plt.tight_layout()

        # Save
        filepath = os.path.join(
            self.output_dir, "fig3_pid",
            f"fig3_pid_response_{self.get_timestamp()}.png"
        )
        plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()

        print(f"[Screenshot] Saved PID plot: {filepath}")
        return True

    # =========================================================================
    # STEERING FUSION VISUALIZATION (Fig 8)
    # =========================================================================

    def log_steering_data(self, timestamp, path_steering, semantic_steering,
                          final_steering, road_offset, front_distance):
        """Log steering fusion data for later plotting."""
        if not self.steering_logging_enabled:
            return

        self.steering_history.append({
            'timestamp': timestamp,
            'path_steering': path_steering,
            'semantic_steering': semantic_steering,
            'final_steering': final_steering,
            'road_offset': road_offset,
            'front_distance': front_distance
        })

    def capture_fig8_steering_fusion(self):
        """Fig 8: Generate steering fusion comparison visualization."""
        if not MATPLOTLIB_AVAILABLE:
            print("[Screenshot] Error: matplotlib required for steering plots")
            return False

        if len(self.steering_history) < 10:
            print("[Screenshot] Error: Not enough steering data. Enable logging with '0' key and drive for a while.")
            return False

        # Extract data
        data = list(self.steering_history)
        timestamps = [d['timestamp'] - data[0]['timestamp'] for d in data]
        path_steerings = [d['path_steering'] for d in data]
        semantic_steerings = [d['semantic_steering'] for d in data]
        final_steerings = [d['final_steering'] for d in data]
        road_offsets = [d['road_offset'] for d in data]
        front_distances = [d['front_distance'] for d in data]

        # Create figure
        fig, axes = plt.subplots(3, 1, figsize=(14, 10), dpi=150)
        fig.suptitle('Steering Fusion Analysis: Path Planning vs Semantic Lane Detection',
                     fontsize=14, fontweight='bold')

        # Plot 1: Steering comparison
        ax1 = axes[0]
        ax1.plot(timestamps, path_steerings, 'b-', label='Path Planning', linewidth=1.5, alpha=0.8)
        ax1.plot(timestamps, semantic_steerings, 'g-', label='Semantic Lane', linewidth=1.5, alpha=0.8)
        ax1.plot(timestamps, final_steerings, 'r-', label='Final (Fused)', linewidth=2)
        ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax1.set_ylabel('Steering Angle')
        ax1.set_title('Steering Commands: Individual Sources vs Fused Output')
        ax1.legend(loc='upper right')
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim([timestamps[0], timestamps[-1]])
        ax1.set_ylim([-1, 1])

        # Plot 2: Road offset
        ax2 = axes[1]
        ax2.fill_between(timestamps, road_offsets, 0, alpha=0.5, color='blue')
        ax2.plot(timestamps, road_offsets, 'b-', linewidth=1.5)
        ax2.axhline(y=0, color='green', linestyle='-', linewidth=2, label='Lane Center')
        ax2.set_ylabel('Road Center Offset')
        ax2.set_title('Semantic Lane Detection: Road Center Offset')
        ax2.legend(loc='upper right')
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim([timestamps[0], timestamps[-1]])
        ax2.set_ylim([-1, 1])

        # Plot 3: Front distance and fusion weight
        ax3 = axes[2]
        ax3.plot(timestamps, front_distances, 'orange', linewidth=1.5, label='Front Distance')
        ax3.axhline(y=15, color='red', linestyle='--', alpha=0.7, label='Obstacle Threshold (15m)')
        ax3.set_xlabel('Time (seconds)')
        ax3.set_ylabel('Distance (m)')
        ax3.set_title('LiDAR Front Distance (affects fusion weights)')
        ax3.legend(loc='upper right')
        ax3.grid(True, alpha=0.3)
        ax3.set_xlim([timestamps[0], timestamps[-1]])
        ax3.set_ylim([0, max(50, max(front_distances) * 1.1)])

        plt.tight_layout()

        # Save
        filepath = os.path.join(
            self.output_dir, "fig8_steering_fusion",
            f"fig8_steering_fusion_{self.get_timestamp()}.png"
        )
        plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()

        print(f"[Screenshot] Saved steering fusion plot: {filepath}")
        return True

    # =========================================================================
    # TRAFFIC LIGHT SEQUENCE CAPTURE (Fig 7)
    # =========================================================================

    def start_traffic_light_sequence(self):
        """Start capturing traffic light response sequence."""
        self.sequence_mode = 'traffic_light'
        self.traffic_light_colors_seen = set()
        self.traffic_light_frames = {}
        self.sequence_frames = []
        self.sequence_start_time = time.time()
        print("[Screenshot] Traffic light sequence capture STARTED")
        print("             Drive towards traffic lights. Will capture red/yellow/green states.")

    def stop_traffic_light_sequence(self):
        """Stop and save traffic light sequence."""
        if self.sequence_mode != 'traffic_light':
            return

        self.sequence_mode = None
        timestamp = self.get_timestamp()

        # Save individual color frames
        for color, frame in self.traffic_light_frames.items():
            filepath = os.path.join(
                self.output_dir, "fig7_traffic_light",
                f"fig7_traffic_{color}_{timestamp}.png"
            )
            self.save_image(frame, filepath, scale=2.0)

        # Create composite if we have multiple colors
        if len(self.traffic_light_frames) > 1:
            self._create_traffic_light_composite(timestamp)

        print(f"[Screenshot] Traffic light sequence STOPPED. Captured {len(self.traffic_light_frames)} states.")

    def update_traffic_light_sequence(self, final_display, traffic_light_status):
        """Update traffic light sequence with current frame."""
        if self.sequence_mode != 'traffic_light':
            return

        # Check timeout
        if time.time() - self.sequence_start_time > self.sequence_max_duration:
            print("[Screenshot] Traffic light sequence timeout.")
            self.stop_traffic_light_sequence()
            return

        # Capture if we see a new color
        if traffic_light_status in ['red', 'yellow', 'green']:
            if traffic_light_status not in self.traffic_light_colors_seen:
                self.traffic_light_colors_seen.add(traffic_light_status)
                self.traffic_light_frames[traffic_light_status] = final_display.copy()
                print(f"[Screenshot] Captured traffic light: {traffic_light_status.upper()}")

                # Auto-stop if we have all three
                if len(self.traffic_light_colors_seen) >= 3:
                    print("[Screenshot] All traffic light colors captured!")
                    self.stop_traffic_light_sequence()

    def _create_traffic_light_composite(self, timestamp):
        """Create a composite image showing traffic light sequence."""
        colors = ['red', 'yellow', 'green']
        frames = [self.traffic_light_frames.get(c) for c in colors if c in self.traffic_light_frames]

        if len(frames) < 2:
            return

        # Resize frames to same height
        target_height = 600
        resized = []
        for frame in frames:
            h, w = frame.shape[:2]
            scale = target_height / h
            new_w = int(w * scale)
            resized.append(cv2.resize(frame, (new_w, target_height), interpolation=cv2.INTER_LANCZOS4))

        # Stack horizontally
        composite = np.hstack(resized)

        # Add labels
        x_offset = 0
        for i, color in enumerate([c for c in colors if c in self.traffic_light_frames]):
            label_colors = {'red': (0, 0, 255), 'yellow': (0, 255, 255), 'green': (0, 255, 0)}
            w = resized[i].shape[1]
            cv2.putText(composite, color.upper(), (x_offset + 20, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, label_colors[color], 3)
            x_offset += w

        filepath = os.path.join(
            self.output_dir, "fig7_traffic_light",
            f"fig7_traffic_composite_{timestamp}.png"
        )
        cv2.imwrite(filepath, composite, [cv2.IMWRITE_PNG_COMPRESSION, 0])
        print(f"[Screenshot] Saved traffic light composite: {filepath}")

    # =========================================================================
    # STUCK RECOVERY SEQUENCE CAPTURE (Fig 9)
    # =========================================================================

    def start_stuck_recovery_sequence(self):
        """Start capturing stuck recovery sequence."""
        self.sequence_mode = 'stuck_recovery'
        self.sequence_frames = []
        self.sequence_start_time = time.time()
        self.last_sequence_capture = 0
        print("[Screenshot] Stuck recovery sequence capture STARTED")
        print("             Will capture frames automatically when stuck recovery triggers.")

    def stop_stuck_recovery_sequence(self):
        """Stop and save stuck recovery sequence."""
        if self.sequence_mode != 'stuck_recovery':
            return

        self.sequence_mode = None
        timestamp = self.get_timestamp()

        # Save individual frames
        for i, frame in enumerate(self.sequence_frames):
            filepath = os.path.join(
                self.output_dir, "fig9_stuck_recovery",
                f"fig9_stuck_{timestamp}_frame{i:03d}.png"
            )
            cv2.imwrite(filepath, frame, [cv2.IMWRITE_PNG_COMPRESSION, 0])

        # Create composite
        if len(self.sequence_frames) >= 3:
            self._create_stuck_recovery_composite(timestamp)

        print(f"[Screenshot] Stuck recovery sequence STOPPED. Captured {len(self.sequence_frames)} frames.")

    def update_stuck_recovery_sequence(self, final_display, mode_string, is_reversing=False):
        """Update stuck recovery sequence with current frame."""
        if self.sequence_mode != 'stuck_recovery':
            return

        # Check timeout
        if time.time() - self.sequence_start_time > self.sequence_max_duration:
            print("[Screenshot] Stuck recovery sequence timeout.")
            self.stop_stuck_recovery_sequence()
            return

        # Capture during stuck recovery events
        is_stuck_event = "STUCK" in mode_string or is_reversing

        if is_stuck_event:
            current_time = time.time()
            if current_time - self.last_sequence_capture > 0.3:  # Capture every 0.3s during event
                self.sequence_frames.append(final_display.copy())
                self.last_sequence_capture = current_time
                print(f"[Screenshot] Captured stuck recovery frame {len(self.sequence_frames)}")

                # Auto-stop after capturing enough frames
                if len(self.sequence_frames) >= 20:
                    print("[Screenshot] Maximum stuck recovery frames captured!")
                    self.stop_stuck_recovery_sequence()

    def _create_stuck_recovery_composite(self, timestamp):
        """Create a composite showing stuck recovery sequence."""
        # Select key frames (beginning, middle, end)
        n = len(self.sequence_frames)
        if n < 3:
            return

        indices = [0, n//3, 2*n//3, n-1]
        frames = [self.sequence_frames[i] for i in indices if i < n]

        # Resize and stack
        target_height = 450
        resized = []
        for frame in frames:
            h, w = frame.shape[:2]
            scale = target_height / h
            new_w = int(w * scale)
            resized.append(cv2.resize(frame, (new_w, target_height), interpolation=cv2.INTER_LANCZOS4))

        composite = np.hstack(resized)

        # Add stage labels
        labels = ['1. STUCK DETECTED', '2. REVERSING', '3. STEERING', '4. RECOVERED']
        x_offset = 0
        for i, label in enumerate(labels[:len(resized)]):
            cv2.putText(composite, label, (x_offset + 20, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            x_offset += resized[i].shape[1]

        filepath = os.path.join(
            self.output_dir, "fig9_stuck_recovery",
            f"fig9_stuck_composite_{timestamp}.png"
        )
        cv2.imwrite(filepath, composite, [cv2.IMWRITE_PNG_COMPRESSION, 0])
        print(f"[Screenshot] Saved stuck recovery composite: {filepath}")

    # =========================================================================
    # KEYBOARD HANDLER
    # =========================================================================

    def handle_key(self, key, final_display=None, camera_display=None,
                   lidar_display=None, town_name="unknown"):
        """
        Handle keyboard input for screenshot capture.
        Returns True if key was handled.
        """
        if key == ord('1'):
            # Fig 2: Full HUD
            if final_display is not None:
                self.capture_fig2_hud(final_display)
            return True

        elif key == ord('2'):
            # Fig 3: PID Plot
            self.capture_fig3_pid()
            return True

        elif key == ord('3'):
            # Fig 4: YOLO
            if camera_display is not None:
                self.capture_fig4_yolo(camera_display)
            return True

        elif key == ord('4'):
            # Fig 5: LiDAR
            if lidar_display is not None:
                self.capture_fig5_lidar(lidar_display)
            return True

        elif key == ord('5'):
            # Fig 6: Map
            if final_display is not None:
                self.capture_fig6_map(final_display, town_name)
            return True

        elif key == ord('6'):
            # Fig 7: Traffic light sequence toggle
            if self.sequence_mode == 'traffic_light':
                self.stop_traffic_light_sequence()
            else:
                self.start_traffic_light_sequence()
            return True

        elif key == ord('7'):
            # Fig 8: Steering fusion plot
            self.capture_fig8_steering_fusion()
            return True

        elif key == ord('8'):
            # Fig 9: Stuck recovery sequence toggle
            if self.sequence_mode == 'stuck_recovery':
                self.stop_stuck_recovery_sequence()
            else:
                self.start_stuck_recovery_sequence()
            return True

        elif key == ord('9'):
            # Fig 10: Full system
            if final_display is not None:
                self.capture_fig10_full(final_display)
            return True

        elif key == ord('0'):
            # Toggle data logging
            self.pid_logging_enabled = not self.pid_logging_enabled
            self.steering_logging_enabled = not self.steering_logging_enabled
            status = "ENABLED" if self.pid_logging_enabled else "DISABLED"
            print(f"[Screenshot] PID/Steering data logging: {status}")
            return True

        elif key == ord('p'):
            # Generate all plots
            print("[Screenshot] Generating all plots from logged data...")
            self.capture_fig3_pid()
            self.capture_fig8_steering_fusion()
            return True

        return False

    # =========================================================================
    # HELPER METHOD FOR REAL-TIME STEERING VISUALIZATION
    # =========================================================================

    def create_steering_fusion_overlay(self, base_image, path_steering, semantic_steering,
                                        final_steering, road_offset):
        """
        Create a real-time steering fusion visualization overlay.
        Can be added to the camera panel or shown separately.
        """
        # Create overlay image
        overlay = base_image.copy()
        h, w = overlay.shape[:2]

        # Draw steering comparison bar at bottom
        bar_y = h - 80
        bar_height = 60
        bar_width = w - 40
        bar_x = 20

        # Background
        cv2.rectangle(overlay, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height),
                     (0, 0, 0), -1)
        cv2.rectangle(overlay, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height),
                     (255, 255, 255), 1)

        # Center line
        center_x = bar_x + bar_width // 2
        cv2.line(overlay, (center_x, bar_y), (center_x, bar_y + bar_height),
                (100, 100, 100), 2)

        # Draw steering indicators
        scale = bar_width // 2

        # Path steering (blue)
        path_x = int(center_x + path_steering * scale)
        cv2.circle(overlay, (path_x, bar_y + 15), 8, (255, 100, 0), -1)
        cv2.putText(overlay, "PATH", (path_x - 20, bar_y + 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 100, 0), 1)

        # Semantic steering (green)
        sem_x = int(center_x + semantic_steering * scale)
        cv2.circle(overlay, (sem_x, bar_y + 30), 8, (0, 255, 0), -1)
        cv2.putText(overlay, "LANE", (sem_x - 20, bar_y + 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)

        # Final steering (red)
        final_x = int(center_x + final_steering * scale)
        cv2.circle(overlay, (final_x, bar_y + 45), 10, (0, 0, 255), -1)
        cv2.putText(overlay, "FINAL", (final_x - 25, bar_y + 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)

        # Labels
        cv2.putText(overlay, "LEFT", (bar_x + 5, bar_y + bar_height - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        cv2.putText(overlay, "RIGHT", (bar_x + bar_width - 45, bar_y + bar_height - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

        return overlay
