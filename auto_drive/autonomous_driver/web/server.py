"""
Web Dashboard Server for Autonomous Driver
FastAPI + WebSocket for real-time control
"""

import asyncio
import base64
import json
import threading
import time
from pathlib import Path
from typing import Dict, Any, Optional

import cv2
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, StreamingResponse
import uvicorn

# Get the directory where this file is located
WEB_DIR = Path(__file__).parent
STATIC_DIR = WEB_DIR / "static"


class WebDashboard:
    """
    Web-based dashboard for autonomous driver control.
    Provides real-time video streaming and sensor control.
    """

    def __init__(self, driver=None):
        self.driver = driver
        self.app = FastAPI(title="Autonomous Driver Dashboard")

        # Sensor states (toggles)
        self.sensor_states = {
            'rgb_camera': True,
            'semantic_camera': True,
            'lidar': True,
            'radar': True,
            'yolo': True,
            'gps': True,
            'imu': True,
        }

        # View options
        self.view_options = {
            'show_yolo_boxes': True,
            'show_lidar_overlay': True,
            'show_semantic_overlay': True,
            'show_radar_points': True,
            'show_telemetry': True,
        }

        # Control parameters
        self.control_params = {
            'max_speed': 45.0,
            'steering_sensitivity': 1.0,
            'brake_sensitivity': 1.0,
            'auto_mode': True,  # True = autonomous, False = manual
        }

        # Manual control state (from keyboard WASD)
        self.manual_steering = 0.0
        self.manual_throttle = 0.0
        self.manual_brake = 0.0

        # Frame buffers (updated by driver)
        self.frames = {
            'rgb': None,
            'semantic': None,
            'lidar_bev': None,
            'radar_bev': None,
            'collage': None,
        }

        # Telemetry data
        self.telemetry = {
            'speed': 0.0,
            'steering': 0.0,
            'throttle': 0.0,
            'brake': 0.0,
            'gps_lat': 0.0,
            'gps_lon': 0.0,
            'compass': 0.0,
            'lidar_front': 100.0,
            'status': 'INITIALIZING',
        }

        # Connected WebSocket clients
        self.clients: list[WebSocket] = []

        # Setup routes
        self._setup_routes()

    def _setup_routes(self):
        """Configure all API routes"""

        # Serve static files
        self.app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

        @self.app.get("/", response_class=HTMLResponse)
        async def dashboard():
            """Serve main dashboard page"""
            index_path = STATIC_DIR / "index.html"
            return index_path.read_text()

        @self.app.get("/api/status")
        async def get_status():
            """Get current system status"""
            return {
                'sensors': self.sensor_states,
                'views': self.view_options,
                'controls': self.control_params,
                'telemetry': self.telemetry,
            }

        @self.app.post("/api/sensor/{sensor_name}/toggle")
        async def toggle_sensor(sensor_name: str):
            """Toggle a sensor on/off"""
            if sensor_name in self.sensor_states:
                self.sensor_states[sensor_name] = not self.sensor_states[sensor_name]
                await self._broadcast_state_update()
                return {'sensor': sensor_name, 'enabled': self.sensor_states[sensor_name]}
            return {'error': 'Unknown sensor'}

        @self.app.post("/api/sensor/{sensor_name}/set/{state}")
        async def set_sensor(sensor_name: str, state: bool):
            """Set sensor state explicitly"""
            if sensor_name in self.sensor_states:
                self.sensor_states[sensor_name] = state
                await self._broadcast_state_update()
                return {'sensor': sensor_name, 'enabled': state}
            return {'error': 'Unknown sensor'}

        @self.app.post("/api/view/{view_name}/toggle")
        async def toggle_view(view_name: str):
            """Toggle a view option"""
            if view_name in self.view_options:
                self.view_options[view_name] = not self.view_options[view_name]
                await self._broadcast_state_update()
                return {'view': view_name, 'enabled': self.view_options[view_name]}
            return {'error': 'Unknown view option'}

        @self.app.post("/api/control/{param}")
        async def set_control_param(param: str, value: float):
            """Set a control parameter"""
            if param in self.control_params:
                self.control_params[param] = value
                await self._broadcast_state_update()
                return {'param': param, 'value': value}
            return {'error': 'Unknown parameter'}

        @self.app.post("/api/mode/auto")
        async def set_auto_mode():
            """Enable autonomous mode"""
            self.control_params['auto_mode'] = True
            await self._broadcast_state_update()
            return {'mode': 'auto'}

        @self.app.post("/api/mode/manual")
        async def set_manual_mode():
            """Enable manual mode"""
            self.control_params['auto_mode'] = False
            await self._broadcast_state_update()
            return {'mode': 'manual'}

        @self.app.get("/video/rgb")
        async def video_rgb():
            """MJPEG stream of RGB camera"""
            return StreamingResponse(
                self._generate_mjpeg('rgb'),
                media_type='multipart/x-mixed-replace; boundary=frame'
            )

        @self.app.get("/video/semantic")
        async def video_semantic():
            """MJPEG stream of semantic camera"""
            return StreamingResponse(
                self._generate_mjpeg('semantic'),
                media_type='multipart/x-mixed-replace; boundary=frame'
            )

        @self.app.get("/video/collage")
        async def video_collage():
            """MJPEG stream of full collage view"""
            return StreamingResponse(
                self._generate_mjpeg('collage'),
                media_type='multipart/x-mixed-replace; boundary=frame'
            )

        @self.app.get("/video/map")
        async def video_map():
            """MJPEG stream of map/route view"""
            return StreamingResponse(
                self._generate_mjpeg('map'),
                media_type='multipart/x-mixed-replace; boundary=frame'
            )

        @self.app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            """WebSocket for real-time updates"""
            await websocket.accept()
            self.clients.append(websocket)

            try:
                # Send initial state
                await websocket.send_json({
                    'type': 'init',
                    'sensors': self.sensor_states,
                    'views': self.view_options,
                    'controls': self.control_params,
                })

                while True:
                    # Receive commands from client
                    data = await websocket.receive_json()
                    await self._handle_ws_message(websocket, data)

            except WebSocketDisconnect:
                self.clients.remove(websocket)
            except Exception as e:
                print(f"WebSocket error: {e}")
                if websocket in self.clients:
                    self.clients.remove(websocket)

    async def _handle_ws_message(self, websocket: WebSocket, data: dict):
        """Handle incoming WebSocket messages"""
        msg_type = data.get('type')

        if msg_type == 'toggle_sensor':
            sensor = data.get('sensor')
            if sensor in self.sensor_states:
                self.sensor_states[sensor] = not self.sensor_states[sensor]
                await self._broadcast_state_update()

        elif msg_type == 'toggle_view':
            view = data.get('view')
            if view in self.view_options:
                self.view_options[view] = not self.view_options[view]
                await self._broadcast_state_update()

        elif msg_type == 'set_param':
            param = data.get('param')
            value = data.get('value')
            if param in self.control_params:
                self.control_params[param] = value
                await self._broadcast_state_update()

        elif msg_type == 'manual_control':
            # Manual steering/throttle from keyboard
            if not self.control_params['auto_mode']:
                self.manual_steering = data.get('steering', 0)
                self.manual_throttle = data.get('throttle', 0)
                self.manual_brake = data.get('brake', 0)

    async def _broadcast_state_update(self):
        """Send state update to all connected clients"""
        message = {
            'type': 'state_update',
            'sensors': self.sensor_states,
            'views': self.view_options,
            'controls': self.control_params,
        }
        for client in self.clients:
            try:
                await client.send_json(message)
            except:
                pass

    async def _generate_mjpeg(self, frame_type: str):
        """Generate MJPEG stream from frame buffer"""
        while True:
            frame = self.frames.get(frame_type)
            if frame is not None:
                # Encode frame as JPEG
                _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            await asyncio.sleep(0.033)  # ~30 FPS

    def update_frame(self, frame_type: str, frame: np.ndarray):
        """Update a frame buffer (called by driver)"""
        if frame is not None:
            self.frames[frame_type] = frame.copy()  # Copy to avoid threading issues

    def update_telemetry(self, telemetry: dict, chart_data: dict = None):
        """Update telemetry data (called by driver)"""
        self.telemetry.update(telemetry)

        # Store chart data for broadcasting
        self._chart_data = chart_data or {}

        # Broadcast to WebSocket clients (non-blocking)
        # Check if event loop is ready (avoids race condition on startup)
        if hasattr(self, '_loop') and self._loop is not None:
            try:
                asyncio.run_coroutine_threadsafe(
                    self._broadcast_telemetry(),
                    self._loop
                )
            except Exception as e:
                pass  # Ignore broadcast errors (client disconnect, etc.)

    async def _broadcast_telemetry(self):
        """Send telemetry to all WebSocket clients"""
        message = {
            'type': 'telemetry',
            'data': self.telemetry,
            'chart_data': getattr(self, '_chart_data', {}),
        }
        for client in self.clients:
            try:
                await client.send_json(message)
            except:
                pass

    def update_sensor_data(self, lidar_points: list, radar_detections: list):
        """Update raw sensor data for Canvas rendering (called by driver)"""
        self._lidar_points = lidar_points
        self._radar_detections = radar_detections

        # Broadcast to WebSocket clients
        if hasattr(self, '_loop') and self._loop is not None:
            try:
                asyncio.run_coroutine_threadsafe(
                    self._broadcast_sensor_data(),
                    self._loop
                )
            except:
                pass

    async def _broadcast_sensor_data(self):
        """Send raw sensor data to all WebSocket clients"""
        message = {
            'type': 'sensor_data',
            'lidar_points': getattr(self, '_lidar_points', []),
            'radar_detections': getattr(self, '_radar_detections', []),
        }
        for client in self.clients:
            try:
                await client.send_json(message)
            except:
                pass

    def run(self, host: str = "0.0.0.0", port: int = 8000):
        """Run the web server"""
        print(f"\n{'='*60}")
        print(f"Web Dashboard: http://localhost:{port}")
        print(f"{'='*60}\n")

        # Store event loop reference for cross-thread communication
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)

        config = uvicorn.Config(self.app, host=host, port=port, log_level="warning")
        server = uvicorn.Server(config)
        self._loop.run_until_complete(server.serve())

    def run_in_thread(self, host: str = "0.0.0.0", port: int = 8000):
        """Run web server in background thread"""
        def _run():
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)

            config = uvicorn.Config(self.app, host=host, port=port, log_level="warning")
            server = uvicorn.Server(config)
            self._loop.run_until_complete(server.serve())

        thread = threading.Thread(target=_run, daemon=True)
        thread.start()

        print(f"\n{'='*60}")
        print(f"Web Dashboard: http://localhost:{port}")
        print(f"{'='*60}\n")

        return thread


# For standalone testing
if __name__ == "__main__":
    dashboard = WebDashboard()
    dashboard.run()
