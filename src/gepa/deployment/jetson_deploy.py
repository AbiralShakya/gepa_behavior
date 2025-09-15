"""
Jetson Nano Deployment Interface for Ironing Robot System

This module provides the deployment interface for running the trained models
on the Jetson Nano with real-time camera input and robot control.
"""

import torch
import torch.nn as nn
import cv2
import numpy as np
import yaml
import time
import threading
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import queue
import logging
from collections import deque

from ..models.world_model_ironing import IroningWorldModel
from ..models.behavior_model_ironing import IroningBehaviorModel


class RealTimeCamera:
    """
    Real-time camera interface for Jetson Nano.
    """
    
    def __init__(self, 
                 camera_id: int = 0,
                 resolution: Tuple[int, int] = (640, 480),
                 fps: int = 30):
        
        self.camera_id = camera_id
        self.resolution = resolution
        self.fps = fps
        
        # Initialize camera
        self.cap = cv2.VideoCapture(camera_id)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
        self.cap.set(cv2.CAP_PROP_FPS, fps)
        
        # Frame buffer
        self.frame_buffer = deque(maxlen=5)  # Keep last 5 frames
        self.latest_frame = None
        self.frame_lock = threading.Lock()
        
        # Camera thread
        self.camera_thread = None
        self.running = False
        
    def start(self):
        """Start camera capture thread."""
        self.running = True
        self.camera_thread = threading.Thread(target=self._capture_loop)
        self.camera_thread.start()
    
    def stop(self):
        """Stop camera capture thread."""
        self.running = False
        if self.camera_thread:
            self.camera_thread.join()
        self.cap.release()
    
    def _capture_loop(self):
        """Main camera capture loop."""
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                with self.frame_lock:
                    self.latest_frame = frame.copy()
                    self.frame_buffer.append(frame.copy())
    
    def get_latest_frame(self) -> Optional[np.ndarray]:
        """Get the latest captured frame."""
        with self.frame_lock:
            return self.latest_frame.copy() if self.latest_frame is not None else None
    
    def get_frame_sequence(self, length: int = 5) -> Optional[np.ndarray]:
        """Get a sequence of recent frames."""
        with self.frame_lock:
            if len(self.frame_buffer) >= length:
                return np.array(list(self.frame_buffer)[-length:])
            return None


class RobotController:
    """
    Robot controller interface for real hardware.
    """
    
    def __init__(self, config: Dict):
        
        self.config = config
        self.safety_limits = config.get('safety_limits', {})
        
        # Robot state
        self.left_arm_joints = np.zeros(7)
        self.right_arm_joints = np.zeros(7)
        self.actuator_position = 0.1
        
        # Control parameters
        self.max_velocity = 0.5  # rad/s
        self.max_acceleration = 1.0  # rad/s²
        self.control_frequency = config.get('control_loop', {}).get('frequency', 10)
        
        # Action smoothing
        self.action_smoothing = config.get('control_loop', {}).get('action_smoothing', True)
        self.previous_actions = None
        self.smoothing_factor = 0.1
        
        # Safety
        self.emergency_stop = False
        self.safety_limits_enabled = config.get('control_loop', {}).get('safety_limits', True)
        
        # Control thread
        self.control_thread = None
        self.running = False
        self.action_queue = queue.Queue(maxsize=10)
        
    def start(self):
        """Start robot control thread."""
        self.running = True
        self.control_thread = threading.Thread(target=self._control_loop)
        self.control_thread.start()
    
    def stop(self):
        """Stop robot control thread."""
        self.running = False
        if self.control_thread:
            self.control_thread.join()
    
    def _control_loop(self):
        """Main robot control loop."""
        dt = 1.0 / self.control_frequency
        
        while self.running:
            try:
                # Get latest action from queue
                actions = self.action_queue.get(timeout=0.1)
                
                if not self.emergency_stop:
                    # Apply actions
                    self._apply_actions(actions)
                
            except queue.Empty:
                # No new actions, continue with current state
                pass
            
            time.sleep(dt)
    
    def _apply_actions(self, actions: Dict[str, np.ndarray]):
        """Apply actions to robots."""
        
        # Extract actions
        left_arm_actions = actions.get('left_arm', self.left_arm_joints)
        right_arm_actions = actions.get('right_arm', self.right_arm_joints)
        actuator_action = actions.get('actuator', np.array([self.actuator_position]))
        
        # Apply action smoothing
        if self.action_smoothing and self.previous_actions is not None:
            left_arm_actions = self._smooth_actions(
                left_arm_actions, 
                self.previous_actions['left_arm']
            )
            right_arm_actions = self._smooth_actions(
                right_arm_actions, 
                self.previous_actions['right_arm']
            )
            actuator_action = self._smooth_actions(
                actuator_action, 
                self.previous_actions['actuator']
            )
        
        # Apply safety limits
        if self.safety_limits_enabled:
            left_arm_actions = self._apply_safety_limits(left_arm_actions, 'left_arm')
            right_arm_actions = self._apply_safety_limits(right_arm_actions, 'right_arm')
            actuator_action = self._apply_safety_limits(actuator_action, 'actuator')
        
        # Update robot states
        self.left_arm_joints = left_arm_actions
        self.right_arm_joints = right_arm_actions
        self.actuator_position = actuator_action[0]
        
        # Store for next smoothing
        self.previous_actions = {
            'left_arm': left_arm_actions,
            'right_arm': right_arm_actions,
            'actuator': actuator_action
        }
        
        # Send commands to hardware (placeholder)
        self._send_to_hardware(left_arm_actions, right_arm_actions, actuator_action)
    
    def _smooth_actions(self, current_actions: np.ndarray, previous_actions: np.ndarray) -> np.ndarray:
        """Apply exponential smoothing to actions."""
        return (1 - self.smoothing_factor) * previous_actions + self.smoothing_factor * current_actions
    
    def _apply_safety_limits(self, actions: np.ndarray, robot_name: str) -> np.ndarray:
        """Apply safety limits to actions."""
        
        # Joint limits
        if robot_name in ['left_arm', 'right_arm']:
            joint_limits = self.safety_limits.get('joint_limits', [[-3.14, 3.14]] * 7)
            for i, limits in enumerate(joint_limits):
                actions[i] = np.clip(actions[i], limits[0], limits[1])
        
        # Actuator limits
        elif robot_name == 'actuator':
            actuator_limits = self.safety_limits.get('actuator_limits', [0.0, 0.2])
            actions[0] = np.clip(actions[0], actuator_limits[0], actuator_limits[1])
        
        return actions
    
    def _send_to_hardware(self, 
                         left_arm_actions: np.ndarray,
                         right_arm_actions: np.ndarray,
                         actuator_action: np.ndarray):
        """Send commands to actual robot hardware."""
        
        # Placeholder for actual hardware interface
        # In practice, this would interface with the robot's control system
        print(f"Left arm: {left_arm_actions}")
        print(f"Right arm: {right_arm_actions}")
        print(f"Actuator: {actuator_action}")
    
    def send_actions(self, actions: Dict[str, np.ndarray]):
        """Send actions to robot controller."""
        try:
            self.action_queue.put_nowait(actions)
        except queue.Full:
            print("Warning: Action queue full, dropping action")
    
    def emergency_stop_robot(self):
        """Emergency stop the robot."""
        self.emergency_stop = True
        print("EMERGENCY STOP ACTIVATED!")
    
    def resume_robot(self):
        """Resume robot operation."""
        self.emergency_stop = False
        print("Robot operation resumed")


class JetsonDeployment:
    """
    Main deployment interface for Jetson Nano.
    """
    
    def __init__(self, config_path: str):
        
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        
        # Load models
        self.world_model = None
        self.behavior_model = None
        self._load_models()
        
        # Initialize components
        self.camera = RealTimeCamera(
            camera_id=self.config['deployment']['real_time_camera']['camera_id'],
            resolution=self.config['deployment']['real_time_camera']['resolution'],
            fps=self.config['deployment']['real_time_camera']['fps']
        )
        
        self.robot_controller = RobotController(self.config['deployment'])
        
        # State tracking
        self.running = False
        self.step_count = 0
        
        # Setup logging
        self.setup_logging()
    
    def setup_logging(self):
        """Setup logging for deployment."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def _load_models(self):
        """Load trained models."""
        
        checkpoint_dir = Path(self.config['experiment']['checkpoint_dir'])
        
        # Load world model
        world_model_path = checkpoint_dir / "world_model_best.pth"
        if world_model_path.exists():
            print("Loading world model...")
            world_model_config = self.config['world_model']
            self.world_model = IroningWorldModel(world_model_config).to(self.device)
            
            checkpoint = torch.load(world_model_path, map_location=self.device)
            self.world_model.load_state_dict(checkpoint['model_state_dict'])
            self.world_model.eval()
            print("World model loaded successfully")
        else:
            print("Warning: World model checkpoint not found")
        
        # Load behavior model
        behavior_model_path = checkpoint_dir / "behavior_model_best.pth"
        if behavior_model_path.exists():
            print("Loading behavior model...")
            model_config = self.config['model']
            self.behavior_model = IroningBehaviorModel(model_config).to(self.device)
            
            checkpoint = torch.load(behavior_model_path, map_location=self.device)
            self.behavior_model.load_state_dict(checkpoint['model_state_dict'])
            self.behavior_model.eval()
            print("Behavior model loaded successfully")
        else:
            print("Warning: Behavior model checkpoint not found")
    
    def _preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """Preprocess camera image for model input."""
        
        # Resize to model input size
        target_size = self.config['model']['visual_encoder']['spatial_resolution']
        image_resized = cv2.resize(image, target_size)
        
        # Normalize to [0, 1]
        image_normalized = image_resized.astype(np.float32) / 255.0
        
        # Convert to tensor and add batch dimension
        image_tensor = torch.FloatTensor(image_normalized).permute(2, 0, 1).unsqueeze(0)
        
        return image_tensor.to(self.device)
    
    def _get_proprioceptive_state(self) -> torch.Tensor:
        """Get current proprioceptive state."""
        
        # Get current robot states
        left_arm_state = self.robot_controller.left_arm_joints
        right_arm_state = self.robot_controller.right_arm_joints
        actuator_state = np.array([self.robot_controller.actuator_position])
        
        # Add velocities (simplified - would compute from actual velocities)
        left_vel = np.zeros_like(left_arm_state)
        right_vel = np.zeros_like(right_arm_state)
        actuator_vel = np.zeros_like(actuator_state)
        
        # Combine all proprioceptive data
        proprioceptive_state = np.concatenate([
            left_arm_state, left_vel,  # 14 dims
            right_arm_state, right_vel,  # 14 dims
            actuator_state, actuator_vel  # 2 dims
        ])
        
        # Convert to tensor and add batch and sequence dimensions
        proprioceptive_tensor = torch.FloatTensor(proprioceptive_state).unsqueeze(0).unsqueeze(0)
        
        return proprioceptive_tensor.to(self.device)
    
    def _predict_actions(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """Predict actions using the behavior model."""
        
        if self.behavior_model is None:
            # Return zero actions if no model loaded
            return {
                'left_arm': np.zeros(7),
                'right_arm': np.zeros(7),
                'actuator': np.array([0.1])
            }
        
        # Preprocess image
        image_tensor = self._preprocess_image(image)
        
        # Create temporal sequence (repeat current frame)
        temporal_window = self.config['model']['behavior_cloning']['temporal_window']
        image_sequence = image_tensor.repeat(temporal_window, 1, 1, 1).unsqueeze(0)
        
        # Get proprioceptive state
        proprioceptive_state = self._get_proprioceptive_state()
        proprioceptive_sequence = proprioceptive_state.repeat(temporal_window, 1, 1)
        
        # Predict actions
        with torch.no_grad():
            predictions = self.behavior_model(image_sequence, proprioceptive_sequence)
        
        # Extract actions
        actions = {}
        for agent_name, action_tensor in predictions.items():
            if agent_name != 'combined':
                actions[agent_name] = action_tensor.cpu().numpy().squeeze()
        
        return actions
    
    def run(self):
        """Run the deployment system."""
        
        print("Starting Jetson Nano deployment...")
        print("Press 'q' to quit, 'e' for emergency stop, 'r' to resume")
        
        # Start components
        self.camera.start()
        self.robot_controller.start()
        
        self.running = True
        
        try:
            while self.running:
                # Get latest camera frame
                frame = self.camera.get_latest_frame()
                if frame is None:
                    time.sleep(0.01)
                    continue
                
                # Predict actions
                actions = self._predict_actions(frame)
                
                # Send actions to robot controller
                self.robot_controller.send_actions(actions)
                
                # Display frame with overlay
                display_frame = self._create_display_frame(frame, actions)
                cv2.imshow("Ironing Robot Deployment", display_frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('e'):
                    self.robot_controller.emergency_stop_robot()
                elif key == ord('r'):
                    self.robot_controller.resume_robot()
                
                self.step_count += 1
                
        except KeyboardInterrupt:
            print("Interrupted by user")
        
        finally:
            # Cleanup
            self.running = False
            self.camera.stop()
            self.robot_controller.stop()
            cv2.destroyAllWindows()
            print("Deployment stopped")
    
    def _create_display_frame(self, frame: np.ndarray, actions: Dict[str, np.ndarray]) -> np.ndarray:
        """Create display frame with overlay information."""
        
        display_frame = frame.copy()
        
        # Add text overlay
        y_offset = 30
        cv2.putText(display_frame, "Ironing Robot - Live Deployment", (10, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        y_offset += 30
        
        # Display current actions
        cv2.putText(display_frame, "Current Actions:", (10, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y_offset += 20
        
        for agent_name, action_values in actions.items():
            if agent_name == 'left_arm':
                cv2.putText(display_frame, f"Left Arm: {action_values[:3]}...", (10, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            elif agent_name == 'right_arm':
                cv2.putText(display_frame, f"Right Arm: {action_values[:3]}...", (10, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            elif agent_name == 'actuator':
                cv2.putText(display_frame, f"Actuator: {action_values[0]:.3f}", (10, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            y_offset += 15
        
        # Add control instructions
        y_offset += 10
        cv2.putText(display_frame, "Controls: q=quit, e=emergency stop, r=resume", (10, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        # Add emergency stop indicator
        if self.robot_controller.emergency_stop:
            cv2.putText(display_frame, "EMERGENCY STOP", (10, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
        
        return display_frame


def main():
    """Main function for deployment."""
    
    import argparse
    
    parser = argparse.ArgumentParser(description="Deploy Ironing Robot on Jetson Nano")
    parser.add_argument("--config", type=str, default="configs/default.yaml",
                       help="Path to configuration file")
    
    args = parser.parse_args()
    
    # Run deployment
    deployment = JetsonDeployment(args.config)
    deployment.run()


if __name__ == "__main__":
    main()
