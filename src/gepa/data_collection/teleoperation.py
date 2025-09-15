"""
Teleoperation Interface for Expert Data Collection

This module provides a teleoperation interface for collecting expert demonstrations
for the ironing robot system. It allows human operators to control the robots
and record their actions for behavior cloning training.
"""

import numpy as np
import cv2
import json
import h5py
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import time
import threading
from collections import deque
import yaml


class TeleoperationController:
    """
    Controller for teleoperating the ironing robot system.
    Supports keyboard and mouse input for controlling robots and actuator.
    """
    
    def __init__(self, 
                 config: Dict,
                 data_save_path: str = "./data/expert_demonstrations"):
        
        self.config = config
        self.data_save_path = Path(data_save_path)
        self.data_save_path.mkdir(parents=True, exist_ok=True)
        
        # Control state
        self.is_recording = False
        self.current_episode = 0
        self.episode_data = []
        
        # Robot control state
        self.left_arm_joints = np.zeros(7)
        self.right_arm_joints = np.zeros(7)
        self.actuator_position = 0.1  # Start above fabric
        
        # Control parameters
        self.joint_step_size = 0.05  # rad
        self.actuator_step_size = 0.005  # m
        self.max_joint_angle = 3.14
        self.min_joint_angle = -3.14
        self.max_actuator_position = 0.2
        self.min_actuator_position = 0.0
        
        # Data collection
        self.recording_frequency = 10  # Hz
        self.last_record_time = 0
        
        # Keyboard mapping
        self.key_mappings = self._setup_key_mappings()
        
        # Create OpenCV window for control interface
        self.control_window = "Ironing Robot Teleoperation"
        cv2.namedWindow(self.control_window, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.control_window, 800, 600)
        
    def _setup_key_mappings(self) -> Dict[str, str]:
        """Setup keyboard mappings for robot control."""
        
        return {
            # Left arm control (WASD + QE for rotation)
            'w': 'left_arm_joint_0_+',
            's': 'left_arm_joint_0_-',
            'a': 'left_arm_joint_1_+',
            'd': 'left_arm_joint_1_-',
            'q': 'left_arm_joint_2_+',
            'e': 'left_arm_joint_2_-',
            'r': 'left_arm_joint_3_+',
            'f': 'left_arm_joint_3_-',
            't': 'left_arm_joint_4_+',
            'g': 'left_arm_joint_4_-',
            'y': 'left_arm_joint_5_+',
            'h': 'left_arm_joint_5_-',
            'u': 'left_arm_joint_6_+',
            'j': 'left_arm_joint_6_-',
            
            # Right arm control (Arrow keys + IJKL for rotation)
            'UP': 'right_arm_joint_0_+',
            'DOWN': 'right_arm_joint_0_-',
            'LEFT': 'right_arm_joint_1_+',
            'RIGHT': 'right_arm_joint_1_-',
            'i': 'right_arm_joint_2_+',
            'k': 'right_arm_joint_2_-',
            'o': 'right_arm_joint_3_+',
            'l': 'right_arm_joint_3_-',
            'p': 'right_arm_joint_4_+',
            ';': 'right_arm_joint_4_-',
            '[': 'right_arm_joint_5_+',
            "'": 'right_arm_joint_5_-',
            ']': 'right_arm_joint_6_+',
            '\\': 'right_arm_joint_6_-',
            
            # Actuator control
            'SPACE': 'actuator_up',
            'SHIFT': 'actuator_down',
            
            # Recording control
            'ENTER': 'start_recording',
            'ESC': 'stop_recording',
            'n': 'new_episode',
            's': 'save_episode',
            
            # Reset
            'r': 'reset_robots',
        }
    
    def process_key_input(self, key: int) -> bool:
        """
        Process keyboard input and return True if should continue, False if should exit.
        
        Args:
            key: OpenCV key code
            
        Returns:
            continue_flag: True to continue, False to exit
        """
        
        # Convert key code to string
        key_str = self._key_to_string(key)
        
        if key_str in self.key_mappings:
            action = self.key_mappings[key_str]
            self._execute_action(action)
        
        # Check for exit
        if key == 27:  # ESC key
            return False
        
        return True
    
    def _key_to_string(self, key: int) -> str:
        """Convert OpenCV key code to string."""
        
        # Special keys
        if key == 13:  # Enter
            return 'ENTER'
        elif key == 27:  # ESC
            return 'ESC'
        elif key == 32:  # Space
            return 'SPACE'
        elif key == 9:  # Tab
            return 'TAB'
        elif key == 8:  # Backspace
            return 'BACKSPACE'
        elif key == 82:  # Up arrow
            return 'UP'
        elif key == 84:  # Down arrow
            return 'DOWN'
        elif key == 81:  # Left arrow
            return 'LEFT'
        elif key == 83:  # Right arrow
            return 'RIGHT'
        else:
            return chr(key).lower()
    
    def _execute_action(self, action: str):
        """Execute the specified action."""
        
        if action == 'start_recording':
            self.start_recording()
        elif action == 'stop_recording':
            self.stop_recording()
        elif action == 'new_episode':
            self.new_episode()
        elif action == 'save_episode':
            self.save_episode()
        elif action == 'reset_robots':
            self.reset_robots()
        elif action.startswith('left_arm_joint_'):
            self._control_left_arm_joint(action)
        elif action.startswith('right_arm_joint_'):
            self._control_right_arm_joint(action)
        elif action.startswith('actuator_'):
            self._control_actuator(action)
    
    def _control_left_arm_joint(self, action: str):
        """Control left arm joint based on action string."""
        
        # Parse action: left_arm_joint_X_+/-
        parts = action.split('_')
        joint_idx = int(parts[3])
        direction = parts[4]
        
        if direction == '+':
            self.left_arm_joints[joint_idx] += self.joint_step_size
        else:
            self.left_arm_joints[joint_idx] -= self.joint_step_size
        
        # Clamp to limits
        self.left_arm_joints[joint_idx] = np.clip(
            self.left_arm_joints[joint_idx],
            self.min_joint_angle,
            self.max_joint_angle
        )
    
    def _control_right_arm_joint(self, action: str):
        """Control right arm joint based on action string."""
        
        # Parse action: right_arm_joint_X_+/-
        parts = action.split('_')
        joint_idx = int(parts[3])
        direction = parts[4]
        
        if direction == '+':
            self.right_arm_joints[joint_idx] += self.joint_step_size
        else:
            self.right_arm_joints[joint_idx] -= self.joint_step_size
        
        # Clamp to limits
        self.right_arm_joints[joint_idx] = np.clip(
            self.right_arm_joints[joint_idx],
            self.min_joint_angle,
            self.max_joint_angle
        )
    
    def _control_actuator(self, action: str):
        """Control linear actuator based on action string."""
        
        if action == 'actuator_up':
            self.actuator_position += self.actuator_step_size
        elif action == 'actuator_down':
            self.actuator_position -= self.actuator_step_size
        
        # Clamp to limits
        self.actuator_position = np.clip(
            self.actuator_position,
            self.min_actuator_position,
            self.max_actuator_position
        )
    
    def start_recording(self):
        """Start recording expert demonstrations."""
        if not self.is_recording:
            self.is_recording = True
            self.episode_data = []
            print("Started recording expert demonstration...")
    
    def stop_recording(self):
        """Stop recording expert demonstrations."""
        if self.is_recording:
            self.is_recording = False
            print("Stopped recording expert demonstration.")
    
    def new_episode(self):
        """Start a new episode."""
        if self.is_recording:
            self.stop_recording()
        
        self.current_episode += 1
        self.episode_data = []
        self.reset_robots()
        print(f"Started new episode {self.current_episode}")
    
    def save_episode(self):
        """Save current episode data."""
        if len(self.episode_data) > 0:
            episode_file = self.data_save_path / f"episode_{self.current_episode:04d}.h5"
            
            with h5py.File(episode_file, 'w') as f:
                # Save episode metadata
                f.attrs['episode_id'] = self.current_episode
                f.attrs['timestamp'] = time.time()
                f.attrs['num_steps'] = len(self.episode_data)
                
                # Save data arrays
                for key, data in self.episode_data[0].items():
                    if isinstance(data, np.ndarray):
                        f.create_dataset(key, data=np.array([d[key] for d in self.episode_data]))
                    else:
                        f.create_dataset(key, data=[d[key] for d in self.episode_data])
            
            print(f"Saved episode {self.current_episode} with {len(self.episode_data)} steps")
        else:
            print("No data to save for current episode")
    
    def reset_robots(self):
        """Reset robots to home position."""
        self.left_arm_joints = np.zeros(7)
        self.right_arm_joints = np.zeros(7)
        self.actuator_position = 0.1
        print("Reset robots to home position")
    
    def record_step(self, observations: Dict[str, Any]):
        """Record one step of data if recording is active."""
        
        if not self.is_recording:
            return
        
        current_time = time.time()
        if current_time - self.last_record_time < 1.0 / self.recording_frequency:
            return
        
        # Prepare step data
        step_data = {
            'timestamp': current_time,
            'left_arm_joints': self.left_arm_joints.copy(),
            'right_arm_joints': self.right_arm_joints.copy(),
            'actuator_position': self.actuator_position,
            'observations': observations
        }
        
        # Add to episode data
        self.episode_data.append(step_data)
        self.last_record_time = current_time
    
    def get_current_actions(self) -> Dict[str, np.ndarray]:
        """Get current actions for all agents."""
        return {
            'left_arm': self.left_arm_joints.copy(),
            'right_arm': self.right_arm_joints.copy(),
            'actuator': np.array([self.actuator_position])
        }
    
    def draw_control_interface(self, observations: Dict[str, Any]) -> np.ndarray:
        """Draw the teleoperation control interface."""
        
        # Create interface image
        interface = np.zeros((600, 800, 3), dtype=np.uint8)
        
        # Title
        cv2.putText(interface, "Ironing Robot Teleoperation", (20, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Recording status
        status_color = (0, 255, 0) if self.is_recording else (0, 0, 255)
        status_text = "RECORDING" if self.is_recording else "NOT RECORDING"
        cv2.putText(interface, status_text, (20, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        
        # Episode info
        cv2.putText(interface, f"Episode: {self.current_episode}", (20, 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(interface, f"Steps: {len(self.episode_data)}", (20, 120), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Control instructions
        y_pos = 160
        instructions = [
            "CONTROLS:",
            "Left Arm: W/S, A/D, Q/E, R/F, T/G, Y/H, U/J",
            "Right Arm: Arrow Keys, I/K, O/L, P/;, [/', ]/\\",
            "Actuator: SPACE (up), SHIFT (down)",
            "",
            "RECORDING:",
            "ENTER: Start/Stop recording",
            "N: New episode",
            "S: Save episode",
            "R: Reset robots",
            "ESC: Exit"
        ]
        
        for instruction in instructions:
            cv2.putText(interface, instruction, (20, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            y_pos += 25
        
        # Current joint angles
        y_pos = 400
        cv2.putText(interface, "CURRENT JOINT ANGLES:", (20, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        y_pos += 30
        
        # Left arm joints
        cv2.putText(interface, "Left Arm:", (20, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y_pos += 20
        
        for i, angle in enumerate(self.left_arm_joints):
            cv2.putText(interface, f"  Joint {i}: {angle:.3f} rad", (40, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
            y_pos += 15
        
        # Right arm joints
        y_pos += 10
        cv2.putText(interface, "Right Arm:", (20, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y_pos += 20
        
        for i, angle in enumerate(self.right_arm_joints):
            cv2.putText(interface, f"  Joint {i}: {angle:.3f} rad", (40, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
            y_pos += 15
        
        # Actuator position
        y_pos += 10
        cv2.putText(interface, f"Actuator: {self.actuator_position:.3f} m", (20, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return interface


class DataCollectionPipeline:
    """
    Main pipeline for collecting expert demonstrations.
    """
    
    def __init__(self, config_path: str):
        
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Initialize environment
        from ..sim.ironing_env import IroningEnvironment
        self.env = IroningEnvironment(config=self.config)
        
        # Initialize teleoperation controller
        self.controller = TeleoperationController(self.config)
        
        # Data collection state
        self.running = True
        
    def run(self):
        """Run the data collection pipeline."""
        
        print("Starting data collection pipeline...")
        print("Use the control interface to teleoperate the robots.")
        print("Press ESC to exit.")
        
        # Reset environment
        observations = self.env.reset()
        
        while self.running:
            # Get current actions from controller
            actions = self.controller.get_current_actions()
            
            # Step environment
            step_result = self.env.step(actions)
            observations = step_result['observations']
            
            # Record step if recording
            self.controller.record_step(observations)
            
            # Draw control interface
            interface = self.controller.draw_control_interface(observations)
            
            # Display interface
            cv2.imshow(self.controller.control_window, interface)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key != 255:  # If a key was pressed
                if not self.controller.process_key_input(key):
                    self.running = False
                    break
            
            # Check if episode is done
            if step_result['done']:
                print("Episode completed!")
                self.controller.new_episode()
                observations = self.env.reset()
        
        # Cleanup
        cv2.destroyAllWindows()
        self.env.close()
        print("Data collection pipeline stopped.")


def main():
    """Main function for running data collection."""
    
    import argparse
    
    parser = argparse.ArgumentParser(description="Ironing Robot Data Collection")
    parser.add_argument("--config", type=str, default="configs/default.yaml",
                       help="Path to configuration file")
    parser.add_argument("--data_path", type=str, default="./data/expert_demonstrations",
                       help="Path to save demonstration data")
    
    args = parser.parse_args()
    
    # Run data collection pipeline
    pipeline = DataCollectionPipeline(args.config)
    pipeline.run()


if __name__ == "__main__":
    main()
