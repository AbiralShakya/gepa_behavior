"""
PyBullet Simulation Environment for Ironing Robot System

This module implements a high-fidelity simulation environment with:
1. Dual robotic arms (KUKA LBR iiwa)
2. Linear actuator with heating pad
3. Realistic fabric physics
4. Multi-camera setup
5. Domain randomization for sim-to-real transfer
"""

import pybullet as p
import pybullet_data
import numpy as np
import cv2
import yaml
from typing import Dict, List, Tuple, Optional, Any
import math
import random
from pathlib import Path


class FabricMesh:
    """
    Represents a deformable fabric mesh for realistic cloth simulation.
    """
    
    def __init__(self, 
                 size: Tuple[float, float] = (0.6, 0.4),
                 resolution: Tuple[int, int] = (32, 32),
                 position: Tuple[float, float, float] = (0.0, 0.0, 0.025),
                 material_properties: Dict = None):
        
        self.size = size
        self.resolution = resolution
        self.position = position
        self.material_properties = material_properties or {
            'stiffness': 1000.0,
            'damping': 0.1,
            'friction': 0.3
        }
        
        # Create mesh vertices
        self.vertices = self._create_vertices()
        self.faces = self._create_faces()
        
        # PyBullet mesh ID
        self.mesh_id = None
        
    def _create_vertices(self) -> np.ndarray:
        """Create vertex positions for the fabric mesh."""
        vertices = []
        
        for i in range(self.resolution[1]):  # y direction
            for j in range(self.resolution[0]):  # x direction
                x = (j / (self.resolution[0] - 1) - 0.5) * self.size[0] + self.position[0]
                y = (i / (self.resolution[1] - 1) - 0.5) * self.size[1] + self.position[1]
                z = self.position[2]
                vertices.append([x, y, z])
        
        return np.array(vertices)
    
    def _create_faces(self) -> np.ndarray:
        """Create triangular faces for the fabric mesh."""
        faces = []
        
        for i in range(self.resolution[1] - 1):
            for j in range(self.resolution[0] - 1):
                # Current row indices
                idx = i * self.resolution[0] + j
                next_idx = (i + 1) * self.resolution[0] + j
                
                # Create two triangles for each quad
                # Triangle 1
                faces.append([idx, idx + 1, next_idx])
                # Triangle 2
                faces.append([idx + 1, next_idx + 1, next_idx])
        
        return np.array(faces)
    
    def load_to_pybullet(self, physics_client_id: int) -> int:
        """Load fabric mesh to PyBullet simulation."""
        
        # Create collision shape
        collision_shape = p.createCollisionShape(
            p.GEOM_MESH,
            vertices=self.vertices,
            indices=self.faces,
            physicsClientId=physics_client_id
        )
        
        # Create visual shape
        visual_shape = p.createVisualShape(
            p.GEOM_MESH,
            vertices=self.vertices,
            indices=self.faces,
            rgbaColor=[0.8, 0.8, 0.9, 1.0],
            physicsClientId=physics_client_id
        )
        
        # Create multi-body
        self.mesh_id = p.createMultiBody(
            baseMass=0.1,  # Light fabric
            baseCollisionShapeIndex=collision_shape,
            baseVisualShapeIndex=visual_shape,
            basePosition=self.position,
            physicsClientId=physics_client_id
        )
        
        # Set material properties
        p.changeDynamics(
            self.mesh_id,
            -1,
            lateralFriction=self.material_properties['friction'],
            spinningFriction=0.1,
            rollingFriction=0.1,
            restitution=0.0,
            physicsClientId=physics_client_id
        )
        
        return self.mesh_id
    
    def get_vertex_positions(self, physics_client_id: int) -> np.ndarray:
        """Get current vertex positions from simulation."""
        if self.mesh_id is None:
            return self.vertices
        
        # Get mesh data
        mesh_data = p.getMeshData(self.mesh_id, -1, physicsClientId=physics_client_id)
        return np.array(mesh_data[1])  # Vertex positions


class LinearActuator:
    """
    Linear actuator with heating pad for ironing.
    """
    
    def __init__(self, 
                 position: Tuple[float, float, float] = (0.0, 0.0, 0.1),
                 range_limits: Tuple[float, float] = (0.0, 0.2),
                 physics_client_id: int = 0):
        
        self.position = position
        self.range_limits = range_limits
        self.physics_client_id = physics_client_id
        
        # Current state
        self.current_position = position[2]  # Z position
        self.target_position = position[2]
        self.velocity = 0.0
        
        # Physical properties
        self.max_velocity = 0.01  # m/s
        self.max_force = 50.0  # N
        self.mass = 0.5  # kg
        
        # Create actuator body
        self.body_id = self._create_actuator_body()
        
    def _create_actuator_body(self) -> int:
        """Create the linear actuator body in PyBullet."""
        
        # Create box shape for the heating pad
        box_size = [0.1, 0.1, 0.02]  # Small heating pad
        
        collision_shape = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=box_size,
            physicsClientId=self.physics_client_id
        )
        
        visual_shape = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=box_size,
            rgbaColor=[0.9, 0.1, 0.1, 1.0],  # Red for heating pad
            physicsClientId=self.physics_client_id
        )
        
        # Create body
        body_id = p.createMultiBody(
            baseMass=self.mass,
            baseCollisionShapeIndex=collision_shape,
            baseVisualShapeIndex=visual_shape,
            basePosition=self.position,
            physicsClientId=self.physics_client_id
        )
        
        # Set dynamics
        p.changeDynamics(
            body_id,
            -1,
            lateralFriction=0.8,
            spinningFriction=0.1,
            rollingFriction=0.1,
            restitution=0.0,
            physicsClientId=self.physics_client_id
        )
        
        return body_id
    
    def set_target_position(self, target_z: float):
        """Set target position for the actuator."""
        self.target_position = np.clip(target_z, self.range_limits[0], self.range_limits[1])
    
    def step(self, dt: float):
        """Update actuator position based on target."""
        
        # Calculate desired velocity
        position_error = self.target_position - self.current_position
        desired_velocity = np.clip(position_error / dt, -self.max_velocity, self.max_velocity)
        
        # Update velocity with simple control
        self.velocity = desired_velocity
        
        # Update position
        self.current_position += self.velocity * dt
        
        # Apply forces to move the body
        if abs(position_error) > 0.001:  # If not at target
            force = np.clip(position_error * 1000, -self.max_force, self.max_force)
            p.applyExternalForce(
                self.body_id,
                -1,
                forceObj=[0, 0, force],
                posObj=[0, 0, 0],
                flags=p.WORLD_FRAME,
                physicsClientId=self.physics_client_id
            )
        
        # Update body position
        p.resetBasePositionAndOrientation(
            self.body_id,
            [self.position[0], self.position[1], self.current_position],
            [0, 0, 0, 1],
            physicsClientId=self.physics_client_id
        )


class IroningEnvironment:
    """
    Main simulation environment for the ironing robot system.
    """
    
    def __init__(self, config_path: str = None, config: Dict = None):
        
        # Load configuration
        if config_path:
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        elif config:
            self.config = config
        else:
            raise ValueError("Either config_path or config must be provided")
        
        # Extract simulation config
        sim_config = self.config['simulation']
        self.timestep = sim_config['timestep']
        self.gravity = sim_config['gravity']
        self.gui = sim_config['gui']
        self.max_steps = sim_config['max_steps_per_episode']
        
        # Initialize PyBullet
        self.physics_client_id = self._initialize_pybullet()
        
        # Load robots
        self.robots = self._load_robots(sim_config['robots'])
        
        # Load workspace
        self.table_id = self._create_table(sim_config['workspace'])
        
        # Load fabric
        fabric_config = sim_config['workspace']
        self.fabric = FabricMesh(
            size=fabric_config['fabric_size'][:2],
            position=fabric_config['fabric_position'],
            material_properties={
                'stiffness': 1000.0,
                'damping': 0.1,
                'friction': 0.3
            }
        )
        self.fabric.load_to_pybullet(self.physics_client_id)
        
        # Load linear actuator
        actuator_config = sim_config['linear_actuator']
        self.actuator = LinearActuator(
            position=actuator_config['position'],
            range_limits=actuator_config['range'],
            physics_client_id=self.physics_client_id
        )
        
        # Setup cameras
        self.cameras = self._setup_cameras(sim_config['cameras'])
        
        # Domain randomization
        self.domain_randomization = self.config.get('domain_randomization', {})
        self.randomize_environment()
        
        # State tracking
        self.step_count = 0
        self.episode_count = 0
        
    def _initialize_pybullet(self) -> int:
        """Initialize PyBullet physics simulation."""
        
        if self.gui:
            physics_client_id = p.connect(p.GUI)
        else:
            physics_client_id = p.connect(p.DIRECT)
        
        # Set additional search path
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        
        # Set gravity
        p.setGravity(0, 0, self.gravity, physicsClientId=physics_client_id)
        
        # Set timestep
        p.setTimeStep(self.timestep, physicsClientId=physics_client_id)
        
        # Load ground plane
        p.loadURDF("plane.urdf", physicsClientId=physics_client_id)
        
        return physics_client_id
    
    def _load_robots(self, robots_config: Dict) -> Dict[str, int]:
        """Load dual robotic arms."""
        
        robots = {}
        
        for robot_name, robot_config in robots_config.items():
            # Load URDF
            urdf_path = robot_config['urdf']
            base_position = robot_config['base_position']
            base_orientation = p.getQuaternionFromEuler(robot_config['base_orientation_euler'])
            
            robot_id = p.loadURDF(
                urdf_path,
                basePosition=base_position,
                baseOrientation=base_orientation,
                physicsClientId=self.physics_client_id
            )
            
            robots[robot_name] = robot_id
            
            # Set joint limits
            joint_limits = robot_config['joint_limits']
            for i, limits in enumerate(joint_limits):
                p.changeDynamics(
                    robot_id,
                    i,
                    jointLowerLimit=limits[0],
                    jointUpperLimit=limits[1],
                    physicsClientId=self.physics_client_id
                )
        
        return robots
    
    def _create_table(self, workspace_config: Dict) -> int:
        """Create the ironing table."""
        
        table_size = workspace_config['table_size']
        table_position = workspace_config['table_position']
        
        # Create table collision shape
        collision_shape = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=[s/2 for s in table_size],
            physicsClientId=self.physics_client_id
        )
        
        # Create table visual shape
        visual_shape = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[s/2 for s in table_size],
            rgbaColor=[0.6, 0.4, 0.2, 1.0],  # Brown table
            physicsClientId=self.physics_client_id
        )
        
        # Create table body
        table_id = p.createMultiBody(
            baseMass=0,  # Static table
            baseCollisionShapeIndex=collision_shape,
            baseVisualShapeIndex=visual_shape,
            basePosition=table_position,
            physicsClientId=self.physics_client_id
        )
        
        return table_id
    
    def _setup_cameras(self, cameras_config: Dict) -> Dict[str, Dict]:
        """Setup multiple cameras for visual feedback."""
        
        cameras = {}
        
        for camera_name, camera_config in cameras_config.items():
            if camera_config['enabled']:
                cameras[camera_name] = {
                    'position': camera_config['position'],
                    'orientation': p.getQuaternionFromEuler(camera_config['orientation_euler']),
                    'resolution': camera_config['resolution'],
                    'fov': camera_config['fov']
                }
        
        return cameras
    
    def get_camera_image(self, camera_name: str) -> np.ndarray:
        """Capture image from specified camera."""
        
        if camera_name not in self.cameras:
            raise ValueError(f"Camera {camera_name} not found")
        
        camera = self.cameras[camera_name]
        
        # Get camera matrix
        view_matrix = p.computeViewMatrix(
            cameraEyePosition=camera['position'],
            cameraTargetPosition=[0, 0, 0],  # Look at origin
            cameraUpVector=[0, 0, 1],
            physicsClientId=self.physics_client_id
        )
        
        # Get projection matrix
        projection_matrix = p.computeProjectionMatrixFOV(
            fov=camera['fov'],
            aspect=1.0,
            nearVal=0.01,
            farVal=10.0,
            physicsClientId=self.physics_client_id
        )
        
        # Render image
        width, height = camera['resolution']
        _, _, rgb_img, depth_img, seg_img = p.getCameraImage(
            width=width,
            height=height,
            viewMatrix=view_matrix,
            projectionMatrix=projection_matrix,
            physicsClientId=self.physics_client_id
        )
        
        # Convert to numpy array
        rgb_img = np.array(rgb_img)[:, :, :3]  # Remove alpha channel
        rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)
        
        return rgb_img
    
    def get_robot_state(self, robot_name: str) -> Dict[str, np.ndarray]:
        """Get current state of specified robot."""
        
        if robot_name not in self.robots:
            raise ValueError(f"Robot {robot_name} not found")
        
        robot_id = self.robots[robot_name]
        
        # Get joint states
        joint_states = p.getJointStates(robot_id, range(p.getNumJoints(robot_id)), 
                                      physicsClientId=self.physics_client_id)
        
        positions = np.array([state[0] for state in joint_states])
        velocities = np.array([state[1] for state in joint_states])
        
        # Get end-effector pose
        ee_link = p.getNumJoints(robot_id) - 1  # Last link
        ee_state = p.getLinkState(robot_id, ee_link, physicsClientId=self.physics_client_id)
        ee_position = np.array(ee_state[0])
        ee_orientation = np.array(ee_state[1])
        
        return {
            'joint_positions': positions,
            'joint_velocities': velocities,
            'ee_position': ee_position,
            'ee_orientation': ee_orientation
        }
    
    def set_robot_actions(self, robot_name: str, joint_actions: np.ndarray):
        """Set joint actions for specified robot."""
        
        if robot_name not in self.robots:
            raise ValueError(f"Robot {robot_name} not found")
        
        robot_id = self.robots[robot_name]
        
        # Apply joint actions
        for i, action in enumerate(joint_actions):
            p.setJointMotorControl2(
                robot_id,
                i,
                p.POSITION_CONTROL,
                targetPosition=action,
                physicsClientId=self.physics_client_id
            )
    
    def set_actuator_action(self, target_position: float):
        """Set target position for linear actuator."""
        self.actuator.set_target_position(target_position)
    
    def get_actuator_state(self) -> Dict[str, float]:
        """Get current state of linear actuator."""
        return {
            'position': self.actuator.current_position,
            'velocity': self.actuator.velocity,
            'target_position': self.actuator.target_position
        }
    
    def get_fabric_state(self) -> np.ndarray:
        """Get current fabric mesh state."""
        return self.fabric.get_vertex_positions(self.physics_client_id)
    
    def get_contact_forces(self) -> List[Dict]:
        """Get contact forces between agents and fabric."""
        
        contacts = []
        
        # Check contacts between robots and fabric
        for robot_name, robot_id in self.robots.items():
            contact_points = p.getContactPoints(
                robot_id, self.fabric.mesh_id,
                physicsClientId=self.physics_client_id
            )
            
            for contact in contact_points:
                contacts.append({
                    'agent': robot_name,
                    'position': contact[5],  # Contact position
                    'force': contact[9],    # Normal force
                    'friction': contact[10]  # Friction force
                })
        
        # Check contacts between actuator and fabric
        if self.actuator.body_id:
            contact_points = p.getContactPoints(
                self.actuator.body_id, self.fabric.mesh_id,
                physicsClientId=self.physics_client_id
            )
            
            for contact in contact_points:
                contacts.append({
                    'agent': 'actuator',
                    'position': contact[5],
                    'force': contact[9],
                    'friction': contact[10]
                })
        
        return contacts
    
    def randomize_environment(self):
        """Apply domain randomization for sim-to-real transfer."""
        
        if not self.domain_randomization.get('enabled', False):
            return
        
        # Randomize lighting
        if 'lighting' in self.domain_randomization:
            lighting_config = self.domain_randomization['lighting']
            # Note: PyBullet doesn't have advanced lighting controls
            # This would be implemented in more advanced simulators
        
        # Randomize material properties
        if 'materials' in self.domain_randomization:
            materials = self.domain_randomization['materials']
            
            # Randomize fabric properties
            if 'fabric_friction_range' in materials:
                friction_range = materials['fabric_friction_range']
                friction = random.uniform(friction_range[0], friction_range[1])
                p.changeDynamics(
                    self.fabric.mesh_id,
                    -1,
                    lateralFriction=friction,
                    physicsClientId=self.physics_client_id
                )
        
        # Randomize physics parameters
        if 'physics' in self.domain_randomization:
            physics = self.domain_randomization['physics']
            
            # Add noise to gravity
            if 'gravity_noise' in physics:
                noise = random.uniform(-physics['gravity_noise'], physics['gravity_noise'])
                p.setGravity(0, 0, self.gravity + noise, physicsClientId=self.physics_client_id)
    
    def step(self, actions: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Execute one simulation step."""
        
        # Apply robot actions
        if 'left_arm' in actions:
            self.set_robot_actions('left_arm', actions['left_arm'])
        
        if 'right_arm' in actions:
            self.set_robot_actions('right_arm', actions['right_arm'])
        
        # Apply actuator action
        if 'actuator' in actions:
            self.set_actuator_action(actions['actuator'][0])
        
        # Update actuator
        self.actuator.step(self.timestep)
        
        # Step simulation
        p.stepSimulation(physicsClientId=self.physics_client_id)
        
        # Update step count
        self.step_count += 1
        
        # Collect observations
        observations = self.get_observations()
        
        # Check if episode is done
        done = self.step_count >= self.max_steps
        
        return {
            'observations': observations,
            'done': done,
            'info': {
                'step_count': self.step_count,
                'episode_count': self.episode_count
            }
        }
    
    def get_observations(self) -> Dict[str, Any]:
        """Get current observations from all sensors."""
        
        observations = {}
        
        # Robot states
        for robot_name in self.robots.keys():
            robot_state = self.get_robot_state(robot_name)
            observations[f'{robot_name}_state'] = robot_state
        
        # Actuator state
        actuator_state = self.get_actuator_state()
        observations['actuator_state'] = actuator_state
        
        # Fabric state
        fabric_state = self.get_fabric_state()
        observations['fabric_state'] = fabric_state
        
        # Contact forces
        contact_forces = self.get_contact_forces()
        observations['contact_forces'] = contact_forces
        
        # Camera images
        for camera_name in self.cameras.keys():
            image = self.get_camera_image(camera_name)
            observations[f'{camera_name}_image'] = image
        
        return observations
    
    def reset(self) -> Dict[str, Any]:
        """Reset environment to initial state."""
        
        # Reset step count
        self.step_count = 0
        
        # Reset robots to initial positions
        for robot_name, robot_id in self.robots.items():
            # Reset to home position (all joints at 0)
            num_joints = p.getNumJoints(robot_id, physicsClientId=self.physics_client_id)
            for i in range(num_joints):
                p.resetJointState(robot_id, i, 0, 0, physicsClientId=self.physics_client_id)
        
        # Reset actuator
        self.actuator.current_position = self.actuator.position[2]
        self.actuator.target_position = self.actuator.position[2]
        self.actuator.velocity = 0.0
        
        # Reset fabric to flat state
        # (This would require more complex mesh manipulation in practice)
        
        # Apply domain randomization
        self.randomize_environment()
        
        # Get initial observations
        observations = self.get_observations()
        
        return observations
    
    def close(self):
        """Close the simulation environment."""
        p.disconnect(physicsClientId=self.physics_client_id)
