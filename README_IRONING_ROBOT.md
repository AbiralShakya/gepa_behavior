# Ironing Robot System

An autonomous ironing robot system using world models and behavior cloning for coordinated multi-agent control of dual robotic arms and a linear actuator with heating pad.

## System Overview

This system implements a complete pipeline for training and deploying an autonomous ironing robot:

1. **World Model**: Learns fabric physics and multi-agent dynamics
2. **Behavior Model**: Learns coordinated actions from expert demonstrations  
3. **Simulation Environment**: High-fidelity PyBullet simulation with fabric physics
4. **Data Collection**: Teleoperation interface for expert demonstrations
5. **Deployment**: Real-time control on Jetson Nano with camera feedback

## Architecture

### World Model (`world_model_ironing.py`)
- **FabricPhysicsModel**: Neural network for predicting fabric deformation
- **MultiAgentStateEncoder**: Encodes states of multiple agents (arms + actuator)
- **VisualEncoder**: Processes camera images into feature vectors
- **WorldModelTransformer**: Sequential prediction using Transformer architecture

### Behavior Model (`behavior_model_ironing.py`)
- **MultiAgentAttention**: Coordinates actions between agents
- **TemporalEncoder**: Captures temporal dependencies in observations
- **ActionSmoother**: Prevents jerky movements
- **BehaviorCloningModel**: Learns from expert demonstrations

### Simulation Environment (`ironing_env.py`)
- **FabricMesh**: Realistic cloth simulation with mesh-based physics
- **LinearActuator**: Heating pad with vertical movement control
- **IroningEnvironment**: Complete PyBullet simulation environment
- **Domain Randomization**: Sim-to-real transfer capabilities

## Quick Start

### 1. Setup Environment

```bash
# Install dependencies
pip install -r requirements.txt

# For PyBullet simulation
pip install pybullet

# For Jetson Nano deployment
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 2. Data Collection

Collect expert demonstrations using the teleoperation interface:

```bash
python -m gepa.data_collection.teleoperation --config configs/default.yaml
```

**Controls:**
- **Left Arm**: W/S, A/D, Q/E, R/F, T/G, Y/H, U/J
- **Right Arm**: Arrow Keys, I/K, O/L, P/;, [/', ]/\\
- **Actuator**: SPACE (up), SHIFT (down)
- **Recording**: ENTER (start/stop), N (new episode), S (save)

### 3. Training

Train the complete system:

```bash
# Train world model only
python -m gepa.training.ironing_trainer --mode world_model --data_path ./data/expert_demonstrations

# Train behavior model only  
python -m gepa.training.ironing_trainer --mode behavior_model --data_path ./data/expert_demonstrations

# Train complete system
python -m gepa.training.ironing_trainer --mode complete --data_path ./data/expert_demonstrations
```

### 4. Simulation Testing

Test the trained models in simulation:

```bash
python -c "
from gepa.sim.ironing_env import IroningEnvironment
import yaml

# Load config
with open('configs/default.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Create environment
env = IroningEnvironment(config=config)

# Run simulation
obs = env.reset()
for i in range(100):
    # Random actions for testing
    actions = {
        'left_arm': np.random.uniform(-0.5, 0.5, 7),
        'right_arm': np.random.uniform(-0.5, 0.5, 7), 
        'actuator': np.array([0.1])
    }
    
    result = env.step(actions)
    obs = result['observations']
    
    if result['done']:
        break

env.close()
"
```

### 5. Jetson Nano Deployment

Deploy on Jetson Nano with real-time camera:

```bash
python -m gepa.deployment.jetson_deploy --config configs/default.yaml
```

**Controls:**
- `q`: Quit
- `e`: Emergency stop
- `r`: Resume operation

## Configuration

The system is configured via `configs/default.yaml`:

### Simulation Configuration
```yaml
simulation:
  robots:
    left_arm:
      urdf: kuka_iiwa/model.urdf
      base_position: [-0.3, 0.0, 0.0]
    right_arm:
      urdf: kuka_iiwa/model.urdf  
      base_position: [0.3, 0.0, 0.0]
  linear_actuator:
    position: [0.0, 0.0, 0.1]
    range: [0.0, 0.2]
  workspace:
    fabric_size: [0.6, 0.4, 0.001]
```

### Model Configuration
```yaml
model:
  architecture: transformer
  num_agents: 3
  agent_action_dims: [7, 7, 1]
  behavior_cloning:
    enabled: true
    temporal_window: 5
```

### World Model Configuration
```yaml
world_model:
  fabric_physics:
    mesh_resolution: 32
    material_properties:
      stiffness: 1000.0
      friction: 0.3
```

## Key Features

### 1. Multi-Agent Coordination
- **Attention Mechanism**: Agents attend to each other's states
- **Coordinated Actions**: Simultaneous control of dual arms + actuator
- **Safety Limits**: Joint limits and velocity constraints

### 2. Fabric Physics
- **Mesh-Based Simulation**: 32x32 grid for realistic cloth deformation
- **Material Properties**: Configurable stiffness, damping, friction
- **Contact Forces**: Real-time force feedback between agents and fabric

### 3. Domain Randomization
- **Lighting Variation**: Intensity and color temperature ranges
- **Material Properties**: Randomized friction and stiffness
- **Physics Parameters**: Gravity and timestep noise

### 4. Real-Time Deployment
- **Camera Interface**: Multi-camera setup with real-time processing
- **Action Smoothing**: Exponential smoothing for smooth movements
- **Emergency Stop**: Safety mechanisms for real hardware

## File Structure

```
gepa_behavior/
├── configs/
│   └── default.yaml              # Main configuration
├── src/gepa/
│   ├── models/
│   │   ├── world_model_ironing.py    # World model implementation
│   │   └── behavior_model_ironing.py # Behavior model implementation
│   ├── sim/
│   │   └── ironing_env.py            # PyBullet simulation environment
│   ├── data_collection/
│   │   └── teleoperation.py          # Expert data collection
│   ├── training/
│   │   └── ironing_trainer.py        # Training pipeline
│   └── deployment/
│       └── jetson_deploy.py          # Jetson Nano deployment
├── data/
│   └── expert_demonstrations/        # Collected expert data
└── checkpoints/                      # Trained model checkpoints
```

## Advanced Usage

### Custom Fabric Materials

Modify fabric properties in the simulation:

```python
fabric = FabricMesh(
    size=(0.6, 0.4),
    material_properties={
        'stiffness': 2000.0,  # Stiffer fabric
        'damping': 0.2,       # More damping
        'friction': 0.5       # Higher friction
    }
)
```

### Custom Robot URDFs

Replace default KUKA arms with custom robots:

```yaml
simulation:
  robots:
    left_arm:
      urdf: path/to/custom_robot.urdf
      base_position: [-0.4, 0.0, 0.0]
```

### Multi-Camera Setup

Configure multiple cameras for different viewpoints:

```yaml
simulation:
  cameras:
    overhead:
      position: [0.0, 0.0, 1.0]
      resolution: [640, 480]
    side_view:
      position: [0.5, 0.0, 0.3]
      resolution: [640, 480]
```

## Troubleshooting

### Common Issues

1. **PyBullet GUI not showing**: Set `gui: true` in config
2. **Camera not working**: Check camera permissions and device ID
3. **Model loading errors**: Ensure checkpoints exist in `checkpoints/` directory
4. **Memory issues**: Reduce batch size or sequence length

### Performance Optimization

1. **Jetson Nano**: Use TensorRT for inference acceleration
2. **Simulation**: Reduce mesh resolution for faster physics
3. **Training**: Use mixed precision training with `torch.cuda.amp`

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this system in your research, please cite:

```bibtex
@software{ironing_robot_2024,
  title={Autonomous Ironing Robot System with World Models and Behavior Cloning},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/gepa_behavior}
}
```

## Acknowledgments

- PyBullet physics engine
- PyTorch deep learning framework
- KUKA LBR iiwa robot models
- NVIDIA Jetson platform
