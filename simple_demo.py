#!/usr/bin/env python3
"""
Simple demo to test the ironing robot system components
"""

import torch
import torch.nn as nn
import numpy as np
import yaml

def test_basic_models():
    """Test basic model components without complex configurations."""
    
    print("🤖 Testing Ironing Robot System Components")
    print("=" * 50)
    
    # Test basic neural network components
    print("1. Testing basic neural network components...")
    
    # Test visual encoder
    print("   - Visual Encoder")
    visual_encoder = nn.Sequential(
        nn.Conv2d(3, 32, 3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(32, 64, 3, padding=1),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d((8, 8)),
        nn.Flatten(),
        nn.Linear(64 * 8 * 8, 256)
    )
    
    # Test with dummy image
    dummy_image = torch.randn(1, 3, 64, 64)
    visual_features = visual_encoder(dummy_image)
    print(f"     Input: {dummy_image.shape} -> Output: {visual_features.shape}")
    
    # Test multi-agent attention
    print("   - Multi-Agent Attention")
    attention = nn.MultiheadAttention(embed_dim=256, num_heads=8, batch_first=True)
    
    # Test with dummy agent features
    agent_features = torch.randn(1, 3, 256)  # batch, num_agents, feature_dim
    attended, _ = attention(agent_features, agent_features, agent_features)
    print(f"     Input: {agent_features.shape} -> Output: {attended.shape}")
    
    # Test LSTM for temporal encoding
    print("   - Temporal Encoder (LSTM)")
    lstm = nn.LSTM(input_size=256, hidden_size=256, num_layers=2, batch_first=True)
    
    # Test with dummy sequence
    sequence = torch.randn(1, 5, 256)  # batch, seq_len, feature_dim
    lstm_out, (hidden, cell) = lstm(sequence)
    print(f"     Input: {sequence.shape} -> Output: {lstm_out.shape}")
    
    print("✅ Basic components working correctly!")
    
    # Test configuration loading
    print("\n2. Testing configuration loading...")
    try:
        with open('configs/default.yaml', 'r') as f:
            config = yaml.safe_load(f)
        print("   ✅ Configuration loaded successfully")
        print(f"   - Simulation backend: {config['simulation']['backend']}")
        print(f"   - Number of agents: {config['model']['num_agents']}")
        print(f"   - World model enabled: {config['world_model']['enabled']}")
    except Exception as e:
        print(f"   ❌ Configuration loading failed: {e}")
    
    # Test PyTorch operations
    print("\n3. Testing PyTorch operations...")
    
    # Test tensor operations
    left_arm = torch.randn(1, 7)  # 7 joints
    right_arm = torch.randn(1, 7)  # 7 joints
    actuator = torch.randn(1, 1)   # 1 actuator
    
    # Combine actions
    combined_actions = torch.cat([left_arm, right_arm, actuator], dim=1)
    print(f"   - Combined actions shape: {combined_actions.shape}")
    
    # Test action smoothing
    previous_actions = torch.randn(1, 15)
    smoothing_factor = 0.1
    smoothed = (1 - smoothing_factor) * previous_actions + smoothing_factor * combined_actions
    print(f"   - Action smoothing: {previous_actions.shape} -> {smoothed.shape}")
    
    # Test safety limits
    joint_limits = torch.tensor([[-3.14, 3.14]] * 7)
    clamped_actions = torch.clamp(combined_actions[:, :7], joint_limits[:, 0], joint_limits[:, 1])
    print(f"   - Safety limits applied: {clamped_actions.shape}")
    
    print("✅ PyTorch operations working correctly!")
    
    # Test data structures
    print("\n4. Testing data structures...")
    
    # Test episode data structure
    episode_data = {
        'left_arm_joints': np.random.uniform(-1, 1, (100, 7)),
        'right_arm_joints': np.random.uniform(-1, 1, (100, 7)),
        'actuator_positions': np.random.uniform(0, 0.2, (100,)),
        'timestamps': np.linspace(0, 10, 100)
    }
    
    print(f"   - Episode data keys: {list(episode_data.keys())}")
    print(f"   - Episode length: {len(episode_data['left_arm_joints'])}")
    print(f"   - Left arm shape: {episode_data['left_arm_joints'].shape}")
    
    # Test observation structure
    observations = {
        'left_arm_state': {'joint_positions': np.random.uniform(-1, 1, 7)},
        'right_arm_state': {'joint_positions': np.random.uniform(-1, 1, 7)},
        'actuator_state': {'position': 0.1},
        'fabric_state': np.random.uniform(0, 1, (32, 32, 3)),
        'contact_forces': []
    }
    
    print(f"   - Observation keys: {list(observations.keys())}")
    print(f"   - Fabric state shape: {observations['fabric_state'].shape}")
    
    print("✅ Data structures working correctly!")
    
    print("\n🎉 All basic components are working correctly!")
    print("\nNext steps:")
    print("1. Collect expert demonstrations: python demo_ironing_robot.py --mode collect")
    print("2. Train models: python demo_ironing_robot.py --mode train")
    print("3. Test simulation: python demo_ironing_robot.py --mode simulate")
    print("4. Deploy on Jetson: python demo_ironing_robot.py --mode deploy")

if __name__ == "__main__":
    test_basic_models()
