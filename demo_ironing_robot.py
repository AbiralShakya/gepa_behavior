#!/usr/bin/env python3
"""
Demo script for the Ironing Robot System

This script demonstrates the complete pipeline:
1. Data collection with teleoperation
2. World model training
3. Behavior model training  
4. Simulation testing
5. Deployment interface

Usage:
    python demo_ironing_robot.py --mode [collect|train|simulate|deploy]
"""

import argparse
import yaml
import numpy as np
import torch
from pathlib import Path
import time

def demo_data_collection():
    """Demo data collection with teleoperation interface."""
    print("=== Data Collection Demo ===")
    print("Starting teleoperation interface...")
    print("Use keyboard controls to operate the robots:")
    print("- Left Arm: W/S, A/D, Q/E, R/F, T/G, Y/H, U/J")
    print("- Right Arm: Arrow Keys, I/K, O/L, P/;, [/', ]/\\")
    print("- Actuator: SPACE (up), SHIFT (down)")
    print("- Recording: ENTER (start/stop), N (new episode), S (save)")
    print("- Exit: ESC")
    
    try:
        from src.gepa.data_collection.teleoperation import DataCollectionPipeline
        
        # Load config
        with open('configs/default.yaml', 'r') as f:
            config = yaml.safe_load(f)
        
        # Run data collection
        pipeline = DataCollectionPipeline(config)
        pipeline.run()
        
    except ImportError as e:
        print(f"Import error: {e}")
        print("Make sure all dependencies are installed")
    except Exception as e:
        print(f"Error during data collection: {e}")

def demo_training():
    """Demo training pipeline."""
    print("=== Training Demo ===")
    print("Training world model and behavior model...")
    
    try:
        from src.gepa.training.ironing_trainer import IroningTrainer
        
        # Initialize trainer
        trainer = IroningTrainer('configs/default.yaml')
        
        # Check if data exists
        data_path = Path('./data/expert_demonstrations')
        if not data_path.exists() or len(list(data_path.glob('*.h5'))) == 0:
            print("No expert demonstration data found!")
            print("Please run data collection first: python demo_ironing_robot.py --mode collect")
            return
        
        # Train complete system
        trainer.train_complete_system(str(data_path))
        
        print("Training completed! Checkpoints saved to ./checkpoints/")
        
    except ImportError as e:
        print(f"Import error: {e}")
        print("Make sure all dependencies are installed")
    except Exception as e:
        print(f"Error during training: {e}")

def demo_simulation():
    """Demo simulation environment."""
    print("=== Simulation Demo ===")
    print("Running PyBullet simulation with random actions...")
    
    try:
        from src.gepa.sim.ironing_env import IroningEnvironment
        
        # Load config
        with open('configs/default.yaml', 'r') as f:
            config = yaml.safe_load(f)
        
        # Create environment
        env = IroningEnvironment(config=config)
        
        # Reset environment
        observations = env.reset()
        print("Environment reset successfully")
        
        # Run simulation for 100 steps
        for step in range(100):
            # Generate random actions
            actions = {
                'left_arm': np.random.uniform(-0.5, 0.5, 7),
                'right_arm': np.random.uniform(-0.5, 0.5, 7),
                'actuator': np.array([0.1 + 0.05 * np.sin(step * 0.1)])
            }
            
            # Step environment
            result = env.step(actions)
            observations = result['observations']
            
            # Print progress
            if step % 20 == 0:
                print(f"Step {step}: Left arm joints = {actions['left_arm'][:3]}")
            
            # Check if done
            if result['done']:
                print("Episode completed!")
                break
        
        # Close environment
        env.close()
        print("Simulation demo completed!")
        
    except ImportError as e:
        print(f"Import error: {e}")
        print("Make sure PyBullet is installed: pip install pybullet")
    except Exception as e:
        print(f"Error during simulation: {e}")

def demo_deployment():
    """Demo deployment interface."""
    print("=== Deployment Demo ===")
    print("Starting Jetson Nano deployment interface...")
    print("This will attempt to load trained models and start camera feed")
    print("Controls: q=quit, e=emergency stop, r=resume")
    
    try:
        from src.gepa.deployment.jetson_deploy import JetsonDeployment
        
        # Check if models exist
        checkpoint_dir = Path('./checkpoints')
        if not checkpoint_dir.exists():
            print("No checkpoints found! Please train models first:")
            print("python demo_ironing_robot.py --mode train")
            return
        
        # Run deployment
        deployment = JetsonDeployment('configs/default.yaml')
        deployment.run()
        
    except ImportError as e:
        print(f"Import error: {e}")
        print("Make sure all dependencies are installed")
    except Exception as e:
        print(f"Error during deployment: {e}")

def demo_model_architecture():
    """Demo model architecture without training."""
    print("=== Model Architecture Demo ===")
    print("Creating and testing model architectures...")
    
    try:
        from src.gepa.models.world_model_ironing import IroningWorldModel
        from src.gepa.models.behavior_model_ironing import IroningBehaviorModel
        
        # Load config
        with open('configs/default.yaml', 'r') as f:
            config = yaml.safe_load(f)
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")
        
        # Test world model
        print("Creating world model...")
        world_model = IroningWorldModel(config['world_model']).to(device)
        
        # Create dummy inputs
        batch_size, seq_len = 2, 10
        robot_states = torch.randn(batch_size, seq_len, 2, 14).to(device)
        actuator_states = torch.randn(batch_size, seq_len, 2).to(device)
        fabric_states = torch.randn(batch_size, seq_len, 32, 32, 3).to(device)
        images = torch.randn(batch_size, seq_len, 3, 64, 64).to(device)
        actions = torch.randn(batch_size, seq_len, 15).to(device)
        contact_forces = torch.randn(batch_size, seq_len, 1, 6).to(device)  # At least 1 contact
        
        # Test forward pass
        with torch.no_grad():
            predictions = world_model(
                robot_states=robot_states,
                actuator_states=actuator_states,
                fabric_states=fabric_states,
                images=images,
                actions=actions,
                contact_forces=contact_forces
            )
        
        print(f"World model output shapes:")
        for key, value in predictions.items():
            print(f"  {key}: {value.shape}")
        
        # Test behavior model
        print("Creating behavior model...")
        behavior_model = IroningBehaviorModel(config['model']).to(device)
        
        # Create dummy inputs
        images = torch.randn(batch_size, 5, 3, 64, 64).to(device)  # temporal_window=5
        proprioceptive_state = torch.randn(batch_size, 5, 30).to(device)
        
        # Test forward pass
        with torch.no_grad():
            actions = behavior_model(images, proprioceptive_state)
        
        print(f"Behavior model output shapes:")
        for key, value in actions.items():
            print(f"  {key}: {value.shape}")
        
        print("Model architecture demo completed successfully!")
        
    except ImportError as e:
        print(f"Import error: {e}")
        print("Make sure all dependencies are installed")
    except Exception as e:
        print(f"Error during model demo: {e}")

def main():
    """Main demo function."""
    
    parser = argparse.ArgumentParser(description="Ironing Robot System Demo")
    parser.add_argument("--mode", 
                       choices=["collect", "train", "simulate", "deploy", "models"],
                       default="simulate",
                       help="Demo mode to run")
    
    args = parser.parse_args()
    
    print("🤖 Ironing Robot System Demo")
    print("=" * 50)
    
    if args.mode == "collect":
        demo_data_collection()
    elif args.mode == "train":
        demo_training()
    elif args.mode == "simulate":
        demo_simulation()
    elif args.mode == "deploy":
        demo_deployment()
    elif args.mode == "models":
        demo_model_architecture()
    else:
        print("Invalid mode. Choose from: collect, train, simulate, deploy, models")

if __name__ == "__main__":
    main()
