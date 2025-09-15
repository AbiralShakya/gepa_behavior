"""
Training Pipeline for Ironing Robot System

This module implements the complete training pipeline for both the world model
and behavior model of the ironing robot system.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import yaml
import h5py
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import time
import logging
from tqdm import tqdm
import wandb
from collections import defaultdict

from ..models.world_model_ironing import IroningWorldModel
from ..models.behavior_model_ironing import IroningBehaviorModel
from ..sim.ironing_env import IroningEnvironment


class ExpertDataset(Dataset):
    """
    Dataset for loading expert demonstration data.
    """
    
    def __init__(self, 
                 data_path: str,
                 sequence_length: int = 10,
                 temporal_window: int = 5,
                 augment_data: bool = True):
        
        self.data_path = Path(data_path)
        self.sequence_length = sequence_length
        self.temporal_window = temporal_window
        self.augment_data = augment_data
        
        # Load all episode files
        self.episode_files = list(self.data_path.glob("episode_*.h5"))
        self.episode_files.sort()
        
        if len(self.episode_files) == 0:
            raise ValueError(f"No episode files found in {data_path}")
        
        # Load and preprocess data
        self.sequences = self._load_sequences()
        
        print(f"Loaded {len(self.sequences)} sequences from {len(self.episode_files)} episodes")
    
    def _load_sequences(self) -> List[Dict[str, Any]]:
        """Load and preprocess sequences from episode files."""
        
        sequences = []
        
        for episode_file in self.episode_files:
            with h5py.File(episode_file, 'r') as f:
                # Get episode data
                num_steps = f.attrs['num_steps']
                
                if num_steps < self.sequence_length:
                    continue  # Skip episodes that are too short
                
                # Extract data arrays
                left_arm_joints = f['left_arm_joints'][:]
                right_arm_joints = f['right_arm_joints'][:]
                actuator_positions = f['actuator_position'][:]
                
                # Create sequences
                for start_idx in range(num_steps - self.sequence_length + 1):
                    end_idx = start_idx + self.sequence_length
                    
                    sequence = {
                        'left_arm_joints': left_arm_joints[start_idx:end_idx],
                        'right_arm_joints': right_arm_joints[start_idx:end_idx],
                        'actuator_positions': actuator_positions[start_idx:end_idx],
                        'episode_id': f.attrs['episode_id'],
                        'start_idx': start_idx
                    }
                    
                    sequences.append(sequence)
        
        return sequences
    
    def __len__(self) -> int:
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a sequence for training."""
        
        sequence = self.sequences[idx]
        
        # Convert to tensors
        data = {
            'left_arm_joints': torch.FloatTensor(sequence['left_arm_joints']),
            'right_arm_joints': torch.FloatTensor(sequence['right_arm_joints']),
            'actuator_positions': torch.FloatTensor(sequence['actuator_positions']),
        }
        
        # Data augmentation
        if self.augment_data:
            data = self._augment_sequence(data)
        
        return data
    
    def _augment_sequence(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Apply data augmentation to sequence."""
        
        # Add noise to joint angles
        noise_std = 0.01
        data['left_arm_joints'] += torch.randn_like(data['left_arm_joints']) * noise_std
        data['right_arm_joints'] += torch.randn_like(data['right_arm_joints']) * noise_std
        data['actuator_positions'] += torch.randn_like(data['actuator_positions']) * noise_std * 0.1
        
        # Temporal jittering (slight time shifts)
        if np.random.random() < 0.3:
            shift = np.random.randint(-2, 3)
            if shift != 0:
                for key in data:
                    data[key] = torch.roll(data[key], shift, dims=0)
        
        return data


class WorldModelTrainer:
    """
    Trainer for the world model.
    """
    
    def __init__(self, config: Dict, device: str = "cuda"):
        
        self.config = config
        self.device = device
        
        # Initialize world model
        world_model_config = config['world_model']
        self.world_model = IroningWorldModel(world_model_config).to(device)
        
        # Initialize optimizer
        self.optimizer = optim.AdamW(
            self.world_model.parameters(),
            lr=world_model_config['learning_rate'],
            weight_decay=world_model_config['weight_decay']
        )
        
        # Training state
        self.epoch = 0
        self.global_step = 0
        self.best_loss = float('inf')
        
        # Setup logging
        self.setup_logging()
    
    def setup_logging(self):
        """Setup logging and wandb."""
        
        # Initialize wandb
        wandb.init(
            project="ironing-robot-world-model",
            config=self.config,
            name=f"world_model_{int(time.time())}"
        )
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """Train for one epoch."""
        
        self.world_model.train()
        epoch_losses = defaultdict(list)
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {self.epoch}")
        
        for batch_idx, batch in enumerate(progress_bar):
            
            # Move batch to device
            batch = {k: v.to(self.device) for k, v in batch.items()}
            
            # Prepare inputs (simplified - would need actual simulation data)
            # In practice, this would come from the simulation environment
            robot_states = self._prepare_robot_states(batch)
            actuator_states = self._prepare_actuator_states(batch)
            fabric_states = self._prepare_fabric_states(batch)
            images = self._prepare_images(batch)
            actions = self._prepare_actions(batch)
            contact_forces = self._prepare_contact_forces(batch)
            
            # Forward pass
            predictions = self.world_model(
                robot_states=robot_states,
                actuator_states=actuator_states,
                fabric_states=fabric_states,
                images=images,
                actions=actions,
                contact_forces=contact_forces
            )
            
            # Compute targets (next states)
            targets = self._compute_targets(batch)
            
            # Compute loss
            losses = self.world_model.compute_loss(predictions, targets)
            
            # Backward pass
            self.optimizer.zero_grad()
            losses['total'].backward()
            torch.nn.utils.clip_grad_norm_(self.world_model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # Log losses
            for key, loss in losses.items():
                epoch_losses[key].append(loss.item())
            
            # Update progress bar
            progress_bar.set_postfix({
                'total_loss': f"{losses['total'].item():.4f}",
                'step': self.global_step
            })
            
            # Log to wandb
            if self.global_step % 10 == 0:
                wandb.log({
                    f"world_model/{key}": loss.item() 
                    for key, loss in losses.items()
                }, step=self.global_step)
            
            self.global_step += 1
        
        # Compute average losses
        avg_losses = {
            key: np.mean(losses) 
            for key, losses in epoch_losses.items()
        }
        
        return avg_losses
    
    def _prepare_robot_states(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Prepare robot states for world model input."""
        
        # Combine left and right arm states
        left_arm = batch['left_arm_joints']  # [batch, seq_len, 7]
        right_arm = batch['right_arm_joints']  # [batch, seq_len, 7]
        
        # Add velocities (simplified - would compute from positions)
        left_vel = torch.zeros_like(left_arm)
        right_vel = torch.zeros_like(right_arm)
        
        # Combine positions and velocities
        left_state = torch.cat([left_arm, left_vel], dim=-1)  # [batch, seq_len, 14]
        right_state = torch.cat([right_arm, right_vel], dim=-1)  # [batch, seq_len, 14]
        
        # Stack for multi-agent format
        robot_states = torch.stack([left_state, right_state], dim=2)  # [batch, seq_len, 2, 14]
        
        return robot_states
    
    def _prepare_actuator_states(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Prepare actuator states for world model input."""
        
        positions = batch['actuator_positions']  # [batch, seq_len]
        
        # Add velocities (simplified)
        velocities = torch.zeros_like(positions)
        
        # Combine position and velocity
        actuator_states = torch.stack([positions, velocities], dim=-1)  # [batch, seq_len, 2]
        
        return actuator_states
    
    def _prepare_fabric_states(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Prepare fabric states for world model input."""
        
        # Simplified fabric state (would be actual mesh data in practice)
        batch_size, seq_len = batch['left_arm_joints'].shape[:2]
        fabric_states = torch.zeros(batch_size, seq_len, 32, 32, 3)
        
        return fabric_states
    
    def _prepare_images(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Prepare images for world model input."""
        
        # Simplified image data (would be actual camera images in practice)
        batch_size, seq_len = batch['left_arm_joints'].shape[:2]
        images = torch.zeros(batch_size, seq_len, 3, 64, 64)
        
        return images
    
    def _prepare_actions(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Prepare actions for world model input."""
        
        # Combine all agent actions
        left_arm = batch['left_arm_joints']  # [batch, seq_len, 7]
        right_arm = batch['right_arm_joints']  # [batch, seq_len, 7]
        actuator = batch['actuator_positions'].unsqueeze(-1)  # [batch, seq_len, 1]
        
        # Concatenate actions
        actions = torch.cat([left_arm, right_arm, actuator], dim=-1)  # [batch, seq_len, 15]
        
        return actions
    
    def _prepare_contact_forces(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Prepare contact forces for world model input."""
        
        # Simplified contact forces (would be actual contact data in practice)
        batch_size, seq_len = batch['left_arm_joints'].shape[:2]
        contact_forces = torch.zeros(batch_size, seq_len, 0, 6)  # No contacts initially
        
        return contact_forces
    
    def _compute_targets(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Compute target values for world model training."""
        
        # For world model, targets are the next states
        # This is simplified - in practice would need actual next state data
        
        batch_size, seq_len = batch['left_arm_joints'].shape[:2]
        
        targets = {
            'next_state': torch.zeros(batch_size, seq_len, 512),  # Simplified
            'visual_features': torch.zeros(batch_size, seq_len, 256),
            'contact_forces': torch.zeros(batch_size, seq_len, 6),
            'fabric_deformation': torch.zeros(batch_size, seq_len, 32, 32, 3)
        }
        
        return targets
    
    def save_checkpoint(self, filepath: str):
        """Save model checkpoint."""
        
        checkpoint = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'model_state_dict': self.world_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_loss': self.best_loss,
            'config': self.config
        }
        
        torch.save(checkpoint, filepath)
        self.logger.info(f"Saved checkpoint to {filepath}")
    
    def load_checkpoint(self, filepath: str):
        """Load model checkpoint."""
        
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.world_model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_loss = checkpoint['best_loss']
        
        self.logger.info(f"Loaded checkpoint from {filepath}")


class BehaviorModelTrainer:
    """
    Trainer for the behavior model.
    """
    
    def __init__(self, config: Dict, device: str = "cuda"):
        
        self.config = config
        self.device = device
        
        # Initialize behavior model
        model_config = config['model']
        self.behavior_model = IroningBehaviorModel(model_config).to(device)
        
        # Initialize optimizer
        self.optimizer = self.behavior_model.get_optimizer()
        
        # Training state
        self.epoch = 0
        self.global_step = 0
        self.best_loss = float('inf')
        
        # Setup logging
        self.setup_logging()
    
    def setup_logging(self):
        """Setup logging and wandb."""
        
        # Initialize wandb
        wandb.init(
            project="ironing-robot-behavior-model",
            config=self.config,
            name=f"behavior_model_{int(time.time())}"
        )
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """Train for one epoch."""
        
        self.behavior_model.train()
        epoch_losses = defaultdict(list)
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {self.epoch}")
        
        for batch_idx, batch in enumerate(progress_bar):
            
            # Move batch to device
            batch = {k: v.to(self.device) for k, v in batch.items()}
            
            # Prepare inputs
            images = self._prepare_images(batch)
            proprioceptive_state = self._prepare_proprioceptive_state(batch)
            
            # Forward pass
            predictions = self.behavior_model(images, proprioceptive_state)
            
            # Compute targets
            targets = self._compute_targets(batch)
            
            # Compute loss
            losses = self.behavior_model.compute_total_loss(predictions, targets)
            
            # Backward pass
            self.optimizer.zero_grad()
            losses['total'].backward()
            torch.nn.utils.clip_grad_norm_(self.behavior_model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # Log losses
            for key, loss in losses.items():
                epoch_losses[key].append(loss.item())
            
            # Update progress bar
            progress_bar.set_postfix({
                'total_loss': f"{losses['total'].item():.4f}",
                'step': self.global_step
            })
            
            # Log to wandb
            if self.global_step % 10 == 0:
                wandb.log({
                    f"behavior_model/{key}": loss.item() 
                    for key, loss in losses.items()
                }, step=self.global_step)
            
            self.global_step += 1
        
        # Compute average losses
        avg_losses = {
            key: np.mean(losses) 
            for key, losses in epoch_losses.items()
        }
        
        return avg_losses
    
    def _prepare_images(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Prepare images for behavior model input."""
        
        # Simplified image data (would be actual camera images in practice)
        batch_size, seq_len = batch['left_arm_joints'].shape[:2]
        images = torch.zeros(batch_size, seq_len, 3, 64, 64)
        
        return images
    
    def _prepare_proprioceptive_state(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Prepare proprioceptive state for behavior model input."""
        
        # Combine robot states
        left_arm = batch['left_arm_joints']  # [batch, seq_len, 7]
        right_arm = batch['right_arm_joints']  # [batch, seq_len, 7]
        actuator = batch['actuator_positions'].unsqueeze(-1)  # [batch, seq_len, 1]
        
        # Add velocities (simplified)
        left_vel = torch.zeros_like(left_arm)
        right_vel = torch.zeros_like(right_arm)
        actuator_vel = torch.zeros_like(actuator)
        
        # Combine all proprioceptive data
        proprioceptive_state = torch.cat([
            left_arm, left_vel,  # 14 dims
            right_arm, right_vel,  # 14 dims
            actuator, actuator_vel  # 2 dims
        ], dim=-1)  # [batch, seq_len, 30]
        
        return proprioceptive_state
    
    def _compute_targets(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Compute target values for behavior model training."""
        
        # For behavior cloning, targets are the expert actions
        left_arm = batch['left_arm_joints']  # [batch, seq_len, 7]
        right_arm = batch['right_arm_joints']  # [batch, seq_len, 7]
        actuator = batch['actuator_positions'].unsqueeze(-1)  # [batch, seq_len, 1]
        
        # Combine expert actions
        expert_actions = torch.cat([left_arm, right_arm, actuator], dim=-1)  # [batch, seq_len, 15]
        
        targets = {
            'expert_actions': expert_actions
        }
        
        return targets
    
    def save_checkpoint(self, filepath: str):
        """Save model checkpoint."""
        
        checkpoint = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'model_state_dict': self.behavior_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_loss': self.best_loss,
            'config': self.config
        }
        
        torch.save(checkpoint, filepath)
        self.logger.info(f"Saved checkpoint to {filepath}")
    
    def load_checkpoint(self, filepath: str):
        """Load model checkpoint."""
        
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.behavior_model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_loss = checkpoint['best_loss']
        
        self.logger.info(f"Loaded checkpoint from {filepath}")


class IroningTrainer:
    """
    Main trainer for the complete ironing robot system.
    """
    
    def __init__(self, config_path: str, device: str = "cuda"):
        
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.device = device
        
        # Initialize trainers
        self.world_model_trainer = WorldModelTrainer(self.config, device)
        self.behavior_model_trainer = BehaviorModelTrainer(self.config, device)
        
        # Training configuration
        self.num_epochs = self.config['experiment']['episodes']
        self.checkpoint_dir = Path(self.config['experiment']['checkpoint_dir'])
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
    def train_world_model(self, data_path: str):
        """Train the world model."""
        
        print("Training world model...")
        
        # Create dataset
        dataset = ExpertDataset(
            data_path=data_path,
            sequence_length=self.config['world_model']['rollout_horizon'],
            augment_data=True
        )
        
        # Create dataloader
        dataloader = DataLoader(
            dataset,
            batch_size=self.config['world_model']['batch_size'],
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        
        # Training loop
        for epoch in range(self.num_epochs):
            self.world_model_trainer.epoch = epoch
            
            # Train epoch
            losses = self.world_model_trainer.train_epoch(dataloader)
            
            # Log epoch results
            print(f"World Model Epoch {epoch}: {losses}")
            
            # Save checkpoint
            if losses['total'] < self.world_model_trainer.best_loss:
                self.world_model_trainer.best_loss = losses['total']
                checkpoint_path = self.checkpoint_dir / f"world_model_best.pth"
                self.world_model_trainer.save_checkpoint(checkpoint_path)
        
        print("World model training completed!")
    
    def train_behavior_model(self, data_path: str):
        """Train the behavior model."""
        
        print("Training behavior model...")
        
        # Create dataset
        dataset = ExpertDataset(
            data_path=data_path,
            sequence_length=self.config['model']['behavior_cloning']['temporal_window'],
            augment_data=True
        )
        
        # Create dataloader
        dataloader = DataLoader(
            dataset,
            batch_size=32,  # Smaller batch size for behavior model
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        
        # Training loop
        for epoch in range(self.num_epochs):
            self.behavior_model_trainer.epoch = epoch
            
            # Train epoch
            losses = self.behavior_model_trainer.train_epoch(dataloader)
            
            # Log epoch results
            print(f"Behavior Model Epoch {epoch}: {losses}")
            
            # Save checkpoint
            if losses['total'] < self.behavior_model_trainer.best_loss:
                self.behavior_model_trainer.best_loss = losses['total']
                checkpoint_path = self.checkpoint_dir / f"behavior_model_best.pth"
                self.behavior_model_trainer.save_checkpoint(checkpoint_path)
        
        print("Behavior model training completed!")
    
    def train_complete_system(self, data_path: str):
        """Train the complete system (world model + behavior model)."""
        
        print("Training complete ironing robot system...")
        
        # Train world model first
        self.train_world_model(data_path)
        
        # Train behavior model
        self.train_behavior_model(data_path)
        
        print("Complete system training finished!")


def main():
    """Main function for training."""
    
    import argparse
    
    parser = argparse.ArgumentParser(description="Train Ironing Robot System")
    parser.add_argument("--config", type=str, default="configs/default.yaml",
                       help="Path to configuration file")
    parser.add_argument("--data_path", type=str, default="./data/expert_demonstrations",
                       help="Path to expert demonstration data")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to use for training")
    parser.add_argument("--mode", type=str, choices=["world_model", "behavior_model", "complete"],
                       default="complete", help="Training mode")
    
    args = parser.parse_args()
    
    # Initialize trainer
    trainer = IroningTrainer(args.config, args.device)
    
    # Train based on mode
    if args.mode == "world_model":
        trainer.train_world_model(args.data_path)
    elif args.mode == "behavior_model":
        trainer.train_behavior_model(args.data_path)
    elif args.mode == "complete":
        trainer.train_complete_system(args.data_path)


if __name__ == "__main__":
    main()
