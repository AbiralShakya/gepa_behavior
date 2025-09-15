"""
World Model for Ironing Robot System

This module implements a world model that learns to predict the dynamics of:
1. Dual robotic arms (left and right)
2. Linear actuator with heating pad
3. Fabric deformation and physics
4. Multi-agent coordination

The world model serves as the robot's "imagination" - it can predict what will
happen in the world without actually executing actions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
import math


class FabricPhysicsModel(nn.Module):
    """
    Neural network model for predicting fabric deformation and physics.
    Uses a mesh-based approach to model cloth dynamics.
    """
    
    def __init__(self, mesh_resolution: int = 32, hidden_dim: int = 256):
        super().__init__()
        self.mesh_resolution = mesh_resolution
        self.num_vertices = mesh_resolution * mesh_resolution
        
        # Encoder for current fabric state
        self.fabric_encoder = nn.Sequential(
            nn.Linear(self.num_vertices * 3, hidden_dim),  # x, y, z for each vertex
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 128)
        )
        
        # Encoder for external forces (robot contacts)
        self.force_encoder = nn.Sequential(
            nn.Linear(6, hidden_dim // 2),  # 3D force + 3D position
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 64)
        )
        
        # Physics prediction network
        self.physics_net = nn.Sequential(
            nn.Linear(128 + 64, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.num_vertices * 3)  # Next vertex positions
        )
        
    def forward(self, fabric_state: torch.Tensor, contact_forces: torch.Tensor) -> torch.Tensor:
        """
        Predict next fabric state given current state and contact forces.
        
        Args:
            fabric_state: [batch, mesh_resolution, mesh_resolution, 3] - current vertex positions
            contact_forces: [batch, num_contacts, 6] - force + position for each contact
            
        Returns:
            next_fabric_state: [batch, mesh_resolution, mesh_resolution, 3] - predicted positions
        """
        batch_size = fabric_state.shape[0]
        
        # Flatten fabric state
        fabric_flat = fabric_state.view(batch_size, -1)
        fabric_features = self.fabric_encoder(fabric_flat)
        
        # Process contact forces
        if contact_forces.shape[1] > 0:  # If there are contacts
            # Average over all contacts
            force_features = self.force_encoder(contact_forces.mean(dim=1))
        else:
            # No contacts - use zero features
            force_features = torch.zeros(batch_size, 64, device=fabric_state.device)
        
        # Combine features and predict
        combined = torch.cat([fabric_features, force_features], dim=1)
        next_vertices = self.physics_net(combined)
        
        # Reshape back to mesh format
        next_fabric_state = next_vertices.view(batch_size, self.mesh_resolution, self.mesh_resolution, 3)
        
        return next_fabric_state


class MultiAgentStateEncoder(nn.Module):
    """
    Encodes the state of multiple agents (robots + actuator) into a unified representation.
    """
    
    def __init__(self, 
                 robot_state_dim: int = 14,  # 7 joints * 2 (pos + vel)
                 actuator_state_dim: int = 2,  # position + velocity
                 hidden_dim: int = 256):
        super().__init__()
        
        # Robot state encoders (shared weights for both arms)
        self.robot_encoder = nn.Sequential(
            nn.Linear(robot_state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2)
        )
        
        # Actuator state encoder
        self.actuator_encoder = nn.Sequential(
            nn.Linear(actuator_state_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 2)
        )
        
        # Multi-agent coordination
        self.coordination_net = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
    def forward(self, 
                left_arm_state: torch.Tensor,
                right_arm_state: torch.Tensor, 
                actuator_state: torch.Tensor) -> torch.Tensor:
        """
        Encode multi-agent state into unified representation.
        
        Args:
            left_arm_state: [batch, robot_state_dim]
            right_arm_state: [batch, robot_state_dim] 
            actuator_state: [batch, actuator_state_dim]
            
        Returns:
            unified_state: [batch, hidden_dim]
        """
        # Encode each agent
        left_features = self.robot_encoder(left_arm_state)
        right_features = self.robot_encoder(right_arm_state)
        actuator_features = self.actuator_encoder(actuator_state)
        
        # Combine all agent features
        combined = torch.cat([
            left_features + right_features,  # Sum robot features
            actuator_features
        ], dim=1)
        
        # Apply coordination network
        unified_state = self.coordination_net(combined)
        
        return unified_state


class VisualEncoder(nn.Module):
    """
    Encodes visual observations (camera images) into feature vectors.
    Uses a CNN backbone for spatial feature extraction.
    """
    
    def __init__(self, 
                 input_channels: int = 3,
                 feature_dim: int = 256,
                 spatial_resolution: Tuple[int, int] = (64, 64)):
        super().__init__()
        
        # Simple CNN encoder
        self.conv_layers = nn.Sequential(
            # First conv block
            nn.Conv2d(input_channels, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # Second conv block
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # Third conv block
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # Fourth conv block
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4))
        )
        
        # Calculate flattened size
        self.flattened_size = 256 * 4 * 4
        
        # Project to desired feature dimension
        self.projection = nn.Sequential(
            nn.Linear(self.flattened_size, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim)
        )
        
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Encode images to feature vectors.
        
        Args:
            images: [batch, channels, height, width]
            
        Returns:
            features: [batch, feature_dim]
        """
        # Extract spatial features
        conv_features = self.conv_layers(images)
        
        # Flatten and project
        flattened = conv_features.view(conv_features.size(0), -1)
        features = self.projection(flattened)
        
        return features


class WorldModelTransformer(nn.Module):
    """
    Main world model using Transformer architecture for sequential prediction.
    Predicts next state given current state and actions.
    """
    
    def __init__(self,
                 state_dim: int = 512,
                 action_dim: int = 15,
                 hidden_dim: int = 512,
                 num_layers: int = 6,
                 num_heads: int = 8,
                 sequence_length: int = 10):
        super().__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.sequence_length = sequence_length
        
        # Input projections
        self.state_projection = nn.Linear(state_dim, hidden_dim)
        self.action_projection = nn.Linear(action_dim, hidden_dim)
        
        # Positional encoding
        self.pos_encoding = self._create_positional_encoding(sequence_length, hidden_dim)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # Output heads for different predictions
        self.state_predictor = nn.Linear(hidden_dim, state_dim)
        self.visual_predictor = nn.Linear(hidden_dim, 256)  # Visual features
        self.contact_predictor = nn.Linear(hidden_dim, 6)   # Contact forces
        
    def _create_positional_encoding(self, seq_len: int, d_model: int) -> torch.Tensor:
        """Create sinusoidal positional encoding."""
        pe = torch.zeros(seq_len, d_model)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)
    
    def forward(self, 
                states: torch.Tensor,
                actions: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Predict next states given current states and actions.
        
        Args:
            states: [batch, seq_len, state_dim] - sequence of states
            actions: [batch, seq_len, action_dim] - sequence of actions
            mask: [batch, seq_len] - attention mask (optional)
            
        Returns:
            predictions: Dict containing various predictions
        """
        batch_size, seq_len = states.shape[:2]
        
        # Project inputs
        state_emb = self.state_projection(states)
        action_emb = self.action_projection(actions)
        
        # Combine state and action embeddings
        combined = state_emb + action_emb
        
        # Add positional encoding
        pos_enc = self.pos_encoding[:, :seq_len, :].to(combined.device)
        combined = combined + pos_enc
        
        # Apply transformer
        if mask is not None:
            # Convert mask to attention mask format
            attn_mask = mask.unsqueeze(1).unsqueeze(2)
            attn_mask = attn_mask.expand(batch_size, seq_len, seq_len)
        else:
            attn_mask = None
            
        transformer_out = self.transformer(combined, src_key_padding_mask=mask)
        
        # Generate predictions
        predictions = {
            'next_state': self.state_predictor(transformer_out),
            'visual_features': self.visual_predictor(transformer_out),
            'contact_forces': self.contact_predictor(transformer_out)
        }
        
        return predictions


class IroningWorldModel(nn.Module):
    """
    Complete world model for the ironing robot system.
    Combines fabric physics, multi-agent coordination, and visual processing.
    """
    
    def __init__(self, config: Dict):
        super().__init__()
        
        # Extract configuration
        self.hidden_dim = config.get('hidden_dim', 512)
        self.sequence_length = config.get('rollout_horizon', 10)
        fabric_config = config.get('fabric_physics', {})
        
        # Component models
        self.fabric_physics = FabricPhysicsModel(
            mesh_resolution=fabric_config.get('mesh_resolution', 32),
            hidden_dim=self.hidden_dim
        )
        
        self.state_encoder = MultiAgentStateEncoder(
            robot_state_dim=14,  # 7 joints * 2
            actuator_state_dim=2,
            hidden_dim=self.hidden_dim
        )
        
        self.visual_encoder = VisualEncoder(
            input_channels=3,
            feature_dim=256,
            spatial_resolution=(64, 64)
        )
        
        self.transformer = WorldModelTransformer(
            state_dim=self.hidden_dim + 256,  # State + visual features
            action_dim=15,  # 7 + 7 + 1 actions
            hidden_dim=self.hidden_dim,  # Keep original hidden dim
            sequence_length=self.sequence_length
        )
        
        # Loss weights
        self.loss_weights = config.get('prediction_loss_weights', {
            'state_prediction': 1.0,
            'visual_prediction': 0.5,
            'contact_prediction': 0.3,
            'fabric_deformation': 0.8
        })
        
    def forward(self, 
                robot_states: torch.Tensor,
                actuator_states: torch.Tensor,
                fabric_states: torch.Tensor,
                images: torch.Tensor,
                actions: torch.Tensor,
                contact_forces: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the complete world model.
        
        Args:
            robot_states: [batch, seq_len, 2, 14] - left and right arm states
            actuator_states: [batch, seq_len, 2] - actuator states
            fabric_states: [batch, seq_len, 32, 32, 3] - fabric mesh states
            images: [batch, seq_len, 3, 64, 64] - camera images
            actions: [batch, seq_len, 15] - multi-agent actions
            contact_forces: [batch, seq_len, num_contacts, 6] - contact forces
            
        Returns:
            predictions: Dict containing all predictions
        """
        batch_size, seq_len = robot_states.shape[:2]
        
        # Encode multi-agent states
        left_arm_states = robot_states[:, :, 0, :]  # [batch, seq_len, 14]
        right_arm_states = robot_states[:, :, 1, :]  # [batch, seq_len, 14]
        
        # Process each timestep
        encoded_states = []
        visual_features = []
        fabric_predictions = []
        
        for t in range(seq_len):
            # Encode multi-agent state
            state_enc = self.state_encoder(
                left_arm_states[:, t],
                right_arm_states[:, t],
                actuator_states[:, t]
            )
            encoded_states.append(state_enc)
            
            # Encode visual features
            visual_enc = self.visual_encoder(images[:, t])
            visual_features.append(visual_enc)
            
            # Predict fabric physics
            fabric_pred = self.fabric_physics(
                fabric_states[:, t],
                contact_forces[:, t]
            )
            fabric_predictions.append(fabric_pred)
        
        # Stack sequences
        encoded_states = torch.stack(encoded_states, dim=1)  # [batch, seq_len, hidden_dim]
        visual_features = torch.stack(visual_features, dim=1)  # [batch, seq_len, 256]
        fabric_predictions = torch.stack(fabric_predictions, dim=1)  # [batch, seq_len, 32, 32, 3]
        
        # Combine state and visual features
        # Ensure dimensions match
        if encoded_states.shape[-1] != visual_features.shape[-1]:
            # Project visual features to match state dimension
            visual_projection = nn.Linear(visual_features.shape[-1], encoded_states.shape[-1]).to(visual_features.device)
            visual_features = visual_projection(visual_features)
        
        combined_features = torch.cat([encoded_states, visual_features], dim=-1)
        
        # Apply transformer for sequential prediction
        transformer_predictions = self.transformer(combined_features, actions)
        
        # Add fabric predictions
        transformer_predictions['fabric_deformation'] = fabric_predictions
        
        return transformer_predictions
    
    def compute_loss(self, predictions: Dict[str, torch.Tensor], 
                    targets: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Compute prediction losses for different components.
        """
        losses = {}
        
        # State prediction loss
        if 'next_state' in predictions and 'next_state' in targets:
            losses['state_prediction'] = F.mse_loss(
                predictions['next_state'], targets['next_state']
            )
        
        # Visual prediction loss
        if 'visual_features' in predictions and 'visual_features' in targets:
            losses['visual_prediction'] = F.mse_loss(
                predictions['visual_features'], targets['visual_features']
            )
        
        # Contact prediction loss
        if 'contact_forces' in predictions and 'contact_forces' in targets:
            losses['contact_prediction'] = F.mse_loss(
                predictions['contact_forces'], targets['contact_forces']
            )
        
        # Fabric deformation loss
        if 'fabric_deformation' in predictions and 'fabric_deformation' in targets:
            losses['fabric_deformation'] = F.mse_loss(
                predictions['fabric_deformation'], targets['fabric_deformation']
            )
        
        # Weighted total loss
        total_loss = sum(
            self.loss_weights.get(key, 1.0) * loss 
            for key, loss in losses.items()
        )
        losses['total'] = total_loss
        
        return losses
