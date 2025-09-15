"""
Behavior Model for Ironing Robot System

This module implements a behavior cloning model that learns to coordinate
multiple agents (dual robotic arms + linear actuator) for ironing tasks.

The behavior model serves as the robot's "decision-making brain" - it takes
visual and proprioceptive input and outputs coordinated actions for all agents.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
import math


class MultiAgentAttention(nn.Module):
    """
    Attention mechanism for coordinating multiple agents.
    Allows agents to attend to each other's states and actions.
    """
    
    def __init__(self, 
                 feature_dim: int = 256,
                 num_agents: int = 3,
                 num_heads: int = 8):
        super().__init__()
        
        self.feature_dim = feature_dim
        self.num_agents = num_agents
        self.num_heads = num_heads
        self.head_dim = feature_dim // num_heads
        
        # Linear projections for Q, K, V
        self.q_linear = nn.Linear(feature_dim, feature_dim)
        self.k_linear = nn.Linear(feature_dim, feature_dim)
        self.v_linear = nn.Linear(feature_dim, feature_dim)
        
        # Output projection
        self.out_linear = nn.Linear(feature_dim, feature_dim)
        
        # Agent-specific projections
        self.agent_projections = nn.ModuleList([
            nn.Linear(feature_dim, feature_dim) for _ in range(num_agents)
        ])
        
    def forward(self, agent_features: torch.Tensor) -> torch.Tensor:
        """
        Apply multi-agent attention.
        
        Args:
            agent_features: [batch, num_agents, feature_dim]
            
        Returns:
            attended_features: [batch, num_agents, feature_dim]
        """
        batch_size, num_agents, feature_dim = agent_features.shape
        
        # Project to Q, K, V
        Q = self.q_linear(agent_features)  # [batch, num_agents, feature_dim]
        K = self.k_linear(agent_features)  # [batch, num_agents, feature_dim]
        V = self.v_linear(agent_features)  # [batch, num_agents, feature_dim]
        
        # Reshape for multi-head attention
        Q = Q.view(batch_size, num_agents, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, num_agents, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, num_agents, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attention_weights = F.softmax(scores, dim=-1)
        
        # Apply attention to values
        attended = torch.matmul(attention_weights, V)
        
        # Reshape back
        attended = attended.transpose(1, 2).contiguous().view(
            batch_size, num_agents, feature_dim
        )
        
        # Apply output projection
        output = self.out_linear(attended)
        
        # Add residual connection
        output = output + agent_features
        
        return output


class TemporalEncoder(nn.Module):
    """
    Encodes temporal sequences of observations for behavior cloning.
    Uses LSTM to capture temporal dependencies.
    """
    
    def __init__(self, 
                 input_dim: int,
                 hidden_dim: int = 256,
                 num_layers: int = 2,
                 temporal_window: int = 5):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.temporal_window = temporal_window
        
        # LSTM for temporal encoding
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.1 if num_layers > 1 else 0
        )
        
        # Output projection
        self.output_projection = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(self, sequences: torch.Tensor) -> torch.Tensor:
        """
        Encode temporal sequences.
        
        Args:
            sequences: [batch, temporal_window, input_dim]
            
        Returns:
            encoded: [batch, hidden_dim]
        """
        # LSTM forward pass
        lstm_out, (hidden, cell) = self.lstm(sequences)
        
        # Use the last hidden state
        last_hidden = hidden[-1]  # [batch, hidden_dim]
        
        # Apply output projection
        encoded = self.output_projection(last_hidden)
        
        return encoded


class ActionSmoother(nn.Module):
    """
    Smooths actions to avoid jerky movements.
    Implements exponential smoothing with learnable parameters.
    """
    
    def __init__(self, 
                 action_dim: int,
                 smoothing_factor: float = 0.1):
        super().__init__()
        
        self.action_dim = action_dim
        self.smoothing_factor = smoothing_factor
        
        # Learnable smoothing parameters
        self.alpha = nn.Parameter(torch.tensor(smoothing_factor))
        
    def forward(self, 
                current_actions: torch.Tensor,
                previous_actions: torch.Tensor) -> torch.Tensor:
        """
        Smooth actions using exponential smoothing.
        
        Args:
            current_actions: [batch, action_dim] - raw actions from policy
            previous_actions: [batch, action_dim] - previous smoothed actions
            
        Returns:
            smoothed_actions: [batch, action_dim] - smoothed actions
        """
        # Clamp alpha to valid range
        alpha = torch.clamp(self.alpha, 0.0, 1.0)
        
        # Exponential smoothing: smoothed = alpha * current + (1 - alpha) * previous
        smoothed_actions = alpha * current_actions + (1 - alpha) * previous_actions
        
        return smoothed_actions


class BehaviorCloningModel(nn.Module):
    """
    Main behavior cloning model for the ironing robot system.
    Learns to coordinate multiple agents from expert demonstrations.
    """
    
    def __init__(self, config: Dict):
        super().__init__()
        
        # Extract configuration
        self.input_dim = config.get('input_dim', 512)
        self.action_dim = config.get('action_dim', 15)
        self.hidden_sizes = config.get('hidden_sizes', [512, 512, 256])
        self.num_agents = config.get('num_agents', 3)
        self.agent_action_dims = config.get('agent_action_dims', [7, 7, 1])
        self.temporal_window = config.get('behavior_cloning', {}).get('temporal_window', 5)
        
        # Visual encoder (shared with world model)
        visual_config = config.get('visual_encoder', {})
        self.visual_encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((8, 8)),
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, visual_config.get('feature_dim', 256))
        )
        
        # Proprioceptive encoder
        proprioceptive_dim = 14 + 14 + 2  # left_arm + right_arm + actuator
        self.proprioceptive_encoder = nn.Sequential(
            nn.Linear(proprioceptive_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128)
        )
        
        # Combine visual and proprioceptive features
        combined_dim = visual_config.get('feature_dim', 256) + 128
        self.feature_combiner = nn.Sequential(
            nn.Linear(combined_dim, self.hidden_sizes[0]),
            nn.ReLU(),
            nn.Linear(self.hidden_sizes[0], self.hidden_sizes[0])
        )
        
        # Temporal encoder
        self.temporal_encoder = TemporalEncoder(
            input_dim=self.hidden_sizes[0],
            hidden_dim=self.hidden_sizes[0],
            temporal_window=self.temporal_window
        )
        
        # Multi-agent attention
        self.attention = MultiAgentAttention(
            feature_dim=self.hidden_sizes[0],
            num_agents=self.num_agents,
            num_heads=config.get('attention_heads', 8)
        )
        
        # Agent-specific policy heads
        self.agent_policies = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.hidden_sizes[0], self.hidden_sizes[1]),
                nn.ReLU(),
                nn.Linear(self.hidden_sizes[1], self.hidden_sizes[2]),
                nn.ReLU(),
                nn.Linear(self.hidden_sizes[2], action_dim)
            ) for action_dim in self.agent_action_dims
        ])
        
        # Action smoother
        self.action_smoother = ActionSmoother(
            action_dim=self.action_dim,
            smoothing_factor=config.get('behavior_cloning', {}).get('smoothing_factor', 0.1)
        )
        
        # Safety limits
        self.safety_limits = {
            'joint_limits': torch.tensor([
                [-3.14, 3.14] for _ in range(7)  # Left arm
            ] + [
                [-3.14, 3.14] for _ in range(7)  # Right arm
            ] + [
                [0.0, 0.2]  # Linear actuator
            ]),
            'max_velocity': 0.5,  # rad/s for joints, m/s for actuator
            'max_acceleration': 1.0  # rad/s² for joints, m/s² for actuator
        }
        
    def forward(self, 
                images: torch.Tensor,
                proprioceptive_state: torch.Tensor,
                previous_actions: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the behavior cloning model.
        
        Args:
            images: [batch, temporal_window, 3, height, width] - camera images
            proprioceptive_state: [batch, temporal_window, 30] - robot states
            previous_actions: [batch, action_dim] - previous actions for smoothing
            
        Returns:
            actions: Dict containing actions for each agent
        """
        batch_size, temporal_window = images.shape[:2]
        
        # Process temporal sequences
        visual_features = []
        proprioceptive_features = []
        
        for t in range(temporal_window):
            # Visual encoding
            visual_feat = self.visual_encoder(images[:, t])
            visual_features.append(visual_feat)
            
            # Proprioceptive encoding
            proprio_feat = self.proprioceptive_encoder(proprioceptive_state[:, t])
            proprioceptive_features.append(proprio_feat)
        
        # Stack temporal features
        visual_features = torch.stack(visual_features, dim=1)  # [batch, temporal_window, visual_dim]
        proprioceptive_features = torch.stack(proprioceptive_features, dim=1)  # [batch, temporal_window, 128]
        
        # Combine features for each timestep
        combined_features = []
        for t in range(temporal_window):
            combined = torch.cat([visual_features[:, t], proprioceptive_features[:, t]], dim=1)
            combined = self.feature_combiner(combined)
            combined_features.append(combined)
        
        # Stack and encode temporally
        combined_sequence = torch.stack(combined_features, dim=1)  # [batch, temporal_window, hidden_dim]
        temporal_features = self.temporal_encoder(combined_sequence)  # [batch, hidden_dim]
        
        # Create agent features (same for all agents in this simple version)
        agent_features = temporal_features.unsqueeze(1).expand(-1, self.num_agents, -1)
        
        # Apply multi-agent attention
        attended_features = self.attention(agent_features)  # [batch, num_agents, hidden_dim]
        
        # Generate actions for each agent
        raw_actions = []
        for i, policy in enumerate(self.agent_policies):
            agent_actions = policy(attended_features[:, i])  # [batch, action_dim_i]
            raw_actions.append(agent_actions)
        
        # Concatenate all agent actions
        raw_combined_actions = torch.cat(raw_actions, dim=1)  # [batch, total_action_dim]
        
        # Apply action smoothing if previous actions provided
        if previous_actions is not None:
            smoothed_actions = self.action_smoother(raw_combined_actions, previous_actions)
        else:
            smoothed_actions = raw_combined_actions
        
        # Apply safety limits
        safe_actions = self.apply_safety_limits(smoothed_actions)
        
        # Split actions back to individual agents
        actions = {}
        start_idx = 0
        for i, action_dim in enumerate(self.agent_action_dims):
            end_idx = start_idx + action_dim
            actions[f'agent_{i}'] = safe_actions[:, start_idx:end_idx]
            start_idx = end_idx
        
        # Add combined actions
        actions['combined'] = safe_actions
        
        return actions
    
    def apply_safety_limits(self, actions: torch.Tensor) -> torch.Tensor:
        """
        Apply safety limits to actions.
        """
        # Clamp to joint limits
        joint_limits = self.safety_limits['joint_limits'].to(actions.device)
        actions = torch.clamp(actions, joint_limits[:, 0], joint_limits[:, 1])
        
        # Apply velocity limits (simplified - would need velocity calculation in practice)
        max_vel = self.safety_limits['max_velocity']
        actions = torch.clamp(actions, -max_vel, max_vel)
        
        return actions
    
    def compute_behavior_cloning_loss(self, 
                                    predicted_actions: torch.Tensor,
                                    expert_actions: torch.Tensor,
                                    weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute behavior cloning loss.
        
        Args:
            predicted_actions: [batch, action_dim] - predicted actions
            expert_actions: [batch, action_dim] - expert actions
            weights: [batch] - optional sample weights
            
        Returns:
            loss: scalar tensor
        """
        # Mean squared error loss
        mse_loss = F.mse_loss(predicted_actions, expert_actions, reduction='none')
        
        # Apply sample weights if provided
        if weights is not None:
            mse_loss = mse_loss.mean(dim=1) * weights
            loss = mse_loss.mean()
        else:
            loss = mse_loss.mean()
        
        return loss
    
    def compute_consistency_loss(self, 
                               actions1: torch.Tensor,
                               actions2: torch.Tensor) -> torch.Tensor:
        """
        Compute consistency loss between similar states.
        Encourages smooth policy behavior.
        """
        return F.mse_loss(actions1, actions2)
    
    def get_action_entropy(self, actions: torch.Tensor) -> torch.Tensor:
        """
        Compute action entropy for regularization.
        """
        # Convert actions to probabilities (simplified)
        action_probs = F.softmax(actions, dim=-1)
        log_probs = F.log_softmax(actions, dim=-1)
        entropy = -(action_probs * log_probs).sum(dim=-1).mean()
        return entropy


class IroningBehaviorModel(nn.Module):
    """
    Complete behavior model for the ironing robot system.
    Combines behavior cloning with multi-agent coordination.
    """
    
    def __init__(self, config: Dict):
        super().__init__()
        
        self.config = config
        self.behavior_model = BehaviorCloningModel(config)
        
        # Training configuration
        self.learning_rate = config.get('learning_rate', 0.0001)
        self.weight_decay = config.get('weight_decay', 1e-4)
        
        # Loss weights
        self.loss_weights = {
            'behavior_cloning': 1.0,
            'consistency': 0.1,
            'entropy': 0.01
        }
        
    def forward(self, 
                images: torch.Tensor,
                proprioceptive_state: torch.Tensor,
                previous_actions: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the complete behavior model.
        """
        return self.behavior_model(images, proprioceptive_state, previous_actions)
    
    def compute_total_loss(self, 
                          predictions: Dict[str, torch.Tensor],
                          targets: Dict[str, torch.Tensor],
                          sample_weights: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Compute total training loss.
        """
        losses = {}
        
        # Behavior cloning loss
        if 'combined' in predictions and 'expert_actions' in targets:
            losses['behavior_cloning'] = self.behavior_model.compute_behavior_cloning_loss(
                predictions['combined'],
                targets['expert_actions'],
                sample_weights
            )
        
        # Consistency loss (if multiple predictions available)
        if 'combined' in predictions and 'consistency_target' in targets:
            losses['consistency'] = self.behavior_model.compute_consistency_loss(
                predictions['combined'],
                targets['consistency_target']
            )
        
        # Entropy regularization
        if 'combined' in predictions:
            losses['entropy'] = self.behavior_model.get_action_entropy(
                predictions['combined']
            )
        
        # Weighted total loss
        total_loss = sum(
            self.loss_weights.get(key, 1.0) * loss 
            for key, loss in losses.items()
        )
        losses['total'] = total_loss
        
        return losses
    
    def get_optimizer(self) -> torch.optim.Optimizer:
        """
        Get optimizer for training.
        """
        return torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
