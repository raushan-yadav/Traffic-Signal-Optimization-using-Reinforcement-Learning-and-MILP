"""
Deep Q-Network (DQN) Agent for Adaptive Traffic Signal Control
Uses neural network to learn optimal signal switching policy

Implementation follows Mnih et al. (2015) DQN paper
Adapted for traffic signal control problem
"""

import numpy as np
import random
from collections import deque
import pickle

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("PyTorch not available. Using simplified Q-learning.")


class DQNetwork(nn.Module):
    """Deep Q-Network architecture"""
    
    def __init__(self, state_size, action_size, hidden_size=64):
        super(DQNetwork, self).__init__()
        
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)
        
    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


class DQNAgent:
    """
    Deep Q-Learning Agent for Traffic Signal Control
    
    Learns to minimize delay by choosing optimal actions:
    - Action 0: Continue current phase
    - Action 1: Switch to next phase
    """
    
    def __init__(self, state_size, action_size, use_pytorch=True):
        self.state_size = state_size
        self.action_size = action_size
        self.use_pytorch = use_pytorch and TORCH_AVAILABLE
        
        # Hyperparameters
        self.gamma = 0.95          # Discount factor
        self.epsilon = 1.0         # Exploration rate
        self.epsilon_min = 0.01    # Minimum exploration
        self.epsilon_decay = 0.995 # Exploration decay
        self.learning_rate = 0.001
        self.batch_size = 64
        self.memory_size = 10000
        
        # Experience replay memory
        self.memory = deque(maxlen=self.memory_size)
        
        # Q-Network
        if self.use_pytorch:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.q_network = DQNetwork(state_size, action_size).to(self.device)
            self.target_network = DQNetwork(state_size, action_size).to(self.device)
            self.target_network.load_state_dict(self.q_network.state_dict())
            self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.learning_rate)
            self.criterion = nn.MSELoss()
        else:
            # Fallback: Simple Q-table (for demonstration)
            self.q_table = {}
        
        # Training tracking
        self.training_step = 0
        self.update_target_every = 100
        
    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay memory"""
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state, training=True):
        """
        Choose action using epsilon-greedy policy
        
        Args:
            state: current state
            training: if True, use exploration; if False, use exploitation only
        """
        if training and random.random() < self.epsilon:
            # Explore: random action
            return random.randrange(self.action_size)
        
        # Exploit: best action based on Q-values
        if self.use_pytorch:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.q_network(state_tensor)
                return q_values.argmax().item()
        else:
            # Simplified Q-table lookup
            state_key = self._state_to_key(state)
            if state_key not in self.q_table:
                self.q_table[state_key] = np.zeros(self.action_size)
            return np.argmax(self.q_table[state_key])
    
    def replay(self):
        """
        Train on batch of experiences from memory
        Implements experience replay for stable learning
        """
        if len(self.memory) < self.batch_size:
            return 0.0
        
        # Sample random batch
        batch = random.sample(self.memory, self.batch_size)
        
        if self.use_pytorch:
            return self._replay_pytorch(batch)
        else:
            return self._replay_qtable(batch)
    
    def _replay_pytorch(self, batch):
        """PyTorch implementation of experience replay"""
        states = torch.FloatTensor([t[0] for t in batch]).to(self.device)
        actions = torch.LongTensor([t[1] for t in batch]).to(self.device)
        rewards = torch.FloatTensor([t[2] for t in batch]).to(self.device)
        next_states = torch.FloatTensor([t[3] for t in batch]).to(self.device)
        dones = torch.FloatTensor([t[4] for t in batch]).to(self.device)
        
        # Current Q-values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Target Q-values (using target network)
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        # Compute loss
        loss = self.criterion(current_q_values, target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Update target network periodically
        self.training_step += 1
        if self.training_step % self.update_target_every == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        
        return loss.item()
    
    def _replay_qtable(self, batch):
        """Simplified Q-table update"""
        total_error = 0.0
        
        for state, action, reward, next_state, done in batch:
            state_key = self._state_to_key(state)
            next_state_key = self._state_to_key(next_state)
            
            if state_key not in self.q_table:
                self.q_table[state_key] = np.zeros(self.action_size)
            if next_state_key not in self.q_table:
                self.q_table[next_state_key] = np.zeros(self.action_size)
            
            # Q-learning update
            target = reward
            if not done:
                target += self.gamma * np.max(self.q_table[next_state_key])
            
            error = target - self.q_table[state_key][action]
            self.q_table[state_key][action] += self.learning_rate * error
            total_error += abs(error)
        
        return total_error / len(batch)
    
    def _state_to_key(self, state):
        """Convert state array to hashable key for Q-table"""
        # Discretize continuous state
        discrete_state = tuple(np.round(state, 2))
        return discrete_state
    
    def decay_epsilon(self):
        """Decay exploration rate"""
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def save(self, filepath):
        """Save agent"""
        if self.use_pytorch:
            torch.save({
                'q_network': self.q_network.state_dict(),
                'target_network': self.target_network.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'epsilon': self.epsilon,
                'training_step': self.training_step
            }, filepath)
        else:
            with open(filepath, 'wb') as f:
                pickle.dump({
                    'q_table': self.q_table,
                    'epsilon': self.epsilon
                }, f)
        print(f"Agent saved to {filepath}")
    
    def load(self, filepath):
        """Load agent"""
        if self.use_pytorch:
            checkpoint = torch.load(filepath, map_location=self.device)
            self.q_network.load_state_dict(checkpoint['q_network'])
            self.target_network.load_state_dict(checkpoint['target_network'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.epsilon = checkpoint['epsilon']
            self.training_step = checkpoint['training_step']
        else:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
                self.q_table = data['q_table']
                self.epsilon = data['epsilon']
        print(f"Agent loaded from {filepath}")


def train_dqn_agent(env, agent, episodes=100, verbose=True):
    """
    Train DQN agent on traffic environment
    
    Args:
        env: TrafficEnvironment instance
        agent: DQNAgent instance
        episodes: number of training episodes
        verbose: print progress
    
    Returns:
        training_history: dict with metrics per episode
    """
    training_history = {
        'episode': [],
        'total_reward': [],
        'avg_delay': [],
        'vehicles_passed': [],
        'avg_queue': [],
        'epsilon': [],
        'loss': []
    }
    
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        losses = []
        
        done = False
        while not done:
            # Select action
            action = agent.act(state)
            
            # Execute action
            next_state, reward, done, info = env.step(action)
            
            # Store experience
            agent.remember(state, action, reward, next_state, done)
            
            # Train agent
            if len(agent.memory) > agent.batch_size:
                loss = agent.replay()
                losses.append(loss)
            
            state = next_state
            total_reward += reward
        
        # Decay exploration
        agent.decay_epsilon()
        
        # Get metrics
        metrics = env.get_metrics()
        
        # Store history
        training_history['episode'].append(episode + 1)
        training_history['total_reward'].append(total_reward)
        training_history['avg_delay'].append(metrics['average_delay'])
        training_history['vehicles_passed'].append(metrics['vehicles_passed'])
        training_history['avg_queue'].append(metrics['average_queue_length'])
        training_history['epsilon'].append(agent.epsilon)
        training_history['loss'].append(np.mean(losses) if losses else 0)
        
        # Print progress
        if verbose and (episode + 1) % 10 == 0:
            print(f"Episode {episode + 1}/{episodes}")
            print(f"  Reward: {total_reward:.2f}")
            print(f"  Avg Delay: {metrics['average_delay']:.2f}s")
            print(f"  Vehicles: {metrics['vehicles_passed']}")
            print(f"  Avg Queue: {metrics['average_queue_length']:.2f}")
            print(f"  Epsilon: {agent.epsilon:.3f}")
            print(f"  Loss: {np.mean(losses) if losses else 0:.4f}")
            print()
    
    return training_history


def evaluate_agent(env, agent, episodes=10):
    """
    Evaluate trained agent (no exploration)
    
    Returns:
        evaluation_metrics: dict with average performance
    """
    total_delays = []
    total_vehicles = []
    total_queues = []
    
    for episode in range(episodes):
        state = env.reset()
        done = False
        
        while not done:
            action = agent.act(state, training=False)  # No exploration
            state, _, done, _ = env.step(action)
        
        metrics = env.get_metrics()
        total_delays.append(metrics['average_delay'])
        total_vehicles.append(metrics['vehicles_passed'])
        total_queues.append(metrics['average_queue_length'])
    
    evaluation_metrics = {
        'avg_delay': np.mean(total_delays),
        'std_delay': np.std(total_delays),
        'avg_vehicles': np.mean(total_vehicles),
        'avg_queue': np.mean(total_queues)
    }
    
    return evaluation_metrics


if __name__ == "__main__":
    # This will be imported by main training script
    print("DQN Agent module loaded successfully")
    print(f"PyTorch available: {TORCH_AVAILABLE}")
    if TORCH_AVAILABLE:
        print(f"Device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")
