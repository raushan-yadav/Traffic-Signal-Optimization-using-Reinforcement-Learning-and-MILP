"""
Traffic Signal Environment for RL Agent
Simulates a 4-way intersection with heterogeneous traffic

Created for comparing MILP vs RL approaches to traffic signal optimization
Modeling based on Indian traffic conditions with mixed vehicle types
"""

import numpy as np
import random
from collections import deque

class TrafficEnvironment:
    """
    Traffic Environment for Indian Mixed Traffic Conditions
    
    Features:
    - 4 approaches (North, South, East, West)
    - Mixed vehicle types: Cars, Bikes, Buses, Autos
    - Queue-based vehicle arrival
    - Realistic delay calculations
    """
    
    def __init__(self, config=None):
        # Default configuration
        self.config = config or {
            'max_queue_length': 50,
            'vehicle_types': ['car', 'bike', 'bus', 'auto'],
            'saturation_flows': {  # vehicles per hour per lane
                'car': 1800,
                'bike': 2400,
                'bus': 1200,
                'auto': 2000
            },
            'pcu_values': {  # Passenger Car Unit equivalents
                'car': 1.0,
                'bike': 0.5,
                'bus': 3.0,
                'auto': 1.5
            },
            'max_green_time': 60,  # seconds
            'min_green_time': 10,   # seconds
            'yellow_time': 3,       # seconds
            'all_red_time': 2       # seconds
        }
        
        # State variables
        self.current_phase = 0  # 0: NS green, 1: EW green
        self.time_in_phase = 0
        self.total_time = 0
        
        # Traffic queues for each approach
        self.queues = {
            'north': deque(),
            'south': deque(),
            'east': deque(),
            'west': deque()
        }
        
        # Performance metrics
        self.total_delay = 0
        self.total_vehicles_passed = 0
        self.vehicles_waiting = 0
        
        # Episode tracking
        self.episode_length = 3600  # 1 hour simulation
        
    def reset(self):
        """Reset environment to initial state"""
        self.current_phase = 0
        self.time_in_phase = 0
        self.total_time = 0
        
        self.queues = {
            'north': deque(),
            'south': deque(),
            'east': deque(),
            'west': deque()
        }
        
        self.total_delay = 0
        self.total_vehicles_passed = 0
        self.vehicles_waiting = 0
        
        return self._get_state()
    
    def _get_state(self):
        """
        Get current state representation
        
        State includes:
        - Queue lengths for each approach
        - Current phase
        - Time in current phase
        - Waiting time of first vehicle in each queue
        """
        state = []
        
        # Queue lengths (normalized by max queue length)
        for direction in ['north', 'south', 'east', 'west']:
            queue_length = len(self.queues[direction])
            state.append(queue_length / self.config['max_queue_length'])
        
        # Current phase (one-hot encoded)
        phase_encoding = [1 if i == self.current_phase else 0 for i in range(2)]
        state.extend(phase_encoding)
        
        # Time in phase (normalized)
        state.append(self.time_in_phase / self.config['max_green_time'])
        
        # Waiting time of first vehicle (normalized by 120 seconds)
        for direction in ['north', 'south', 'east', 'west']:
            if len(self.queues[direction]) > 0:
                wait_time = self.queues[direction][0]['wait_time']
                state.append(min(wait_time / 120.0, 1.0))
            else:
                state.append(0.0)
        
        return np.array(state, dtype=np.float32)
    
    def _generate_vehicles(self):
        """
        Generate vehicles based on time-varying arrival rates
        Models peak and off-peak hours
        """
        # Time of day factor (peak hours have higher traffic)
        hour = (self.total_time // 3600) % 24
        if 7 <= hour <= 10 or 17 <= hour <= 20:  # Peak hours
            arrival_multiplier = 1.5
        else:  # Off-peak
            arrival_multiplier = 0.7
        
        # Generate vehicles for each approach
        for direction in ['north', 'south', 'east', 'west']:
            # Random arrival based on Poisson process
            arrival_rate = 0.3 * arrival_multiplier  # vehicles per second
            
            if random.random() < arrival_rate:
                vehicle_type = random.choice(self.config['vehicle_types'])
                vehicle = {
                    'type': vehicle_type,
                    'arrival_time': self.total_time,
                    'wait_time': 0,
                    'pcu': self.config['pcu_values'][vehicle_type]
                }
                self.queues[direction].append(vehicle)
    
    def _discharge_vehicles(self):
        """
        Discharge vehicles based on current green phase
        Models saturation flow and queue discharge
        """
        vehicles_discharged = 0
        
        # Determine which approaches have green
        if self.current_phase == 0:  # North-South green
            green_approaches = ['north', 'south']
        else:  # East-West green
            green_approaches = ['east', 'west']
        
        # Discharge vehicles from green approaches
        for direction in green_approaches:
            if len(self.queues[direction]) > 0:
                # Saturation flow (vehicles per second)
                vehicle = self.queues[direction][0]
                sat_flow = self.config['saturation_flows'][vehicle['type']] / 3600
                
                # Probability of discharge (simplified model)
                discharge_prob = min(sat_flow, 0.9)
                
                if random.random() < discharge_prob:
                    vehicle = self.queues[direction].popleft()
                    
                    # Calculate delay
                    delay = self.total_time - vehicle['arrival_time']
                    self.total_delay += delay
                    self.total_vehicles_passed += 1
                    vehicles_discharged += 1
        
        return vehicles_discharged
    
    def _update_waiting_times(self):
        """Update waiting times for all queued vehicles"""
        for direction in self.queues:
            for vehicle in self.queues[direction]:
                vehicle['wait_time'] += 1
    
    def _calculate_reward(self):
        """
        Calculate reward for RL agent
        
        Reward components:
        - Negative reward for queue lengths (congestion)
        - Negative reward for waiting time
        - Positive reward for throughput
        """
        # Queue length penalty
        total_queue = sum(len(self.queues[d]) for d in self.queues)
        queue_penalty = -total_queue * 0.1
        
        # Waiting time penalty (prioritize long waits)
        wait_penalty = 0
        for direction in self.queues:
            if len(self.queues[direction]) > 0:
                max_wait = max(v['wait_time'] for v in self.queues[direction])
                wait_penalty -= max_wait * 0.05
        
        # Throughput reward
        throughput_reward = self.total_vehicles_passed * 0.5
        
        # Total reward
        reward = queue_penalty + wait_penalty + throughput_reward
        
        return reward
    
    def step(self, action):
        """
        Execute one time step
        
        Actions:
        0 - Continue current phase
        1 - Switch to next phase
        """
        # Store previous metrics for reward calculation
        prev_vehicles_passed = self.total_vehicles_passed
        
        # Execute action
        if action == 1:  # Switch phase
            # Yellow + All-red time
            transition_time = self.config['yellow_time'] + self.config['all_red_time']
            
            # Switch phase
            self.current_phase = 1 - self.current_phase
            self.time_in_phase = 0
            
            # Advance time during transition
            for _ in range(transition_time):
                self._generate_vehicles()
                self._update_waiting_times()
                self.total_time += 1
        
        # Continue current phase for 1 second
        self._generate_vehicles()
        self._discharge_vehicles()
        self._update_waiting_times()
        
        self.time_in_phase += 1
        self.total_time += 1
        
        # Get new state
        state = self._get_state()
        
        # Calculate reward
        reward = self._calculate_reward()
        
        # Check if episode is done
        done = self.total_time >= self.episode_length
        
        # Info dict
        info = {
            'total_delay': self.total_delay,
            'vehicles_passed': self.total_vehicles_passed,
            'avg_queue_length': sum(len(self.queues[d]) for d in self.queues) / 4,
            'current_phase': self.current_phase,
            'time_in_phase': self.time_in_phase
        }
        
        return state, reward, done, info
    
    def get_queue_lengths(self):
        """Get current queue lengths"""
        return {d: len(self.queues[d]) for d in self.queues}
    
    def get_metrics(self):
        """Get performance metrics"""
        avg_delay = self.total_delay / max(self.total_vehicles_passed, 1)
        avg_queue = sum(len(self.queues[d]) for d in self.queues) / 4
        
        return {
            'total_delay': self.total_delay,
            'average_delay': avg_delay,
            'vehicles_passed': self.total_vehicles_passed,
            'average_queue_length': avg_queue,
            'throughput': self.total_vehicles_passed / (self.total_time / 3600)  # vehicles/hour
        }


if __name__ == "__main__":
    # Test the environment
    env = TrafficEnvironment()
    state = env.reset()
    
    print("Initial State Shape:", state.shape)
    print("Initial State:", state)
    
    # Run a few random steps
    for i in range(10):
        action = random.choice([0, 1])
        state, reward, done, info = env.step(action)
        
        print(f"\nStep {i+1}:")
        print(f"  Action: {'Continue' if action == 0 else 'Switch'}")
        print(f"  Reward: {reward:.2f}")
        print(f"  Queue Lengths: {env.get_queue_lengths()}")
        print(f"  Phase: {info['current_phase']}, Time in Phase: {info['time_in_phase']}")
        
        if done:
            break
    
    print("\n" + "="*50)
    print("Final Metrics:")
    metrics = env.get_metrics()
    for key, value in metrics.items():
        print(f"  {key}: {value:.2f}")
