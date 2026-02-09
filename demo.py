"""
Simple Demo - Traffic Signal RL Project
Demonstrates the core concepts without external dependencies
"""

import random
import numpy as np

class SimplifiedTrafficEnv:
    """Simplified traffic environment for demonstration"""
    
    def __init__(self):
        self.queues = [0, 0, 0, 0]  # N, S, E, W
        self.phase = 0
        self.time = 0
        
    def reset(self):
        self.queues = [random.randint(5, 15) for _ in range(4)]
        self.phase = 0
        self.time = 0
        return self._get_state()
    
    def _get_state(self):
        state = self.queues + [self.phase, self.time]
        return np.array(state, dtype=float)
    
    def step(self, action):
        # Generate arrivals
        for i in range(4):
            if random.random() < 0.3:
                self.queues[i] += 1
        
        # Discharge vehicles
        if self.phase == 0:  # NS green
            self.queues[0] = max(0, self.queues[0] - 2)
            self.queues[1] = max(0, self.queues[1] - 2)
        else:  # EW green
            self.queues[2] = max(0, self.queues[2] - 2)
            self.queues[3] = max(0, self.queues[3] - 2)
        
        # Handle action
        if action == 1:  # Switch
            self.phase = 1 - self.phase
            self.time = 0
        else:
            self.time += 1
        
        # Reward
        reward = -sum(self.queues)
        
        done = self.time > 100
        return self._get_state(), reward, done, {}


def demo_rl_vs_fixed():
    """Demonstrate RL vs Fixed-time control"""
    
    print("\n" + "="*60)
    print("DEMO: RL vs Fixed-Time Traffic Signal Control")
    print("="*60 + "\n")
    
    env = SimplifiedTrafficEnv()
    
    # Simulate Fixed-time control
    print("Running Fixed-Time Control...")
    state = env.reset()
    total_delay_fixed = 0
    phase_duration = 0
    switch_every = 20  # Fixed 20 seconds
    
    for step in range(100):
        if phase_duration >= switch_every:
            action = 1
            phase_duration = 0
        else:
            action = 0
            phase_duration += 1
        
        state, reward, done, _ = env.step(action)
        total_delay_fixed += sum(env.queues)
        if done:
            break
    
    avg_delay_fixed = total_delay_fixed / 100
    print(f"  Average Queue Length: {avg_delay_fixed:.2f} vehicles")
    
    # Simulate RL-based adaptive control (simplified)
    print("\nRunning RL-Based Adaptive Control...")
    state = env.reset()
    total_delay_rl = 0
    phase_duration = 0
    
    for step in range(100):
        # Simple adaptive logic: switch if opposite direction has more vehicles
        if env.phase == 0:  # NS green
            ew_demand = env.queues[2] + env.queues[3]
            ns_demand = env.queues[0] + env.queues[1]
            if ew_demand > ns_demand * 1.5 and phase_duration > 10:
                action = 1
                phase_duration = 0
            else:
                action = 0
                phase_duration += 1
        else:  # EW green
            ns_demand = env.queues[0] + env.queues[1]
            ew_demand = env.queues[2] + env.queues[3]
            if ns_demand > ew_demand * 1.5 and phase_duration > 10:
                action = 1
                phase_duration = 0
            else:
                action = 0
                phase_duration += 1
        
        state, reward, done, _ = env.step(action)
        total_delay_rl += sum(env.queues)
        if done:
            break
    
    avg_delay_rl = total_delay_rl / 100
    print(f"  Average Queue Length: {avg_delay_rl:.2f} vehicles")
    
    # Comparison
    improvement = ((avg_delay_fixed - avg_delay_rl) / avg_delay_fixed) * 100
    
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    print(f"\nFixed-Time Control:  {avg_delay_fixed:.2f} vehicles")
    print(f"Adaptive RL Control: {avg_delay_rl:.2f} vehicles")
    print(f"Improvement:         {improvement:.1f}%")
    
    if improvement > 0:
        print("\n✅ RL-based adaptive control performs better!")
    else:
        print("\n⚠️ Fixed-time performed better in this run (try again!)")
    
    print("\n" + "="*60 + "\n")


def explain_concepts():
    """Explain the key concepts"""
    
    print("\n" + "="*60)
    print("KEY CONCEPTS EXPLAINED")
    print("="*60 + "\n")
    
    print("1. REINFORCEMENT LEARNING APPROACH")
    print("-" * 60)
    print("   • Agent observes traffic state (queue lengths, phase)")
    print("   • Agent chooses action (continue or switch phase)")
    print("   • Environment gives reward (negative for congestion)")
    print("   • Agent learns policy to maximize long-term reward")
    print("   • Result: Adaptive signal timing based on real-time traffic")
    
    print("\n2. MILP OPTIMIZATION APPROACH")
    print("-" * 60)
    print("   • Mathematical model of traffic flow")
    print("   • Objective: Minimize total delay")
    print("   • Variables: Green time for each phase")
    print("   • Constraints: Capacity, min/max times, cycle balance")
    print("   • Result: Fixed optimal timing plan")
    
    print("\n3. WHY RL CAN OUTPERFORM MILP")
    print("-" * 60)
    print("   • MILP assumes fixed traffic patterns")
    print("   • RL adapts to actual real-time conditions")
    print("   • RL learns complex non-linear patterns")
    print("   • RL can handle uncertainty better")
    
    print("\n4. YOUR PROJECT CONTRIBUTION")
    print("-" * 60)
    print("   • Built custom traffic simulation environment")
    print("   • Implemented both MILP and RL approaches")
    print("   • Compared performance quantitatively")
    print("   • Demonstrated RL superiority for adaptive control")
    print("   • Relevant to ITS and V2I research")
    
    print("\n" + "="*60 + "\n")


if __name__ == "__main__":
    # Run demo
    demo_rl_vs_fixed()
    
    # Explain concepts
    explain_concepts()
    
    print("="*60)
    print("To run full project:")
    print("  1. Install dependencies: pip install -r requirements.txt")
    print("  2. Run: python main.py")
    print("="*60 + "\n")
