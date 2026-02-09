# Traffic Signal Optimization: Comparing RL and MILP Approaches

A comparative study of traffic signal control methods for Indian mixed traffic conditions.

## Overview

This project compares two different approaches for optimizing traffic signals at a 4-way intersection:
- Traditional optimization using Mixed Integer Linear Programming (MILP)
- Adaptive control using Deep Q-Learning (Reinforcement Learning)

The motivation is that most traffic signals in India use fixed timing, which doesn't adapt to real-time traffic conditions. This project explores whether RL-based adaptive control can improve performance.

## Problem Statement

Traffic signals in urban India face unique challenges:
- Heterogeneous traffic (cars, bikes, buses, autos moving together)
- High density and less lane discipline
- Variable traffic patterns (peak hours, accidents, events)

**Question:** Can adaptive RL-based control outperform traditional fixed-time optimization?

## Approach

### 1. Traffic Simulation Environment

Built a custom simulator modeling:
- 4-way intersection (North, South, East, West)
- Different vehicle types with PCU (Passenger Car Unit) values
- Time-varying traffic arrival patterns
- Queue formation and discharge dynamics

### 2. MILP Baseline

Uses Webster's formula and linear programming to find optimal fixed signal timing:
- Calculates optimal cycle time based on traffic flows
- Determines green splits for each phase
- Minimizes expected delay

### 3. Deep Q-Learning Agent

Trains a neural network to learn adaptive signal control:
- Observes queue lengths, current phase, waiting times
- Decides when to continue or switch signal phase
- Learns from experience using reward feedback
- Uses techniques like experience replay and target networks

## Implementation

**Environment:**
- State: Queue lengths (4), phase info (2), time in phase (1), waiting times (4) = 11 dimensions
- Actions: Continue current phase (0) or Switch phase (1)
- Reward: Penalty for queues and waiting, reward for throughput

**MILP Optimizer:**
- Webster's optimal cycle calculation
- Constraint-based optimization (PuLP library)
- Capacity and timing constraints

**DQN Agent:**
- Neural network: 11 inputs → 64 → 64 → 2 outputs
- Experience replay buffer
- Epsilon-greedy exploration
- Target network for stable learning

## Results

Preliminary testing shows:
- RL achieves lower average delays compared to fixed-time
- Adapts better to varying traffic conditions
- Improvement around 15-20% in average delay per vehicle

The RL agent learns to extend green phases when traffic is heavy and switch earlier when traffic is light, which fixed-time control can't do.

## Usage

### Requirements
```bash
pip install numpy matplotlib pulp torch
```

### Quick Demo
```bash
python demo.py
```
Runs a simplified comparison between fixed-time and adaptive control.

### Full Training
```bash
python main.py
```
Trains the DQN agent and compares with MILP baseline. Generates plots and metrics.

## Files

- `environment.py` - Traffic simulation environment
- `milp_optimizer.py` - MILP-based optimization
- `dqn_agent.py` - Deep Q-Learning agent
- `main.py` - Training and comparison script
- `demo.py` - Quick demonstration

## Key Learnings

**Traffic Engineering:**
- Signal timing theory (Webster's method)
- Flow-density-speed relationships
- Indian traffic characteristics

**Optimization:**
- MILP formulation for real problems
- Constraint modeling
- Solver usage

**Reinforcement Learning:**
- DQN algorithm implementation
- Experience replay importance
- Reward function design challenges
- Balancing exploration vs exploitation

## Future Work

- Extend to multiple intersections (network optimization)
- Integrate real traffic data from sensors
- Add V2I communication simulation
- Test other RL algorithms (PPO, A3C)
- Implement bus priority mechanisms

## References

- Webster, F.V. (1958). Traffic Signal Settings
- Mnih et al. (2015). Human-level control through deep reinforcement learning
- Highway Capacity Manual (HCM) - delay formulas
- IRC guidelines for Indian traffic conditions

## Contact

Raushan Yadav  
Research Intern - Operations Research, IIT Kharagpur  
Email: raushanyadav1001@gmail.com  
[LinkedIn](https://linkedin.com/in/raushanyadav) | [GitHub](https://github.com/raushan-yadav)

---

*Developed as part of research on intelligent transportation systems and traffic optimization.*
