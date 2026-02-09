# Setup and Usage Guide

## Installation

### 1. Clone the repository
```bash
git clone https://github.com/raushan-yadav/traffic-signal-rl-optimization.git
cd traffic-signal-rl-optimization
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

**Note:** If you don't have PyTorch installed and want to avoid it, the code will fall back to a simplified Q-table implementation.

## Running the Code

### Quick Demo (No training required)
```bash
python demo.py
```
This runs a simplified comparison showing the difference between fixed-time and adaptive control.

### Full Training and Comparison
```bash
python main.py
```
This will:
1. Train the DQN agent (default 50 episodes, takes ~10-15 minutes)
2. Run MILP optimization 
3. Compare both approaches
4. Generate performance plots
5. Save results to `results/` folder

**For better results:** Edit `main.py` and change episodes to 100-200 (will take longer but agent learns better)

## Understanding the Output

After running `main.py`, you'll see:
- Training progress printed to console
- Final comparison metrics
- Plot saved as `results/comparison_results.png`
- Trained agent saved as `results/dqn_agent.pth`

### Metrics Explained

- **Average Delay**: Time vehicles spend waiting (lower is better)
- **Vehicles Passed**: Throughput (higher is better)  
- **Average Queue Length**: Congestion level (lower is better)

## Customization

### Adjust Traffic Patterns
Edit `environment.py`:
```python
# Line ~112 - Change peak hour traffic
arrival_multiplier = 1.5  # Increase for heavier traffic
```

### Change Signal Constraints
Edit `milp_optimizer.py`:
```python
# Line ~22 - Adjust min/max green times
self.min_green = 10   # seconds
self.max_green = 90   # seconds
```

### Tune RL Parameters
Edit `dqn_agent.py`:
```python
# Line ~49 - Modify hyperparameters
self.learning_rate = 0.001
self.gamma = 0.95  # Discount factor
```

## Troubleshooting

**Issue:** "No module named 'torch'"
- **Solution:** Install PyTorch: `pip install torch` or let it use Q-table fallback

**Issue:** "No module named 'pulp'"  
- **Solution:** `pip install pulp`

**Issue:** Training takes too long
- **Solution:** Reduce episodes in `main.py` or use `demo.py` instead

**Issue:** Plots don't show
- **Solution:** Check `results/` folder, image saved there as PNG

## Project Structure

```
traffic-signal-rl-optimization/
├── environment.py          # Traffic simulation
├── milp_optimizer.py       # MILP baseline  
├── dqn_agent.py           # RL agent
├── main.py                # Main script
├── demo.py                # Quick demo
├── requirements.txt       # Dependencies
├── README.md             # Documentation
├── SETUP.md              # This file
└── results/              # Output folder (created when running)
```

## Expected Results

With default settings (50 episodes):
- RL should show 10-15% improvement over MILP
- Average delay: RL ~30s, MILP ~35-40s

With more training (200+ episodes):
- RL improvement increases to 15-20%
- Agent learns better policies

## Notes

- The environment models typical Indian traffic with mixed vehicle types
- MILP provides optimal *fixed-time* solution
- RL learns *adaptive* control that responds to real-time conditions
- Results may vary slightly due to randomness in traffic generation and RL training
