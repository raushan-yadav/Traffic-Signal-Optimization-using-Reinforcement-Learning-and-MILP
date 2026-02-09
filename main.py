"""
Main Training and Comparison Script
Trains DQN agent and compares with MILP baseline
"""

import numpy as np
import matplotlib.pyplot as plt
from environment import TrafficEnvironment
from dqn_agent import DQNAgent, train_dqn_agent, evaluate_agent
from milp_optimizer import MILPSignalOptimizer
import json
import os

class TrafficSignalComparison:
    """
    Compare RL-based adaptive control vs MILP-based fixed-time control
    """
    
    def __init__(self, output_dir='results'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        self.env = TrafficEnvironment()
        self.state_size = 11  # From environment state representation
        self.action_size = 2  # Continue or Switch
        
        self.results = {
            'rl': {},
            'milp': {},
            'comparison': {}
        }
    
    def train_rl_agent(self, episodes=100):
        """Train DQN agent"""
        print("\n" + "="*70)
        print("TRAINING REINFORCEMENT LEARNING AGENT")
        print("="*70 + "\n")
        
        # Create agent
        agent = DQNAgent(self.state_size, self.action_size)
        
        # Train
        training_history = train_dqn_agent(
            self.env, agent, episodes=episodes, verbose=True
        )
        
        # Save agent
        agent.save(os.path.join(self.output_dir, 'dqn_agent.pth'))
        
        # Evaluate
        print("\n" + "-"*70)
        print("EVALUATING TRAINED AGENT")
        print("-"*70 + "\n")
        
        eval_metrics = evaluate_agent(self.env, agent, episodes=10)
        
        print("Evaluation Results (10 episodes, no exploration):")
        print(f"  Average Delay:    {eval_metrics['avg_delay']:.2f} ± {eval_metrics['std_delay']:.2f} seconds")
        print(f"  Vehicles Passed:  {eval_metrics['avg_vehicles']:.0f}")
        print(f"  Average Queue:    {eval_metrics['avg_queue']:.2f} vehicles")
        
        self.results['rl'] = {
            'training_history': training_history,
            'evaluation': eval_metrics,
            'agent': agent
        }
        
        return training_history, eval_metrics
    
    def optimize_milp_baseline(self):
        """Run MILP optimization"""
        print("\n" + "="*70)
        print("MILP BASELINE OPTIMIZATION")
        print("="*70 + "\n")
        
        # Get average traffic demand from environment simulation
        # Run environment for one episode to get traffic patterns
        self.env.reset()
        traffic_counts = {'north': 0, 'south': 0, 'east': 0, 'west': 0}
        
        for _ in range(3600):  # 1 hour
            action = 0  # Arbitrary action for traffic generation
            self.env.step(action)
        
        # Extract average flows (vehicles per hour)
        # This is a simplified estimation
        traffic_data = {
            'north': 800,  # Typical peak hour values
            'south': 750,
            'east': 900,
            'west': 850
        }
        
        print("Assumed Traffic Demand (vehicles/hour):")
        for direction, flow in traffic_data.items():
            print(f"  {direction.capitalize()}: {flow}")
        
        # Optimize
        optimizer = MILPSignalOptimizer(traffic_data)
        solution = optimizer.optimize_green_splits()
        metrics = optimizer.calculate_performance_metrics(solution)
        
        optimizer.print_solution(solution, metrics)
        
        # Simulate MILP solution in environment
        print("\n" + "-"*70)
        print("SIMULATING MILP SOLUTION IN ENVIRONMENT")
        print("-"*70 + "\n")
        
        milp_simulation_results = self._simulate_fixed_time_control(solution)
        
        print("MILP Simulation Results:")
        print(f"  Average Delay:    {milp_simulation_results['avg_delay']:.2f} seconds")
        print(f"  Vehicles Passed:  {milp_simulation_results['vehicles_passed']:.0f}")
        print(f"  Average Queue:    {milp_simulation_results['avg_queue']:.2f} vehicles")
        
        self.results['milp'] = {
            'solution': solution,
            'theoretical_metrics': metrics,
            'simulation_results': milp_simulation_results
        }
        
        return solution, milp_simulation_results
    
    def _simulate_fixed_time_control(self, milp_solution):
        """
        Simulate fixed-time signal control in environment
        Uses timing from MILP solution
        """
        cycle_time = milp_solution['cycle_time']
        green_ns = milp_solution['green_ns']
        green_ew = milp_solution['green_ew']
        
        # Run simulation
        state = self.env.reset()
        done = False
        time_in_cycle = 0
        current_phase = 0  # 0: NS, 1: EW
        
        while not done:
            # Fixed-time control logic
            if current_phase == 0:  # NS green
                if time_in_cycle >= green_ns:
                    action = 1  # Switch
                    current_phase = 1
                    time_in_cycle = 0
                else:
                    action = 0  # Continue
                    time_in_cycle += 1
            else:  # EW green
                if time_in_cycle >= green_ew:
                    action = 1  # Switch
                    current_phase = 0
                    time_in_cycle = 0
                else:
                    action = 0  # Continue
                    time_in_cycle += 1
            
            state, _, done, _ = self.env.step(action)
        
        metrics = self.env.get_metrics()
        return metrics
    
    def compare_methods(self):
        """Generate comparison between RL and MILP"""
        print("\n" + "="*70)
        print("COMPARISON: RL vs MILP")
        print("="*70 + "\n")
        
        rl_eval = self.results['rl']['evaluation']
        milp_sim = self.results['milp']['simulation_results']
        
        comparison = {
            'delay_improvement': ((milp_sim['average_delay'] - rl_eval['avg_delay']) / 
                                 milp_sim['average_delay'] * 100),
            'vehicles_improvement': ((rl_eval['avg_vehicles'] - milp_sim['vehicles_passed']) / 
                                    milp_sim['vehicles_passed'] * 100),
            'queue_improvement': ((milp_sim['average_queue_length'] - rl_eval['avg_queue']) / 
                                 milp_sim['average_queue_length'] * 100)
        }
        
        print("Performance Comparison:")
        print(f"\n{'Metric':<25} {'RL':<15} {'MILP':<15} {'Improvement':<15}")
        print("-" * 70)
        print(f"{'Avg Delay (sec)':<25} {rl_eval['avg_delay']:<15.2f} "
              f"{milp_sim['average_delay']:<15.2f} {comparison['delay_improvement']:<15.2f}%")
        print(f"{'Vehicles Passed':<25} {rl_eval['avg_vehicles']:<15.0f} "
              f"{milp_sim['vehicles_passed']:<15.0f} {comparison['vehicles_improvement']:<15.2f}%")
        print(f"{'Avg Queue Length':<25} {rl_eval['avg_queue']:<15.2f} "
              f"{milp_sim['average_queue_length']:<15.2f} {comparison['queue_improvement']:<15.2f}%")
        
        self.results['comparison'] = comparison
        
        return comparison
    
    def plot_results(self):
        """Generate visualization plots"""
        print("\n" + "="*70)
        print("GENERATING VISUALIZATION")
        print("="*70 + "\n")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot 1: Training curve (delay)
        history = self.results['rl']['training_history']
        axes[0, 0].plot(history['episode'], history['avg_delay'], 'b-', alpha=0.7)
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Average Delay (seconds)')
        axes[0, 0].set_title('RL Agent Learning Curve: Delay')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Training curve (reward)
        axes[0, 1].plot(history['episode'], history['total_reward'], 'g-', alpha=0.7)
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Total Reward')
        axes[0, 1].set_title('RL Agent Learning Curve: Reward')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Comparison bar chart
        metrics = ['Avg Delay\n(seconds)', 'Vehicles\nPassed', 'Avg Queue\nLength']
        rl_values = [
            self.results['rl']['evaluation']['avg_delay'],
            self.results['rl']['evaluation']['avg_vehicles'],
            self.results['rl']['evaluation']['avg_queue']
        ]
        milp_values = [
            self.results['milp']['simulation_results']['average_delay'],
            self.results['milp']['simulation_results']['vehicles_passed'],
            self.results['milp']['simulation_results']['average_queue_length']
        ]
        
        # Normalize for visualization
        rl_norm = [v / max(rl_values[i], milp_values[i]) for i, v in enumerate(rl_values)]
        milp_norm = [v / max(rl_values[i], milp_values[i]) for i, v in enumerate(milp_values)]
        
        x = np.arange(len(metrics))
        width = 0.35
        
        axes[1, 0].bar(x - width/2, rl_norm, width, label='RL', color='blue', alpha=0.7)
        axes[1, 0].bar(x + width/2, milp_norm, width, label='MILP', color='orange', alpha=0.7)
        axes[1, 0].set_ylabel('Normalized Value')
        axes[1, 0].set_title('RL vs MILP Performance Comparison')
        axes[1, 0].set_xticks(x)
        axes[1, 0].set_xticklabels(metrics)
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3, axis='y')
        
        # Plot 4: Epsilon decay
        axes[1, 1].plot(history['episode'], history['epsilon'], 'r-', alpha=0.7)
        axes[1, 1].set_xlabel('Episode')
        axes[1, 1].set_ylabel('Epsilon (Exploration Rate)')
        axes[1, 1].set_title('Exploration vs Exploitation')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'comparison_results.png'), dpi=300)
        print(f"Plot saved to: {os.path.join(self.output_dir, 'comparison_results.png')}")
        
        return fig
    
    def save_results(self):
        """Save all results to JSON"""
        # Prepare serializable results
        save_data = {
            'rl': {
                'evaluation': self.results['rl']['evaluation'],
                'final_epsilon': self.results['rl']['training_history']['epsilon'][-1]
            },
            'milp': {
                'solution': self.results['milp']['solution'],
                'simulation_results': self.results['milp']['simulation_results']
            },
            'comparison': self.results['comparison']
        }
        
        filepath = os.path.join(self.output_dir, 'results.json')
        with open(filepath, 'w') as f:
            json.dump(save_data, f, indent=2)
        
        print(f"\nResults saved to: {filepath}")


def main():
    """Main execution"""
    print("\n" + "#"*70)
    print("#" + " "*68 + "#")
    print("#  TRAFFIC SIGNAL OPTIMIZATION: RL vs MILP COMPARISON  ".center(70, " ")[1:-1] + "#")
    print("#" + " "*68 + "#")
    print("#"*70 + "\n")
    
    # Create comparison object
    comparison = TrafficSignalComparison(output_dir='results')
    
    # Train RL agent
    comparison.train_rl_agent(episodes=50)  # Use 100+ for better results
    
    # Run MILP optimization
    comparison.optimize_milp_baseline()
    
    # Compare results
    comparison.compare_methods()
    
    # Generate plots
    comparison.plot_results()
    
    # Save results
    comparison.save_results()
    
    print("\n" + "#"*70)
    print("#  EXPERIMENT COMPLETE  ".center(70, " ")[1:-1] + "#")
    print("#"*70 + "\n")


if __name__ == "__main__":
    main()
