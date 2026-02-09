"""
MILP-based Traffic Signal Optimization
Uses Webster's Method with MILP for optimal fixed-time signal timing

Based on classic traffic engineering theory adapted for Indian conditions
Serves as baseline for comparison with RL approach
"""

from pulp import *
import numpy as np

class MILPSignalOptimizer:
    """
    Mixed Integer Linear Programming approach for signal timing
    Based on Webster's method with extensions for Indian traffic
    """
    
    def __init__(self, traffic_data):
        """
        Initialize optimizer with traffic data
        
        Args:
            traffic_data: dict with flow rates for each approach
                {
                    'north': flow_rate (veh/hr),
                    'south': flow_rate,
                    'east': flow_rate,
                    'west': flow_rate
                }
        """
        self.traffic_data = traffic_data
        self.saturation_flow = 1800  # vehicles/hour/lane (typical for Indian conditions)
        self.lost_time_per_phase = 4  # seconds (startup + clearance)
        self.min_green = 10  # seconds
        self.max_green = 90  # seconds
        self.yellow_time = 3  # seconds
        self.all_red_time = 2  # seconds
        
    def calculate_flow_ratios(self):
        """
        Calculate critical flow ratios for each phase
        
        Flow ratio (y) = demand flow / saturation flow
        """
        flow_ratios = {}
        
        # Phase 1: North-South
        ns_flow = max(self.traffic_data['north'], self.traffic_data['south'])
        flow_ratios['ns'] = ns_flow / self.saturation_flow
        
        # Phase 2: East-West
        ew_flow = max(self.traffic_data['east'], self.traffic_data['west'])
        flow_ratios['ew'] = ew_flow / self.saturation_flow
        
        return flow_ratios
    
    def webster_optimal_cycle(self):
        """
        Calculate optimal cycle time using Webster's formula
        
        C = (1.5L + 5) / (1 - Y)
        where:
            L = total lost time per cycle
            Y = sum of critical flow ratios
        """
        flow_ratios = self.calculate_flow_ratios()
        
        # Total lost time
        num_phases = 2
        L = num_phases * self.lost_time_per_phase
        
        # Sum of critical flow ratios
        Y = sum(flow_ratios.values())
        
        # Webster's formula
        if Y >= 0.95:  # Oversaturated condition
            print("Warning: Traffic is oversaturated (Y >= 0.95)")
            Y = 0.95
        
        C_optimal = (1.5 * L + 5) / (1 - Y)
        
        # Practical bounds
        C_optimal = max(60, min(C_optimal, 180))
        
        return C_optimal, flow_ratios
    
    def optimize_green_splits(self):
        """
        Use MILP to optimize green time splits
        
        Objective: Minimize total delay
        Decision variables: green times for each phase
        Constraints: cycle time, min/max green, flow balance
        """
        # Calculate optimal cycle time
        C_optimal, flow_ratios = self.webster_optimal_cycle()
        
        print(f"\nWebster's Optimal Cycle Time: {C_optimal:.1f} seconds")
        print(f"Flow Ratios: NS={flow_ratios['ns']:.3f}, EW={flow_ratios['ew']:.3f}")
        
        # Create optimization problem
        prob = LpProblem("Traffic_Signal_Timing", LpMinimize)
        
        # Decision variables: green times
        g_ns = LpVariable("green_NS", lowBound=self.min_green, upBound=self.max_green)
        g_ew = LpVariable("green_EW", lowBound=self.min_green, upBound=self.max_green)
        
        # Cycle time variable
        C = LpVariable("cycle_time", lowBound=60, upBound=180)
        
        # Auxiliary variables for delay calculation
        # Delay proportional to (C * (1 - g/C)^2) / (1 - y*g/C)
        # Simplified linear approximation for MILP
        
        # Objective: Minimize weighted delay
        # Using approximation: delay ~ queue_length ~ flow * (C - g)
        ns_delay_approx = self.traffic_data['north'] * (C - g_ns) + \
                          self.traffic_data['south'] * (C - g_ns)
        ew_delay_approx = self.traffic_data['east'] * (C - g_ew) + \
                          self.traffic_data['west'] * (C - g_ew)
        
        prob += ns_delay_approx + ew_delay_approx, "Total_Delay"
        
        # Constraint 1: Cycle time composition
        total_lost_time = 2 * self.lost_time_per_phase
        prob += g_ns + g_ew + total_lost_time == C, "Cycle_Time_Balance"
        
        # Constraint 2: Capacity constraint (flow ratio)
        # Effective green / Cycle >= flow ratio
        prob += g_ns >= flow_ratios['ns'] * C * 1.1, "NS_Capacity"  # 1.1 safety factor
        prob += g_ew >= flow_ratios['ew'] * C * 1.1, "EW_Capacity"
        
        # Constraint 3: Target cycle time (soft constraint via bounds)
        prob += C >= C_optimal * 0.9, "Min_Cycle"
        prob += C <= C_optimal * 1.1, "Max_Cycle"
        
        # Solve
        prob.solve(PULP_CBC_CMD(msg=0))
        
        # Extract solution
        solution = {
            'status': LpStatus[prob.status],
            'cycle_time': value(C),
            'green_ns': value(g_ns),
            'green_ew': value(g_ew),
            'yellow_time': self.yellow_time,
            'all_red_time': self.all_red_time,
            'red_ns': value(C) - value(g_ns) - self.yellow_time - self.all_red_time,
            'red_ew': value(C) - value(g_ew) - self.yellow_time - self.all_red_time
        }
        
        return solution
    
    def calculate_performance_metrics(self, solution):
        """
        Calculate expected performance metrics for the solution
        
        Uses HCM (Highway Capacity Manual) delay formulas
        """
        C = solution['cycle_time']
        g_ns = solution['green_ns']
        g_ew = solution['green_ew']
        
        flow_ratios = self.calculate_flow_ratios()
        
        # Calculate degree of saturation (X)
        X_ns = flow_ratios['ns'] * C / g_ns
        X_ew = flow_ratios['ew'] * C / g_ew
        
        # Webster's delay formula (per vehicle)
        # d = C(1-g/C)^2 / (2(1-X))
        
        def webster_delay(C, g, X):
            if X >= 1.0:
                return float('inf')  # Oversaturated
            return (C * (1 - g/C)**2) / (2 * (1 - X))
        
        delay_ns = webster_delay(C, g_ns, X_ns)
        delay_ew = webster_delay(C, g_ew, X_ew)
        
        # Weighted average delay
        total_flow = sum(self.traffic_data.values())
        ns_flow = self.traffic_data['north'] + self.traffic_data['south']
        ew_flow = self.traffic_data['east'] + self.traffic_data['west']
        
        avg_delay = (delay_ns * ns_flow + delay_ew * ew_flow) / total_flow
        
        metrics = {
            'avg_delay_per_vehicle': avg_delay,
            'delay_ns': delay_ns,
            'delay_ew': delay_ew,
            'saturation_ns': X_ns,
            'saturation_ew': X_ew,
            'level_of_service': self._get_los(avg_delay)
        }
        
        return metrics
    
    def _get_los(self, delay):
        """
        Determine Level of Service based on delay
        
        LOS ranges (HCM):
        A: 0-10s, B: 10-20s, C: 20-35s, D: 35-55s, E: 55-80s, F: >80s
        """
        if delay <= 10:
            return 'A'
        elif delay <= 20:
            return 'B'
        elif delay <= 35:
            return 'C'
        elif delay <= 55:
            return 'D'
        elif delay <= 80:
            return 'E'
        else:
            return 'F'
    
    def print_solution(self, solution, metrics):
        """Pretty print the optimization solution"""
        print("\n" + "="*60)
        print("MILP OPTIMIZATION RESULTS")
        print("="*60)
        
        print(f"\nOptimization Status: {solution['status']}")
        
        print(f"\n--- Signal Timing Plan ---")
        print(f"Cycle Time:        {solution['cycle_time']:.1f} seconds")
        print(f"\nNorth-South Phase:")
        print(f"  Green Time:      {solution['green_ns']:.1f} seconds")
        print(f"  Yellow Time:     {solution['yellow_time']} seconds")
        print(f"  All-Red Time:    {solution['all_red_time']} seconds")
        print(f"  Red Time:        {solution['red_ns']:.1f} seconds")
        
        print(f"\nEast-West Phase:")
        print(f"  Green Time:      {solution['green_ew']:.1f} seconds")
        print(f"  Yellow Time:     {solution['yellow_time']} seconds")
        print(f"  All-Red Time:    {solution['all_red_time']} seconds")
        print(f"  Red Time:        {solution['red_ew']:.1f} seconds")
        
        print(f"\n--- Performance Metrics ---")
        print(f"Average Delay:     {metrics['avg_delay_per_vehicle']:.2f} seconds/vehicle")
        print(f"NS Phase Delay:    {metrics['delay_ns']:.2f} seconds/vehicle")
        print(f"EW Phase Delay:    {metrics['delay_ew']:.2f} seconds/vehicle")
        print(f"NS Saturation:     {metrics['saturation_ns']:.3f}")
        print(f"EW Saturation:     {metrics['saturation_ew']:.3f}")
        print(f"Level of Service:  {metrics['level_of_service']}")
        
        print("="*60 + "\n")


def main():
    """Example usage"""
    
    # Example traffic data (vehicles per hour)
    # Peak hour scenario
    traffic_data = {
        'north': 800,
        'south': 750,
        'east': 900,
        'west': 850
    }
    
    print("Traffic Demand (vehicles/hour):")
    for direction, flow in traffic_data.items():
        print(f"  {direction.capitalize()}: {flow}")
    
    # Create optimizer
    optimizer = MILPSignalOptimizer(traffic_data)
    
    # Optimize signal timing
    solution = optimizer.optimize_green_splits()
    
    # Calculate metrics
    metrics = optimizer.calculate_performance_metrics(solution)
    
    # Print results
    optimizer.print_solution(solution, metrics)
    
    # Test with different scenarios
    print("\n" + "#"*60)
    print("TESTING DIFFERENT TRAFFIC SCENARIOS")
    print("#"*60)
    
    scenarios = {
        'Off-Peak': {
            'north': 400, 'south': 350,
            'east': 450, 'west': 420
        },
        'Unbalanced (NS Heavy)': {
            'north': 1200, 'south': 1100,
            'east': 500, 'west': 480
        },
        'Evening Rush': {
            'north': 1000, 'south': 950,
            'east': 1100, 'west': 1050
        }
    }
    
    for scenario_name, scenario_data in scenarios.items():
        print(f"\n{'='*60}")
        print(f"Scenario: {scenario_name}")
        print(f"{'='*60}")
        
        optimizer = MILPSignalOptimizer(scenario_data)
        solution = optimizer.optimize_green_splits()
        metrics = optimizer.calculate_performance_metrics(solution)
        
        print(f"Cycle: {solution['cycle_time']:.0f}s | "
              f"NS Green: {solution['green_ns']:.0f}s | "
              f"EW Green: {solution['green_ew']:.0f}s | "
              f"Avg Delay: {metrics['avg_delay_per_vehicle']:.1f}s | "
              f"LOS: {metrics['level_of_service']}")


if __name__ == "__main__":
    main()
