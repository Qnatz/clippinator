import numpy as np # Make sure numpy is listed as a dependency if not already

class PerformanceMonitor:
    def __init__(self):
        self.metrics = {
            'total_tasks': 0,
            'successful_tasks': 0,
            # 'avg_response_time': [], # Placeholder, as time logging is not in the snippet
            'steps_per_task': [],
            'error_rate': 0.0 # Store as float
        }
    
    def log_execution(self, result: dict, success: bool = True): # Added success parameter
        self.metrics['total_tasks'] += 1
        if success:
            self.metrics['successful_tasks'] += 1

        if 'intermediate_steps' in result and result['intermediate_steps'] is not None:
            self.metrics['steps_per_task'].append(len(result['intermediate_steps']))
        else:
            # Handle cases where there are no intermediate steps (e.g., direct output or error before steps)
            self.metrics['steps_per_task'].append(0) 
        
        # Update error rate
        if self.metrics['total_tasks'] > 0:
            self.metrics['error_rate'] = ((self.metrics['total_tasks'] - self.metrics['successful_tasks']) / self.metrics['total_tasks']) * 100
        else:
            self.metrics['error_rate'] = 0.0

    def generate_report(self):
        print("\n=== PERFORMANCE REPORT ===")
        print(f"Total Tasks Executed: {self.metrics['total_tasks']}")
        print(f"Successful Tasks: {self.metrics['successful_tasks']}")
        if self.metrics['steps_per_task']: # Check if list is not empty
            print(f"Average Steps per Task: {np.mean(self.metrics['steps_per_task']):.2f}")
        else:
            print("Average Steps per Task: N/A (No tasks with steps logged)")
        print(f"Error Rate: {self.metrics['error_rate']:.2f}%")
        # Potentially return the report as a string or dict for programmatic access
