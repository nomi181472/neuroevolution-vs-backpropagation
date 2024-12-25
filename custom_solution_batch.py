from src.evotorch.core import SolutionBatch
import torch

class CustomSolutionBatch(SolutionBatch):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Initialize a dictionary to track fitness history for each solution
        self.fitness_history = {i: [] for i in range(self._popsize)}

    def update_fitness_history(self):
        """
        Updates the fitness history for each solution in the batch.
        """
        for idx, solution in enumerate(self):
            fitness = solution.get_fitness()
            self.fitness_history[idx].append(fitness)

    def compute_convergence_rate(self):
        """
        Computes the convergence rate based on fitness improvements.
        Returns a dictionary of convergence rates for each solution.
        """
        convergence_rates = {}
        for idx, history in self.fitness_history.items():
            if len(history) > 1:
                # Compute convergence rate between the last two generations
                convergence_rate = (history[-1] - history[-2]) / (history[-2] + 1e-8)  # Avoid division by zero
                convergence_rates[idx] = convergence_rate
            else:
                convergence_rates[idx] = 0.0  # No rate if there's only one fitness value
        return convergence_rates

    def log_metrics(self):
        """
        Logs fitness history and convergence rates.
        """
        convergence_rates = self.compute_convergence_rate()
        for idx, rate in convergence_rates.items():
            print(f"Solution {idx}: Fitness History = {self.fitness_history[idx]}, Convergence Rate = {rate}")
