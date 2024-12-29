import csv
import os
from src.evotorch.logging import Logger
from src.evotorch.algorithms import SearchAlgorithm


class CSVLogger(Logger):
    def __init__(self, searcher: SearchAlgorithm, path='data', filename='status_log.csv'):
        super().__init__(searcher)
        os.makedirs(path, exist_ok=True)
        self.filename = os.path.join(path, filename)
        if os.path.exists(self.filename):
            os.remove(self.filename)  # Remove existing file if it exists
        self.fieldnames = [
            'iteration', 'pop_best_eval', 'mean_eval', 'median_eval',
            'best_eval', 'worst_eval', 'total_interaction_count',
            'total_episode_count', 'ensembled_reward','best_current_reward','pop_best_current_reward',
            'ensembled_steps', 'best_current_steps', 'pop_best_current_steps',
            'elapsed_time'
        ]
        with open(self.filename, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=self.fieldnames)
            writer.writeheader()

    def _log(self, status: dict):
        data = {
            'iteration': status['iter'],
            'pop_best_eval': status['pop_best_eval'],
            'mean_eval': status['mean_eval'],
            'median_eval': status['median_eval'],
            'best_eval': status['best_eval'],
            'worst_eval': status['worst_eval'],
            'total_interaction_count': status['total_interaction_count'],
            'total_episode_count': status['total_episode_count'],

            'ensembled_reward': status.get('ensembled_reward', 0),
            'ensembled_steps': status.get('ensembled_steps', 0),

            'pop_best_current_reward': status.get('pop_best_current_reward', 0),
            'pop_best_current_steps': status.get('pop_best_current_steps', 0),

            'best_current_reward': status.get('best_current_reward', 0),
            'best_current_steps': status.get('best_current_steps', 0),

            'elapsed_time': status.get('elapsed_time', 0),




        }
        with open(self.filename, 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=self.fieldnames)
            writer.writerow(data)
