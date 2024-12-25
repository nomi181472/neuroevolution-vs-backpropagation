import csv
import os
from src.evotorch.logging import Logger
from src.evotorch.algorithms import SearchAlgorithm


class CSVLogger(Logger):
    def __init__(self, searcher: SearchAlgorithm,path='data', filename='status_log.csv'):
        super().__init__(searcher)
        os.makedirs(path, exist_ok=True)
        self.filename =os.path.join( path,filename)
        if os.path.exists(filename):
            os.remove(filename)  # Remove existing file if it exists
        self.fieldnames = [
            'iteration', 'pop_best_eval', 'mean_eval', 'median_eval',
            'best_eval', 'worst_eval', 'total_interaction_count', 'total_episode_count'
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
            'total_episode_count': status['total_episode_count']
        }
        print(data)
        print()
        with open(self.filename, 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=self.fieldnames)
            writer.writerow(data)
