import os
import copy
import gymnasium as gym
import torch
import time
from torch import nn
import random
import numpy as np
from src.evotorch.algorithms import GeneticAlgorithm
from need_ga import NeedGA
from src.evotorch.logging import StdOutLogger
from src.evotorch.neuroevolution import GymNE
from evotorch.decorators import pass_info
from custom_logger import CSVLogger
from src.evotorch.operators import GaussianMutation
from custom_operators import GreedyCrossover, MultiPointCrossOver
from src.evotorch.operators import TwoPointCrossOver
from ensemble_model import Ensemble

# Utility functions
class Utils:
    @staticmethod
    def set_seed(seed):
        """Set the random seed for reproducibility."""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    @staticmethod
    def create_directory(path):
        """Create a directory if it does not exist."""
        if not os.path.exists(path):
            os.makedirs(path)

# Neural Network Policy
@pass_info
class LinearPolicy(nn.Module):
    def __init__(self, obs_length, act_length, num_layers=2, neurons=8, bias=True, **kwargs):
        super().__init__()

        layers = []
        input_size = obs_length

        if num_layers == 1:
            layers.append(nn.Linear(input_size, act_length, bias=bias))
        else:
            for _ in range(num_layers - 1):
                layers.append(nn.Linear(input_size, neurons, bias=bias))
                layers.append(nn.ReLU())
                input_size = neurons

            layers.append(nn.Linear(input_size, act_length, bias=bias))

        self.layers = nn.ModuleList(layers)

    def forward(self, obs):
        x = obs
        for layer in self.layers:
            x = layer(x)
        return x

# RL Trainer
class RLTrainer:
    def __init__(self, env_name, save_weights_after_iter=100, seed=4):
        self.env_name = env_name
        self.save_weights_after_iter = save_weights_after_iter
        self.seed = seed
        Utils.set_seed(self.seed)

        self.metrics = {
            "best_eval": float('-inf'),
            "worst_eval": float('inf'),
            "iter": 1,
            "median_eval": float('-inf'),
            "pop_best_eval": float('-inf'),
            "mean_eval": float('-inf'),
        }

        self.folder_name = f"./data/{env_name}"
        self.weights_path = f"{self.folder_name}/weights"
        self.current_iteration = 0
        self.current_performer = {}
        self.repertoire = {}

        self._initialize_directories()
        self.problem = self._create_problem(env_name)
        self.searcher = self._create_searcher()
        self._setup_hooks()

        self.logger = StdOutLogger(self.searcher)
        self.csv_logger = CSVLogger(self.searcher, self.folder_name)

    def _initialize_directories(self):
        Utils.create_directory("data")
        Utils.create_directory(self.folder_name)
        Utils.create_directory(self.weights_path)

    def _create_problem(self, env_name):
        def create_env():
            return gym.make(env_name)

        return GymNE(
            env=create_env,
            network=LinearPolicy,
            network_args={'bias': True},
            num_actors='max',
            observation_normalization=True,
        )

    def _create_searcher(self):
        return NeedGA(
            self.problem,
            operators=[
                GreedyCrossover(problem=self.problem,top_n=3,num_children=40,),
                GaussianMutation(problem=self.problem, stdev=0.1, mutation_probability=0.1)
            ],
            popsize=100,
            elitist=True,
            re_evaluate=True,
            re_evaluate_parents_first=True,
            _allow_empty_operator_list=True,
        )

    def _setup_hooks(self):
        #self.searcher.before_step_hook.append(self.check_metrics)
        #self.searcher.after_step_hook.append(self.run_current_top_models)
        self.searcher.before_step_hook.append(self.before_epoch_hook)
        self.searcher.after_step_hook.append(self.after_epoch_hook)

    def before_epoch_hook(self):
        """Hook to execute before each epoch starts."""
        self.epoch_start_time = time.time()  # Start timer

    def after_epoch_hook(self)->{}:
        """Hook to execute after each epoch ends."""
        try:
            time_eval={}
            self.epoch_end_time = time.time()  # End timer
            elapsed_time = self.epoch_end_time - self.epoch_start_time
            time_eval['elapsed_time'] = elapsed_time  # Log elapsed time
            print(f"Epoch {self.current_iteration}: Elapsed time: {elapsed_time:.2f} seconds")

            # Call the existing hooks
            self.check_metrics()
            #evals=self.run_current_top_models()
            self.calculate_qd_metrics()
            return {**time_eval}

        except Exception as e:
            print(f"Error in after_epoch_hook: {e}")
        return {}

    def check_metrics(self):
        try:
            current_status = copy.deepcopy(self.searcher.status)
            if "iter" not in current_status:
                return

            self.current_iteration = current_status['iter']
            if self.current_iteration < self.save_weights_after_iter:
                return

            for metric_name, sol_name in [
                ("pop_best_eval", "pop_best"),
                ("best_eval", "best"),
                ("median_eval", "pop_best"),
                ("mean_eval", "pop_best"),
            ]:
                self._save_weights_based_on_metrics(metric_name, sol_name, current_status)

        except Exception as e:
            print(f"Error in check_metrics: {e}")

    def _save_weights_based_on_metrics(self, metric_name, sol_name, current_status):
        if metric_name in current_status and current_status[metric_name] > self.metrics[metric_name]:
            current_score = current_status[metric_name]
            solution = current_status[sol_name]
            current_policy = self.problem.parameterize_net(solution.access_values())

            self.current_performer[sol_name] = copy.deepcopy(current_policy)

            path = f"{self.weights_path}/{metric_name}"
            Utils.create_directory(path)
            file_name = f"{path}/iter_{current_status['iter']}_score_{current_score}.pth"
            torch.save(current_policy.state_dict(), file_name)
            self.metrics[metric_name] = current_score
            print(f"Saved weights to {file_name}")

    def calculate_qd_metrics(self):
        """Calculate Quality Diversity (QD) metrics."""
        fitness_values = [entry['fitness'] for entry in self.repertoire.values() if entry['fitness'] > -np.inf]


        coverage = len(fitness_values)               # Number of filled niches
        qd_score = sum(fitness_values)               # Sum of fitness values across all niches

        print(f"QD Metrics - , Coverage: {coverage}, QD Score: {qd_score}")
        #self.logger.(f"QD Metrics , Coverage: {coverage}, QD Score: {qd_score}")

    @torch.no_grad()
    def evaluate_and_record(self, policy, save_path, iteration):
        iteration_path = os.path.join(f"{save_path}", f"video_{iteration}")
        Utils.create_directory(iteration_path)

        env = gym.make(self.env_name, render_mode="rgb_array")
        env = gym.wrappers.RecordVideo(env, iteration_path, episode_trigger=lambda _: True)

        total_rewards = []
        total_steps = []

        for seed in range(10):
            obs, _ = env.reset(seed=seed)
            obs = torch.from_numpy(obs)
            total_reward = 0
            steps = 0
            done = False

            while not done and steps <= 1700:
                action = torch.argmax(policy(obs)).item()
                obs, reward, done, _, _ = env.step(action)
                obs = torch.from_numpy(obs)
                total_reward += reward
                steps += 1

            total_rewards.append(total_reward)
            total_steps.append(steps)

        env.close()
        avg_reward = sum(total_rewards) / len(total_rewards)
        avg_steps = sum(total_steps) / len(total_steps)
        return avg_reward, avg_steps

    def run_current_top_models(self)->dict:
        if self.current_iteration <= self.save_weights_after_iter:
            return {}

        try:
            evals = {}

            if "best" in self.current_performer:
                best = self.current_performer["best"]
                evals['best_current_reward'], evals['best_current_steps'] = self.evaluate_and_record(best, f"{self.folder_name}/best_records", self.current_iteration)

            if "pop_best" in self.current_performer:
                pop_best = self.current_performer["pop_best"]
                evals['pop_best_current_reward'], evals['pop_best_current_steps'] = self.evaluate_and_record(pop_best, f"{self.folder_name}/pop_best_current", self.current_iteration)

            models = self.searcher.ensembled_models
            #
            if len(models) > 0:
                ensemble_policy = Ensemble(models)
                evals['ensemble_reward'], evals['ensemble_steps'] = self.evaluate_and_record(ensemble_policy, f"{self.folder_name}/ensemble_records", self.current_iteration)

            return evals

        except Exception as e:
            print(f"Error in run_current_top_models: {e}")
        return {}

    def train(self, num_iterations):
        print("Starting training...")
        self.searcher.run(num_iterations)
        best_solution = self.searcher.status["best"]
        final_policy = self.problem.to_policy(best_solution)
        return final_policy

# Main execution
if __name__ == "__main__":
    trainer = RLTrainer(env_name="LunarLander-v3", save_weights_after_iter=400)
    final_policy = trainer.train(num_iterations=700)
    print("Training completed.")
