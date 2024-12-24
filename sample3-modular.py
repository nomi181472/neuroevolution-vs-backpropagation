import os
import copy
import gymnasium as gym
import torch
from src.evotorch.algorithms import Cosyne
from src.evotorch.logging import StdOutLogger
from src.evotorch.neuroevolution import GymNE
from custom_logger import CSVLogger


class RLTrainer:
    def __init__(self, env_name, save_weights_after_iter=2):
        self.env_name = env_name
        self.save_weights_after_iter = save_weights_after_iter
        self.metrics = {
            "best_eval": -100000,
            "worst_eval": 100000000,
            "iter": 1,
            "median_eval": -100000,
            "pop_best_eval": -100000,
            "mean_eval": -10000,
        }
        self.folder_name = f"./data/{env_name}"
        self.weights_path = f"{self.folder_name}/weights"
        self._initialize_directories()
        self.problem = self._create_problem(env_name)
        self.searcher = self._create_searcher()
        self.searcher.before_step_hook.append(self.check_metrics)
        #self.logger = StdOutLogger(self.searcher)
        self.csvlogg = CSVLogger(self.searcher)

    def _initialize_directories(self):
        """Create necessary directories for saving weights and videos."""
        self._check_and_create("data")
        self._check_and_create(self.folder_name)
        self._check_and_create(self.weights_path)

    @staticmethod
    def _check_and_create(path):
        """Create a directory if it does not exist."""
        if not os.path.exists(path):
            os.makedirs(path)

    def _create_problem(self, env_name):
        """Create the GymNE problem for the given environment."""
        def create_env():
            env = gym.make(env_name)
            return env

        return GymNE(
            env=create_env,
            network="Linear(obs_length, act_length)",
            network_args={'bias': False},
            num_actors='max',
            observation_normalization=True,
        )

    def _create_searcher(self):
        """Create the evolutionary searcher with Cosyne."""
        return Cosyne(
            self.problem,
            popsize=100,
            num_elites=1,
            tournament_size=10,
            mutation_stdev=0.3,
            mutation_probability=0.5,
            permute_all=True,
        )

    def evaluate_and_record(self, policy, save_path, iteration):
        """Evaluate the policy and save video with iteration number."""

        iteration_path = os.path.join(save_path, f"iteration_{iteration}")
        os.makedirs(iteration_path, exist_ok=True)

        env = gym.make(self.env_name, render_mode="rgb_array")
        env = gym.wrappers.RecordVideo(env, iteration_path, episode_trigger=lambda _=iteration: True)
        obs, _ = env.reset()
        obs = torch.from_numpy(obs)
        total_reward = 0
        done = False
        steps = 0

        while not done and steps <= 1000:
            action = torch.argmax(policy(obs)).item()
            obs, reward, done, _, _ = env.step(action)
            obs = torch.from_numpy(obs)
            total_reward += reward
            steps += 1

        env.close()

        return total_reward

    def save_weights_based_on_metrics(self, metric_name, sol_name, current_status):
        """Save weights based on specific metrics."""
        if metric_name in current_status and current_status[metric_name] >= self.metrics[metric_name]:
            current_score = current_status[metric_name]
            solution = current_status[sol_name]
            best_policy = self.problem.parameterize_net(solution.access_values())
            self.evaluate_and_record(best_policy, self.folder_name, current_status['iter'])

            path = f"{self.weights_path}/{metric_name}"
            self._check_and_create(path)
            file_name = f"{path}/iter_{current_status['iter']}_score_{current_score}.pth"
            torch.save(best_policy.state_dict(), file_name)
            self.metrics[metric_name] = current_score
            print(f"Saved weights to {file_name}")

    def check_metrics(self):
        """Check and update metrics after each iteration."""
        try:
            current_status = copy.deepcopy(self.searcher.status)
            if "iter" not in current_status:
                return
            iteration = current_status['iter']
            if iteration < self.save_weights_after_iter:
                return

            for metric_name, sol_name in [
                ("pop_best_eval", "pop_best"),
                ("best_eval", "best"),
                ("median_eval", "pop_best"),
                ("mean_eval", "pop_best"),
            ]:
                self.save_weights_based_on_metrics(metric_name, sol_name, current_status)

        except Exception as e:
            print(f"Exception occurred: {e}")

    def train(self, num_iterations):
        """Run the evolutionary search process."""
        print("Starting training...")
        self.searcher.run(num_iterations)
        population_center = self.searcher.status["best"]
        policy = self.problem.to_policy(population_center)
        return policy


if __name__ == "__main__":
    # Training setup
    trainer = RLTrainer(env_name="LunarLander-v3", save_weights_after_iter=2)
    final_policy = trainer.train(num_iterations=300)
    print("Training completed.")
