import os
import copy
import gymnasium as gym
import torch
from need import Need
from src.evotorch.algorithms import GeneticAlgorithm
from src.evotorch.logging import StdOutLogger
from src.evotorch.neuroevolution import GymNE
from custom_logger import CSVLogger
from evotorch.decorators import pass_info
from torch import nn
from ensemble_model import  Ensemble

import random
import numpy as np

def set_seed(seed):
    """Set the random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
# Set the seed at the beginning
SEED = 4
set_seed(SEED)

num_of_layers=2
neurons=8
@pass_info
class LinearPolicy(nn.Module):
    def __init__(
            self,
            obs_length: int,  # Number of observations from the environment
            act_length: int,  # Number of actions of the environment
            bias: bool = True,  # Whether the policy should use biases
            **kwargs  # Anything else that is passed
    ):
        super().__init__()  # Always call super init for nn Modules

        layers = []
        input_size = obs_length


        if num_of_layers == 1:
            # Directly map input to output without hidden layers
            layers.append(nn.Linear(input_size, act_length, bias=bias))
        else:
            # Add hidden layers
            for _ in range(num_of_layers - 1):
                layers.append(nn.Linear(input_size, neurons, bias=bias))
                layers.append(nn.ReLU())
                input_size = neurons

            # Add the final output layer
            layers.append(nn.Linear(input_size, act_length, bias=bias))

        # Register layers as a ModuleList
        self.layers = nn.ModuleList(layers)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        # Pass through each layer
        x = obs
        for layer in self.layers:
            x = layer(x)
        return x


class RLTrainer:
    def __init__(self, env_name, save_weights_after_iter=2,seed=4):
        self.env_name = env_name
        self.save_weights_after_iter = save_weights_after_iter
        self.seed = seed
        self.set_seed(self.seed)  # Apply the seed
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
       # self.problem.after_eval_hook.append(self.run_ensemble)
        self.searcher = self._create_searcher()
        self.searcher.before_step_hook.append(self.check_metrics)
        self.searcher.after_step_hook.append(self.run_current_top_models)
        self.logger = StdOutLogger(self.searcher)
        self.csvlogg = CSVLogger(self.searcher,self.folder_name)
        self.current_iteration=0
        self.current_performer={}

    @staticmethod
    def set_seed(seed):
        """Set the random seed for reproducibility."""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    def run_ensemble(self,status):
        evals={"ensembled_reward": 0,'best_current':0,'pop_best_current':0}
        try:

            models = self.searcher.ensembled_models

            if (len(models)) > 0:
                print(self.searcher.status)
                best = self.problem.parameterize_net(self.searcher.status["best"])
                print(best)
                pop_best = self.problem.parameterize_net(self.searcher.status["pop_best"])
                print(pop_best)

                policy = Ensemble(models)
                print(policy)

                evals['ensembled_reward'] = self.evaluate_and_record(policy, f"{self.folder_name}/ensemble_records", self.current_iteration)
                evals['best_current'] = self.evaluate_and_record(best, f"{self.folder_name}/best_records",
                                                                     self.current_iteration)
                evals['pop_best_current'] = self.evaluate_and_record(pop_best, f"{self.folder_name}/pop_best_current",
                                                                 self.current_iteration)

                #reward2=self.problem.visualize(policy)

                return evals
        except Exception as e:
            print(f"ensemble failed {e}")
        return evals
    def run_current_top_models(self,):

        evals={"ensembled_reward": 0,'best_current_reward':0,'pop_best_current_reward':0,}
        evals_steps = {"ensembled_steps": 0, 'best_current_steps': 0, 'pop_best_current_steps': 0, }
        try:
            if not (self.current_iteration>self.save_weights_after_iter):
                return  {**evals_steps,**evals}

            models = self.searcher.ensembled_models

            if (len(models)) > 0:

                print(self.current_performer.keys())
                if "best" in self.current_performer:
                    best = self.current_performer["best"]
                    evals['best_current_reward'], evals['best_current_steps'] = self.evaluate_and_record(best,
                                                                                                  f"{self.folder_name}/best_records",
                                                                                                  self.current_iteration)

                if "pop_best" in self.current_performer:
                    pop_best = self.current_performer["pop_best"]
                    evals['pop_best_current_reward'], evals['pop_best_current_steps'] = self.evaluate_and_record(pop_best,
                                                                                                          f"{self.folder_name}/pop_best_current",
                                                                                                          self.current_iteration)

                policy = Ensemble(models)
                evals['ensembled_reward'],evals['ensembled_steps'] = self.evaluate_and_record(policy, f"{self.folder_name}/ensemble_records", self.current_iteration)


                #reward2=self.problem.visualize(policy)

                return {**evals_steps,**evals}
        except Exception as e:
            print(f"ensemble failed {e}")
        return {**evals_steps,**evals}
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
            network=LinearPolicy,
            network_args={'bias': False},
            num_actors='max',
            observation_normalization=True,
        )

    def _create_searcher(self):
        return GeneticAlgorithm(
            self.problem,

            operators=[],  # Define operators if needed, otherwise leave empty
            popsize=100,  # Population size
            elitist=True,  # Use elitism to retain the best solutions
            re_evaluate=True,  # Re-evaluate individuals after each generation
            re_evaluate_parents_first=True,  # Re-evaluate parents first if specified
            _allow_empty_operator_list=True,  # Allow empty operator list

        )
        # return Need(
        #     self.problem,
        #     popsize=100,
        #     num_elites=4,
        #     tournament_size=10,
        #     mutation_stdev=0.3,
        #     mutation_probability=0.5,
        #     permute_all=True,
        #     n_ensemble=5
        #
        # )
    @torch.no_grad()
    def evaluate_and_record(self, policy, save_path, iteration)->float:
        """Evaluate the policy and save video with iteration number."""

        iteration_path = os.path.join(f"{save_path}", f"video_{iteration}")
        os.makedirs(iteration_path, exist_ok=True)

        env = gym.make(self.env_name, render_mode="rgb_array")
        env = gym.wrappers.RecordVideo(env, iteration_path, episode_trigger=lambda _=iteration: True)
        total_rewards=[]
        total_steps=[]
        current_seed:int=11
        while current_seed >0:
            current_seed=current_seed-1
            obs, _ = env.reset(seed=current_seed)
            obs = torch.from_numpy(obs)
            total_reward = 0
            done = False
            steps = 0

            while not done and steps <= 1700:
                action = torch.argmax(policy(obs)).item()
                obs, reward, done, _, _ = env.step(action)
                obs = torch.from_numpy(obs)
                total_reward += reward
                steps += 1
            total_steps.append(steps)
            total_rewards.append(total_reward)
        env.close()
        avg_reward=sum(total_rewards)/(len(total_rewards)+0.00000001)
        avg_steps=sum(total_steps)/(len(total_steps)+0.00000001)
        return avg_reward,avg_steps

    def save_weights_based_on_metrics(self, metric_name, sol_name, current_status):
        """Save weights based on specific metrics."""
        if metric_name in current_status and current_status[metric_name] > self.metrics[metric_name]:
            current_score = current_status[metric_name]
            solution = current_status[sol_name]
            current_policy = self.problem.parameterize_net(solution.access_values())
            #self.evaluate_and_record(best_policy, self.folder_name, current_status['iter'])
            self.current_performer[sol_name]=copy.deepcopy(current_policy)
            path = f"{self.weights_path}/{metric_name}"
            self._check_and_create(path)
            file_name = f"{path}/iter_{current_status['iter']}_score_{current_score}.pth"
            torch.save(current_policy.state_dict(), file_name)
            self.metrics[metric_name] = current_score
            print(f"Saved weights to {file_name}")

    def check_metrics(self):
        iteration=0
        """Check and update metrics after each iteration."""
        try:
            current_status = copy.deepcopy(self.searcher.status)
            if "iter" not in current_status:
                return
            iteration = current_status['iter']
            self.current_iteration=iteration
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
    trainer = RLTrainer(env_name="LunarLander-v3", save_weights_after_iter=100)
    final_policy = trainer.train(num_iterations=300)
    print("Training completed.")