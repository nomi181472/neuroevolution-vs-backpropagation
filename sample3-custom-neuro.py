from src.evotorch.algorithms import Cosyne
from src.evotorch.logging import StdOutLogger
from src.evotorch.neuroevolution import GymNE
import os
import gymnasium as gym
import copy
import torch
# Specialized Problem class for RL
# Function to evaluate the policy and save video with iteration number
def evaluate_and_record(policy, env_name, save_path, iteration):
    print("interacting with env")
    # Create a unique directory for each iteration
    iteration_path = os.path.join(save_path, f"iteration_{iteration}")
    os.makedirs(iteration_path, exist_ok=True)

    # Wrap the environment with RecordVideo
    env = gym.make(env_name, render_mode="rgb_array")
    env = gym.wrappers.RecordVideo(env, iteration_path, episode_trigger=lambda _: True)  # Always save this episode
    obs,_ = env.reset()
    obs=torch.from_numpy(obs)
    total_reward = 0
    done = False
    ii=0

    while not done and ii <=1000:

        action = policy(obs)
        action = torch.argmax(action).item()
        obs, reward, done, info,_ = env.step(action)
        obs=torch.from_numpy(obs)
        total_reward += reward
        ii=ii+1

    env.close()
    print("interaction closing..")
    return total_reward


metrics={
    "best_eval":-100000,
    "worst_eval":100000000,
    "iter":1,
    "median_eval":-100000,
    "pop_best_eval":-100000,
    "mean_eval":-10000
}
def check_and_create(path):
    print(f"creating:{path}")
    if not os.path.exists(path):
        os.mkdir(path)

save_weights_after_iter=2
env_name="LunarLander-v3"
check_and_create("data")
folder_name = f"./data/{env_name}"
check_and_create(folder_name)
weights_path = f"{folder_name}/weights"

def check_metrics():
    global searcher,weights_path,save_weights_after_iter,env_name

    try:
        current_status=copy.deepcopy(searcher.status)
        current_status = copy.deepcopy(searcher.status)
        if not "iter" in current_status:
            return
        iterr = current_status['iter']
        if not iterr >= save_weights_after_iter:
            return

        def save_weights_based_on_metrics(metric_name, sol_name):
            if metric_name in current_status and current_status[metric_name] >= metrics[metric_name]:
                current_score = current_status[metric_name]

                solution = current_status[sol_name]
                best_policy = problem.parameterize_net(solution.access_values())
                evaluate_and_record(best_policy,env_name,env_name,iterr)
                path = f"{weights_path}/{metric_name}"
                check_and_create(path)
                file_name = f"{path}/iter_{iterr}_score_{current_score}.pth"
                torch.save(best_policy.state_dict(), file_name)
                metrics[metric_name] = current_score
                print(f"saved {file_name}")

        m_name = "pop_best_eval"
        s_name = "pop_best"
        save_weights_based_on_metrics(m_name, s_name)
        m_name = "best_eval"
        s_name = "best"
        save_weights_based_on_metrics(m_name, s_name)
        m_name = "median_eval"
        s_name = "pop_best"
        save_weights_based_on_metrics(m_name, s_name)
        m_name = "mean_eval"
        s_name = "pop_best"
        save_weights_based_on_metrics(m_name, s_name)


    except Exception as e:
        print(f"Exception occur:{e}")
def get_problem(env_name,path_video,isRecord=False):
    def create_env():
        env:gym.Env
        if isRecord:
            env=gym.make(env_name)
        else:
            env = gym.make(env_name, render_mode="rgb_array",)
            # Wrap the environment with RecordVideo
            env = gym.wrappers.RecordVideo(env, path_video, episode_trigger=lambda x: x % 100 == 0)
        return env
    return GymNE(

        env=create_env,  # ame of the environment
        network="Linear(obs_length, act_length)",  # Linear policy that we defined earlier
        network_args={'bias': False},  # Linear policy should not use biases
        num_actors='max',  # Use 4 available CPUs. You can modify this value, or use 'max' to exploit all available CPUs
        observation_normalization=True,  # Observation normalization was not used in Lunar Lander experiments
    )


problem =get_problem(env_name,"./data2")



print("problem is created")
searcher = Cosyne(

                problem,
                popsize=20,
    **{
        "num_elites": 1,

        "tournament_size": 10,
        "mutation_stdev": 0.3,
        "mutation_probability": 0.5,
        "permute_all": True,
    }
            )
searcher.before_step_hook.append(check_metrics)
logger = StdOutLogger(searcher)
searcher.run(300)


population_center = searcher.status["best"]
policy = problem.to_policy(population_center)