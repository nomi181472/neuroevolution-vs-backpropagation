from src.evotorch.algorithms import PGPE
from src.evotorch.logging import StdOutLogger
from src.evotorch.neuroevolution import GymNE
from src.evotorch.algorithms import Cosyne

# Specialized Problem class for RL
problem = GymNE(
    env_name="LunarLander-v3",
    # Linear policy
    network="Linear(obs_length, act_length)",
    observation_normalization=True,
    decrease_rewards_by=5.0,
    # Use all available CPU cores
    num_actors=4,

)

print("problem is created")
searcher = Cosyne(

                problem,
                popsize=6,
    **{
        "num_elites": 1,

        "tournament_size": 10,
        "mutation_stdev": 0.3,
        "mutation_probability": 0.5,
        "permute_all": True,
    }
            )
logger = StdOutLogger(searcher)
print("Algorithm added")
searcher.run(2)
print("Iteration completed")
population_center = searcher.status["best"]
policy = problem.to_policy(population_center)
print(problem.visualize(policy,))