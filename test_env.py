test_env = gym.make("LunarLanderContinuous-v3")

# Create an instance of the LinearPolicy network
policy = LinearPolicy(obs_length=test_env.observation_space.shape[0], act_length=test_env.action_space.shape[0], bias=False)

# Load the best weights into the policy network
policy.linear.weight.data = torch.tensor(best_model_weights['linear.weight'])
if 'linear.bias' in best_model_weights:
    policy.linear.bias.data = torch.tensor(best_model_weights['linear.bias'])

# Run a test episode and visualize the environment
obs = test_env.reset()
done = False
total_reward = 0

while not done:
    test_env.render()
    obs_tensor = torch.tensor(obs, dtype=torch.float32)
    action = policy(obs_tensor).detach().numpy()
    obs, reward, done, info = test_env.step(action)
    total_reward += reward

print("Total reward from test episode: ", total_reward)

test_env.close()